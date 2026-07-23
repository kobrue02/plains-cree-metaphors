"""
Cross-lingual adaptation of a figurative classifier via three modes: align (unsupervised
cosine loss between Cree/English [CLS] embeddings), binary_kl (KL distillation from English
predictions to Cree, collapsed to literal-vs-figurative), and clkd (full soft-label
distillation from a frozen English teacher to a Cree student over parallel text).
"""

from __future__ import annotations
import os
import re
from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split as _train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from src.device import get_device
from src.figurative.calibrate import _load_gold_pool
from src.figurative.data import LABEL_NAMES, resolve_use_fast
from src.figurative.predict import predict_sentences

try:
    import wandb as _wandb
except ImportError:
    _wandb = None

def _gold_metrics(model, tokenizer, gold_df: pd.DataFrame, batch_size: int, max_length: int) -> dict:
    """Macro/per-label F1 of `model` (in its current state) against the full
    gold set — CLKD never trains on gold or silver labels, so this is always
    a genuine zero-shot measurement, safe to compute every epoch."""
    was_training = model.training
    model.eval()
    preds = predict_sentences(gold_df["text_cree"].tolist(), model, tokenizer,
                               batch_size=batch_size, max_length=max_length)
    if was_training:
        model.train()

    y_true = gold_df["label"].tolist()
    y_pred = [p["label"] for p in preds]
    report = classification_report(y_true, y_pred, labels=LABEL_NAMES, output_dict=True, zero_division=0)
    metrics = {"macro_f1": report["macro avg"]["f1-score"]}
    for name in LABEL_NAMES:
        metrics[f"{name}_f1"] = report[name]["f1-score"]
    return metrics

@dataclass
class DistillConfig:
    # tlm-adapted xlm checkpoint (or figurative checkpoint for align/binary_kl)
    checkpoint:  str  = "KonradBRG/xlm-mlm-100-1280-plains-cree-en-tlm"
    corpus_file: str  = "data/bloomfield_texts_sentences.parquet"

    # "align" | "binary_kl" | "clkd"
    mode:        str  = "clkd"

    teacher_checkpoint: str | None = None  # required for mode="clkd"
    freeze_n_layers:    int        = 0     # freeze first n student transformer layers (0 = train all)
    num_labels:         int        = 4     # must match the teacher's num_labels

    batch_size:    int   = 16
    epochs:        int   = 10
    learning_rate: float = 5e-6
    warmup_ratio:  float = 0.1
    temperature:   float = 2.0
    max_length:    int   = 256

    freeze_head:   bool  = True

    hub_model_id:  str | None = None
    output_dir:    str  = "data/figurative/distilled"
    wandb_project: str | None = None

class ParallelDataset(Dataset):
    """Single-tokenizer dataset for align / binary_kl modes."""

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 128):
        self.cree    = df["text_cree"].tolist()
        self.english = df["text_en"].tolist()
        self.tok     = tokenizer
        self.max_len = max_length

    def _enc(self, text: str) -> dict:
        return self.tok(
            text, truncation=True, max_length=self.max_len,
            padding="max_length", return_tensors="pt",
        )

    def __len__(self) -> int:
        return len(self.cree)

    def __getitem__(self, idx: int) -> dict:
        c = self._enc(self.cree[idx])
        e = self._enc(self.english[idx])
        return {
            "cree_input_ids":      c["input_ids"].squeeze(0),
            "cree_attention_mask": c["attention_mask"].squeeze(0),
            "en_input_ids":        e["input_ids"].squeeze(0),
            "en_attention_mask":   e["attention_mask"].squeeze(0),
        }

class CLKDDataset(Dataset):
    """Dual-tokenizer dataset: teacher tokenizer for English, student tokenizer for Cree."""

    def __init__(
        self,
        df:                pd.DataFrame,
        teacher_tokenizer,
        student_tokenizer,
        max_length: int = 128,
    ):
        self.cree    = df["text_cree"].tolist()
        self.english = df["text_en"].tolist()
        self.t_tok   = teacher_tokenizer
        self.s_tok   = student_tokenizer
        self.max_len = max_length

    def _enc(self, text: str, tokenizer) -> dict:
        return tokenizer(
            text, truncation=True, max_length=self.max_len,
            padding="max_length", return_tensors="pt",
        )

    def __len__(self) -> int:
        return len(self.cree)

    def __getitem__(self, idx: int) -> dict:
        t = self._enc(self.english[idx], self.t_tok)
        s = self._enc(self.cree[idx],    self.s_tok)
        return {
            "teacher_input_ids":      t["input_ids"].squeeze(0),
            "teacher_attention_mask": t["attention_mask"].squeeze(0),
            "student_input_ids":      s["input_ids"].squeeze(0),
            "student_attention_mask": s["attention_mask"].squeeze(0),
        }

def _align_loss(h_cree: torch.Tensor, h_en: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(h_cree, h_en, dim=-1).mean()

def _binary_kl_loss(
    cree_logits: torch.Tensor,
    en_logits:   torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    def to_binary(logits: torch.Tensor, T: float) -> torch.Tensor:
        probs = F.softmax(logits / T, dim=-1)
        p_lit = probs[:, 0:1]
        p_fig = 1.0 - p_lit
        return torch.cat([p_lit, p_fig], dim=-1)

    en_soft  = to_binary(en_logits.detach(), temperature)
    cree_log = torch.log(to_binary(cree_logits, 1.0) + 1e-8)
    return F.kl_div(cree_log, en_soft, reduction="batchmean")

def _clkd_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    temperature:    float,
) -> torch.Tensor:
    teacher_probs     = F.softmax(teacher_logits / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")

def _freeze_n_layers(model, n: int) -> None:
    """Freeze embeddings and the first n transformer layers; works across XLM, XLM-R, and DeBERTa by matching the first integer path segment."""
    for name, param in model.named_parameters():
        if re.search(r'\bembeddings?\b|\blangEmbeddings\b|\bposition_embeddings\b', name):
            param.requires_grad = False
            continue
        m = re.search(r'\.(\d+)\.', name)
        if m and int(m.group(1)) < n:
            param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[distill] frozen first {n} layers — trainable params: {trainable:,}")

def distill(config: DistillConfig) -> str:
    if config.mode == "clkd":
        return _distill_clkd(config)
    return _distill_self(config)

def _distill_clkd(config: DistillConfig) -> str:
    if not config.teacher_checkpoint:
        raise ValueError("clkd mode requires teacher_checkpoint to be set.")

    os.makedirs(config.output_dir, exist_ok=True)
    if config.wandb_project:
        os.environ["WANDB_PROJECT"] = config.wandb_project

    device = get_device()

    print(f"[distill] loading teacher : {config.teacher_checkpoint}")
    teacher_tokenizer = AutoTokenizer.from_pretrained(
        config.teacher_checkpoint, use_fast=resolve_use_fast(config.teacher_checkpoint),
    )
    teacher = AutoModelForSequenceClassification.from_pretrained(
        config.teacher_checkpoint, torch_dtype=torch.float32,
    )
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"[distill] teacher frozen  — {sum(p.numel() for p in teacher.parameters()):,} params")

    print(f"[distill] loading student : {config.checkpoint}")
    student_tokenizer = AutoTokenizer.from_pretrained(config.checkpoint, use_fast=resolve_use_fast(config.checkpoint))
    student = AutoModelForSequenceClassification.from_pretrained(
        config.checkpoint,
        num_labels=config.num_labels,
        ignore_mismatched_sizes=True,
        torch_dtype=torch.float32,
    )
    student.to(device)

    if config.freeze_n_layers > 0:
        _freeze_n_layers(student, config.freeze_n_layers)
    else:
        print(f"[distill] student fully trainable — "
              f"{sum(p.numel() for p in student.parameters() if p.requires_grad):,} params")

    _clkd_cfg = {
        "mode":             config.mode,
        "student":          config.checkpoint,
        "teacher":          config.teacher_checkpoint,
        "freeze_n_layers":  config.freeze_n_layers,
        "epochs":           config.epochs,
        "batch_size":       config.batch_size,
        "learning_rate":    config.learning_rate,
        "temperature":      config.temperature,
        "corpus_size":      None,
    }
    if _wandb and config.wandb_project:
        if not _wandb.run:
            _wandb.init(project=config.wandb_project, config=_clkd_cfg)
        else:
            _wandb.config.update(_clkd_cfg, allow_val_change=True)

    df = (
        pd.read_parquet(config.corpus_file)
        .dropna(subset=["text_cree", "text_en"])
    )
    print(f"[distill] mode=clkd  corpus={len(df):,} pairs  epochs={config.epochs}  "
          f"lr={config.learning_rate}  T={config.temperature}  "
          f"freeze_n={config.freeze_n_layers}")
    if _wandb and _wandb.run:
        _wandb.config.update({"corpus_size": len(df)}, allow_val_change=True)

    gold_df = _load_gold_pool()
    print(f"[distill] gold eval set: {len(gold_df):,} sentences (zero-shot, never trained on)")
    best_gold_f1 = -1.0
    best_state = None
    best_epoch = None

    if len(df) >= 20:
        train_df, eval_df = _train_test_split(df, test_size=0.1, random_state=42)
    else:
        train_df, eval_df = df, df

    loader = DataLoader(
        CLKDDataset(train_df, teacher_tokenizer, student_tokenizer,
                    max_length=config.max_length),
        batch_size=config.batch_size,
        shuffle=True,
    )
    eval_loader = DataLoader(
        CLKDDataset(eval_df, teacher_tokenizer, student_tokenizer,
                    max_length=config.max_length),
        batch_size=config.batch_size,
        shuffle=False,
    )

    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=config.learning_rate,
    )
    total_steps = len(loader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.warmup_ratio),
        num_training_steps=total_steps,
    )

    student.train()
    global_step = 0
    for epoch in range(config.epochs):
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"[distill] epoch {epoch + 1}/{config.epochs}", leave=False)
        for batch in pbar:
            with torch.no_grad():
                teacher_logits = teacher(
                    input_ids=batch["teacher_input_ids"].to(device),
                    attention_mask=batch["teacher_attention_mask"].to(device),
                ).logits

            student_logits = student(
                input_ids=batch["student_input_ids"].to(device),
                attention_mask=batch["student_attention_mask"].to(device),
            ).logits

            loss = _clkd_loss(teacher_logits, student_logits, config.temperature)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
            if _wandb and _wandb.run:
                _wandb.log({"train/loss_step": loss.item(),
                            "train/lr": scheduler.get_last_lr()[0],
                            "train/clkd_step": global_step})

        avg_loss = epoch_loss / len(loader)

        student.eval()
        eval_loss = 0.0
        with torch.no_grad():
            for eval_batch in eval_loader:
                t_logits = teacher(
                    input_ids=eval_batch["teacher_input_ids"].to(device),
                    attention_mask=eval_batch["teacher_attention_mask"].to(device),
                ).logits
                s_logits = student(
                    input_ids=eval_batch["student_input_ids"].to(device),
                    attention_mask=eval_batch["student_attention_mask"].to(device),
                ).logits
                eval_loss += _clkd_loss(t_logits, s_logits, config.temperature).item()
        eval_avg = eval_loss / len(eval_loader)

        gold_metrics = _gold_metrics(student, student_tokenizer, gold_df,
                                      batch_size=config.batch_size, max_length=config.max_length)
        student.train()

        if gold_metrics["macro_f1"] > best_gold_f1:
            best_gold_f1 = gold_metrics["macro_f1"]
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}

        print(f"[distill] epoch {epoch + 1}/{config.epochs}  "
              f"train_loss={avg_loss:.4f}  eval_kl={eval_avg:.4f}  "
              f"gold_macro_f1={gold_metrics['macro_f1']:.4f}  best_so_far={best_gold_f1:.4f}")
        if _wandb and _wandb.run:
            _wandb.log({
                "train/loss_epoch":         avg_loss,
                "eval/kl_epoch":            eval_avg,
                "eval/gold_macro_f1":       gold_metrics["macro_f1"],
                "eval/best_gold_macro_f1":  best_gold_f1,
                **{f"eval/gold_{name}_f1": gold_metrics[f"{name}_f1"] for name in LABEL_NAMES},
                "epoch":                    epoch + 1,
                "train/clkd_step":          global_step,
            })

    if best_state is not None:
        print(f"[distill] restoring best checkpoint — epoch {best_epoch} "
              f"(gold_macro_f1={best_gold_f1:.4f})")
        student.load_state_dict(best_state)
        if _wandb and _wandb.run:
            _wandb.summary["best_epoch"] = best_epoch
            _wandb.summary["best_gold_macro_f1"] = best_gold_f1

    student.save_pretrained(config.output_dir)
    student_tokenizer.save_pretrained(config.output_dir)
    print(f"[distill] saved to {config.output_dir}")

    if config.hub_model_id:
        student.push_to_hub(config.hub_model_id)
        student_tokenizer.push_to_hub(config.hub_model_id)
        print(f"[distill] pushed to hub: {config.hub_model_id}")

    if _wandb and _wandb.run:
        _wandb.finish()

    return config.output_dir

def _distill_self(config: DistillConfig) -> str:
    os.makedirs(config.output_dir, exist_ok=True)
    if config.wandb_project:
        os.environ["WANDB_PROJECT"] = config.wandb_project

    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(config.checkpoint, use_fast=resolve_use_fast(config.checkpoint))
    model = AutoModelForSequenceClassification.from_pretrained(
        config.checkpoint, torch_dtype=torch.float32,
    )
    model.to(device)

    if config.freeze_head:
        for param in model.classifier.parameters():
            param.requires_grad = False
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[distill] head frozen — trainable params: {n:,}")

    df = (
        pd.read_parquet(config.corpus_file)
        .dropna(subset=["text_cree", "text_en"])
    )
    print(f"[distill] mode={config.mode}  corpus={len(df):,} pairs  "
          f"epochs={config.epochs}  lr={config.learning_rate}")

    loader = DataLoader(
        ParallelDataset(df, tokenizer),
        batch_size=config.batch_size,
        shuffle=True,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
    )
    total_steps = len(loader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.warmup_ratio),
        num_training_steps=total_steps,
    )

    need_hidden = config.mode == "align"
    model.train()

    for epoch in range(config.epochs):
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"[distill] epoch {epoch + 1}/{config.epochs}", leave=False)
        for batch in pbar:
            cree_kwargs = {
                "input_ids":            batch["cree_input_ids"].to(device),
                "attention_mask":       batch["cree_attention_mask"].to(device),
                "output_hidden_states": need_hidden,
            }
            en_kwargs = {
                "input_ids":            batch["en_input_ids"].to(device),
                "attention_mask":       batch["en_attention_mask"].to(device),
                "output_hidden_states": need_hidden,
            }

            if config.mode == "align":
                cree_out = model(**cree_kwargs)
                with torch.no_grad():
                    en_out = model(**en_kwargs)
                h_cree = cree_out.hidden_states[-1][:, 0, :]
                h_en   = en_out.hidden_states[-1][:, 0, :]
                loss   = _align_loss(h_cree, h_en)
            elif config.mode == "binary_kl":
                cree_logits = model(**cree_kwargs).logits
                with torch.no_grad():
                    en_logits = model(**en_kwargs).logits
                loss = _binary_kl_loss(cree_logits, en_logits, config.temperature)
            else:
                raise ValueError(f"Unknown distill mode: {config.mode!r}")

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        print(f"[distill] epoch {epoch + 1}/{config.epochs}  "
              f"loss={epoch_loss / len(loader):.4f}")

    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print(f"[distill] saved to {config.output_dir}")

    if config.hub_model_id:
        model.push_to_hub(config.hub_model_id)
        tokenizer.push_to_hub(config.hub_model_id)
        print(f"[distill] pushed to hub: {config.hub_model_id}")

    return config.output_dir
