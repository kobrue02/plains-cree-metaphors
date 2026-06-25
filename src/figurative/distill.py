"""
Cross-lingual adaptation of a figurative classifier.

Three modes:

  align      — cosine loss between [CLS] of Cree and English.  No label
               assumption.  Only the encoder updates; the classification
               head is frozen.

  binary_kl  — KL divergence between the model's predictions on the Cree
               text and (teacher) predictions on its English translation,
               collapsed to binary (literal vs. figurative).  Temperature
               scaling hedges against cross-linguistic type mismatches.

  clkd       — Cross-Lingual Knowledge Distillation.  A dedicated frozen
               English teacher (e.g. DeBERTa-v3 fine-tuned on VUA20+MAGPIE)
               predicts the full soft-label distribution on the English side
               of the parallel corpus; the student (XLM TLM checkpoint) is
               trained on the Cree side to match via KL divergence.
               Optionally the first N student layers are frozen to preserve
               the cross-lingual representations built during TLM pre-training.
"""

from __future__ import annotations
import os
import re
from dataclasses import dataclass

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

try:
    import wandb as _wandb
except ImportError:
    _wandb = None


@dataclass
class DistillConfig:
    # Student: TLM-adapted XLM checkpoint (or figurative checkpoint for align/binary_kl)
    checkpoint:  str  = "KonradBRG/xlm-mlm-100-1280-plains-cree-en-tlm"
    corpus_file: str  = "data/bloomfield_texts_sentences.csv"

    # "align" | "binary_kl" | "clkd"
    mode:        str  = "clkd"

    # ── CLKD-specific ─────────────────────────────────────────────────────────
    # Frozen English teacher — required for mode="clkd"
    teacher_checkpoint: str | None = None
    # Freeze the first N transformer layers of the student (0 = train all)
    freeze_n_layers:    int        = 0
    # Output classes — must match the teacher's num_labels
    num_labels:         int        = 4

    # ── Training ──────────────────────────────────────────────────────────────
    batch_size:    int   = 16
    epochs:        int   = 10
    learning_rate: float = 5e-6
    warmup_ratio:  float = 0.1
    temperature:   float = 2.0

    # ── align / binary_kl only ────────────────────────────────────────────────
    freeze_head:   bool  = True

    # ── Output ────────────────────────────────────────────────────────────────
    hub_model_id:  str | None = None
    output_dir:    str  = "data/figurative/distilled"
    wandb_project: str | None = None


# ── Datasets ──────────────────────────────────────────────────────────────────

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


# ── Loss functions ─────────────────────────────────────────────────────────────

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


# ── Layer freezing ─────────────────────────────────────────────────────────────

def _freeze_n_layers(model, n: int) -> None:
    """Freeze embeddings and the first *n* transformer layers.

    Matches generically across architectures: XLM uses ``attentions.{i}.``,
    XLM-R/DeBERTa use ``encoder.layer.{i}.``.  Any parameter whose first
    integer path segment is < n is frozen, as are all embedding layers.
    """
    for name, param in model.named_parameters():
        if re.search(r'\bembeddings?\b|\blangEmbeddings\b|\bposition_embeddings\b', name):
            param.requires_grad = False
            continue
        m = re.search(r'\.(\d+)\.', name)
        if m and int(m.group(1)) < n:
            param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[distill] frozen first {n} layers — trainable params: {trainable:,}")


# ── Main entry point ───────────────────────────────────────────────────────────

def distill(config: DistillConfig) -> str:
    """Run cross-lingual adaptation and return the output directory."""
    if config.mode == "clkd":
        return _distill_clkd(config)
    return _distill_self(config)


# ── CLKD ──────────────────────────────────────────────────────────────────────

def _distill_clkd(config: DistillConfig) -> str:
    if not config.teacher_checkpoint:
        raise ValueError("clkd mode requires teacher_checkpoint to be set.")

    os.makedirs(config.output_dir, exist_ok=True)
    if config.wandb_project:
        os.environ["WANDB_PROJECT"] = config.wandb_project

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Teacher: frozen English figurative classifier
    print(f"[distill] loading teacher : {config.teacher_checkpoint}")
    _t_use_fast = "deberta-v3" not in config.teacher_checkpoint.lower()
    teacher_tokenizer = AutoTokenizer.from_pretrained(config.teacher_checkpoint, use_fast=_t_use_fast)
    teacher = AutoModelForSequenceClassification.from_pretrained(
        config.teacher_checkpoint, torch_dtype=torch.float32,
    )
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"[distill] teacher frozen  — {sum(p.numel() for p in teacher.parameters()):,} params")

    # Student: TLM checkpoint loaded as sequence classifier
    print(f"[distill] loading student : {config.checkpoint}")
    student_tokenizer = AutoTokenizer.from_pretrained(config.checkpoint)
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

    if _wandb and config.wandb_project:
        _wandb.init(
            project=config.wandb_project,
            config={
                "mode":             config.mode,
                "student":          config.checkpoint,
                "teacher":          config.teacher_checkpoint,
                "freeze_n_layers":  config.freeze_n_layers,
                "epochs":           config.epochs,
                "batch_size":       config.batch_size,
                "learning_rate":    config.learning_rate,
                "temperature":      config.temperature,
                "corpus_size":      None,
            },
        )

    df = (
        pd.read_csv(config.corpus_file, encoding="utf-8-sig")
        .dropna(subset=["text_cree", "text_en"])
    )
    print(f"[distill] mode=clkd  corpus={len(df):,} pairs  epochs={config.epochs}  "
          f"lr={config.learning_rate}  T={config.temperature}  "
          f"freeze_n={config.freeze_n_layers}")
    if _wandb and _wandb.run:
        _wandb.config.update({"corpus_size": len(df)}, allow_val_change=True)

    loader = DataLoader(
        CLKDDataset(df, teacher_tokenizer, student_tokenizer),
        batch_size=config.batch_size,
        shuffle=True,
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
        for batch in loader:
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
            if _wandb and _wandb.run:
                _wandb.log({"train/loss_step": loss.item(),
                            "train/lr": scheduler.get_last_lr()[0]}, step=global_step)

        avg_loss = epoch_loss / len(loader)
        print(f"[distill] epoch {epoch + 1}/{config.epochs}  loss={avg_loss:.4f}")
        if _wandb and _wandb.run:
            _wandb.log({"train/loss_epoch": avg_loss, "epoch": epoch + 1}, step=global_step)

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


# ── Self-distillation (align / binary_kl) ─────────────────────────────────────

def _distill_self(config: DistillConfig) -> str:
    os.makedirs(config.output_dir, exist_ok=True)
    if config.wandb_project:
        os.environ["WANDB_PROJECT"] = config.wandb_project

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(config.checkpoint)
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
        pd.read_csv(config.corpus_file, encoding="utf-8-sig")
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
        for batch in loader:
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
