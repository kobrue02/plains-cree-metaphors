"""
Cross-lingual adaptation of a figurative classifier using Cree-English parallel
sentences.  Two modes:

  align      — cosine loss between [CLS] of Cree and English.  No label
               assumption whatsoever.  Only the encoder updates; the
               classification head is frozen.

  binary_kl  — KL divergence between the model's predictions on the Cree
               text and (teacher) predictions on its English translation,
               collapsed to binary (literal vs. figurative).  Temperature
               scaling hedges against cross-linguistic type mismatches
               (the teacher is made deliberately soft).
"""

from __future__ import annotations
import os
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


@dataclass
class DistillConfig:
    # Starting checkpoint — should be a figurative-fine-tuned model
    checkpoint:    str   = "KonradBRG/xlm-r-plains-cree-en-tlm-figurative"
    corpus_file:   str   = "data/bloomfield_texts_sentences.csv"

    # "align" | "binary_kl"
    mode:          str   = "align"

    batch_size:    int   = 16
    epochs:        int   = 10
    # Lower than fine-tuning — we nudge gently, not retrain
    learning_rate: float = 5e-6
    warmup_ratio:  float = 0.1

    # binary_kl only: flatten teacher distribution to account for
    # genuine cross-linguistic type differences (higher T = softer)
    temperature:   float = 2.0

    # Freeze the classification head (recommended for both modes)
    freeze_head:   bool  = True

    hub_model_id:  str | None = None
    output_dir:    str   = "data/figurative/distilled"
    wandb_project: str | None = None


class ParallelDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 128):
        self.cree    = df["text_cree"].tolist()
        self.english = df["text_en"].tolist()
        self.tok     = tokenizer
        self.max_len = max_length

    def _enc(self, text: str) -> dict:
        return self.tok(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
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


def _align_loss(h_cree: torch.Tensor, h_en: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(h_cree, h_en, dim=-1).mean()


def _binary_kl_loss(
    cree_logits: torch.Tensor,
    en_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    def to_binary(logits: torch.Tensor, T: float) -> torch.Tensor:
        probs = F.softmax(logits / T, dim=-1)          # (B, n_classes)
        p_lit = probs[:, 0:1]
        p_fig = 1.0 - p_lit
        return torch.cat([p_lit, p_fig], dim=-1)       # (B, 2)

    en_soft  = to_binary(en_logits.detach(), temperature)     # teacher
    cree_log = torch.log(to_binary(cree_logits, 1.0) + 1e-8)  # student
    return F.kl_div(cree_log, en_soft, reduction="batchmean")


def distill(config: DistillConfig) -> str:
    """Run cross-lingual adaptation and return the output directory."""
    os.makedirs(config.output_dir, exist_ok=True)

    if config.wandb_project:
        os.environ["WANDB_PROJECT"] = config.wandb_project

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(config.checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(config.checkpoint)
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
                "input_ids":           batch["cree_input_ids"].to(device),
                "attention_mask":      batch["cree_attention_mask"].to(device),
                "output_hidden_states": need_hidden,
            }
            en_kwargs = {
                "input_ids":           batch["en_input_ids"].to(device),
                "attention_mask":      batch["en_attention_mask"].to(device),
                "output_hidden_states": need_hidden,
            }

            if config.mode == "align":
                cree_out = model(**cree_kwargs)
                with torch.no_grad():
                    en_out = model(**en_kwargs)
                # [CLS] token from last hidden layer
                h_cree = cree_out.hidden_states[-1][:, 0, :]
                h_en   = en_out.hidden_states[-1][:, 0, :]
                loss = _align_loss(h_cree, h_en)

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
