"""
Fine-tune a TLM-adapted encoder on the full-pool LLM (silver) labels, evaluating
against the same fixed 228-sentence gold set CLKD is scored against every epoch
— a direct silver-SFT vs. CLKD comparison on identical held-out data, using the
same "best checkpoint by gold macro F1" selection CLKD uses (see distill.py's
best_gold_f1 tracking). Gold sentences are always excluded from training (see
calibrate.py's _load_gold_pool).
"""

from __future__ import annotations
import os
from collections import Counter
from dataclasses import dataclass

import pandas as pd
import torch
from torch.utils.data import Sampler, WeightedRandomSampler
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

from src.device import get_trainer_device_kwargs
from src.figurative.calibrate import _read_labeled, _to_records, _load_gold_pool, LABEL2ID
from src.figurative.data import FigurativeDataset, resolve_use_fast, LABEL_NAMES, NUM_LABELS
from src.figurative.distill import _freeze_n_layers
from src.figurative.evaluate import compute_metrics
from src.figurative.config import FigurativeConfig
from src.figurative.hierarchical import HierarchicalFigurativeConfig, HierarchicalFigurativeModel


class _BalancedSamplerTrainer(Trainer):
    """Draws each training batch via a WeightedRandomSampler (inverse class
    frequency) instead of the default RandomSampler. At ~18:1 literal:figurative
    imbalance (idiom alone ~55:1), plain random batches of 16 rarely contained a
    minority-class example, so loss reweighting alone produced sparse, spiky
    minority-class gradients rather than a steady learning signal — balancing
    batch composition directly fixes the exposure problem loss weighting can't."""

    def _get_train_sampler(self, train_dataset=None) -> Sampler:
        train_dataset = train_dataset if train_dataset is not None else self.train_dataset
        labels = [int(train_dataset[i]["labels"]) for i in range(len(train_dataset))]
        counts = Counter(labels)
        sample_weights = [1.0 / counts[label] for label in labels]
        return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


@dataclass
class SilverSFTConfig:
    checkpoint:      str                                     # TLM-adapted encoder to start from
    silver_file:     str        = "data/figurative/deepseek_labels_qwen_qwen3.5-122b-a10b.parquet"
    label_col:       str        = "deepseek_label"
    output_dir:      str        = "data/figurative/silver_sft"
    hub_model_id:    str | None = None
    wandb_project:   str | None = None
    epochs:          int        = 5
    batch_size:      int        = 16
    learning_rate:   float      = 2e-5
    max_length:      int        = 128
    freeze_n_layers: int        = 0     # freeze embeddings + first n transformer layers (0 = train all)
    hierarchical:    bool       = False  # binary (literal/figurative) + conditional 3-way head, see hierarchical.py


def _load_silver_records(config: SilverSFTConfig) -> list[dict]:
    """The full silver pool (all Qwen-labelled sentences), minus any overlap with
    gold. No class balancing/downsampling here — _BalancedSamplerTrainer handles
    the literal/figurative imbalance via a weighted batch sampler instead."""
    df = _read_labeled(config.silver_file, config.label_col)

    gold_texts = set(_load_gold_pool()["text_cree"])
    n_before = len(df)
    df = df[~df["text_cree"].isin(gold_texts)]
    if len(df) != n_before:
        print(f"[silver_sft] excluded {n_before - len(df)} sentences overlapping the gold set")

    records = _to_records(df)
    print(f"[silver_sft] {len(records):,} sentences — {df['label'].value_counts().to_dict()}")
    return records


def train_on_silver(config: SilverSFTConfig) -> str:
    os.makedirs(config.output_dir, exist_ok=True)
    if config.wandb_project:
        os.environ["WANDB_PROJECT"] = config.wandb_project

    train_recs = _load_silver_records(config)
    eval_recs  = _to_records(_load_gold_pool())
    print(f"[silver_sft] train={len(train_recs)} (silver)  eval={len(eval_recs)} (gold, zero-shot every epoch)")

    tokenizer = AutoTokenizer.from_pretrained(config.checkpoint, use_fast=resolve_use_fast(config.checkpoint))
    ds_config = FigurativeConfig(encoder=config.checkpoint,
                                 max_length=config.max_length,
                                 batch_size=config.batch_size)
    train_ds = FigurativeDataset(train_recs, tokenizer, ds_config)
    eval_ds  = FigurativeDataset(eval_recs,  tokenizer, ds_config)

    if config.hierarchical:
        model_config = HierarchicalFigurativeConfig(base_checkpoint=config.checkpoint)
        model = HierarchicalFigurativeModel(model_config)
        print(f"[silver_sft] hierarchical head: binary (literal/figurative) "
              f"+ conditional 3-way (idiom/metaphor/simile)")
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            config.checkpoint,
            num_labels=NUM_LABELS,
            ignore_mismatched_sizes=True,   # bare TLM checkpoint has no classification head yet
            id2label={i: l for i, l in enumerate(LABEL_NAMES)},
            label2id=LABEL2ID,
            torch_dtype=torch.float32,
        )

    if config.freeze_n_layers > 0:
        _freeze_n_layers(model, config.freeze_n_layers)
    else:
        print(f"[silver_sft] fully trainable — "
              f"{sum(p.numel() for p in model.parameters()):,} params")

    hub_kwargs = ({"push_to_hub": True, "hub_model_id": config.hub_model_id}
                  if config.hub_model_id else {})

    args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size * 2,
        learning_rate=config.learning_rate,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        save_only_model=True,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=10,
        report_to="wandb" if config.wandb_project else "none",
        **get_trainer_device_kwargs(),
        **hub_kwargs,
    )

    trainer = _BalancedSamplerTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print(f"[silver_sft] starting from {config.checkpoint}  "
          f"({len(train_recs):,} silver-labeled train sentences)")
    trainer.train()
    tokenizer.save_pretrained(config.output_dir)
    trainer.save_model(config.output_dir)
    print(f"[silver_sft] saved → {config.output_dir}")

    # load_best_model_at_end has already restored the checkpoint with the best
    # gold macro F1 across epochs — this re-evaluate just prints that number.
    final_metrics = trainer.evaluate()
    print(f"[silver_sft] best gold macro_f1={final_metrics['eval_macro_f1']:.4f}  "
          + "  ".join(f"{l}={final_metrics[f'eval_{l}_f1']:.2f}" for l in LABEL_NAMES))

    if config.hub_model_id:
        trainer.push_to_hub()
        print(f"[silver_sft] pushed → {config.hub_model_id}")

    return config.output_dir
