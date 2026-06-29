"""
Calibrate a CLKD model on a small set of DeepSeek-labelled Cree sentences.

Low-LR pass that adjusts the classifier toward the true Plains Cree figurative
distribution without forgetting the cross-lingual representations built during
TLM and CLKD.
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

from src.figurative.data import FigurativeDataset, class_weights_from, LABEL_NAMES, NUM_LABELS
from src.figurative.evaluate import compute_metrics
from src.figurative.config import FigurativeConfig

LABEL2ID = {l: i for i, l in enumerate(LABEL_NAMES)}

LABEL_MAP = {
    "literal": "literal", "none": "literal",
    "idiom": "idiom", "proverb": "idiom",
    "metaphor": "metaphor",
    "simile": "simile",
}


@dataclass
class CalibrateConfig:
    checkpoint:   str            # CLKD model to start from
    annot_file:   str  = "data/figurative/bloomfield_annotated.parquet"
    output_dir:   str  = "data/calibrated"
    hub_model_id: str | None = None
    wandb_project: str | None = None
    epochs:        int   = 10
    batch_size:    int   = 8
    learning_rate: float = 5e-6
    max_length:    int   = 128
    literal_ratio: int   = 3     # literals per figurative sentence
    gold_only:     bool  = False  # restrict to footnote_applies=True rows


class _WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        loss    = nn.CrossEntropyLoss(
            weight=self._class_weights.to(outputs.logits.device)
            if self._class_weights is not None else None
        )(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


def _load_records(config: CalibrateConfig) -> list[dict]:
    df = pd.read_parquet(config.annot_file)
    df["label"] = (df["label"].str.strip().str.lower()
                   .map(lambda x: LABEL_MAP.get(x, "literal")))
    df = df.dropna(subset=["text_cree", "label"])

    if config.gold_only:
        df = df[df["footnote_applies"] == True]

    figurative = df[df["label"] != "literal"]
    literals   = df[df["label"] == "literal"]
    n_lit      = min(len(literals), len(figurative) * config.literal_ratio)
    balanced   = pd.concat([figurative,
                            literals.sample(n=n_lit, random_state=42)])

    records = [
        {"text": row["text_cree"], "label": LABEL2ID.get(row["label"], 0)}
        for _, row in balanced.iterrows()
    ]
    counts = balanced["label"].value_counts().to_dict()
    print(f"[calibrate] {len(records)} sentences — {counts}")
    return records


def calibrate(config: CalibrateConfig) -> str:
    os.makedirs(config.output_dir, exist_ok=True)

    if config.wandb_project:
        os.environ["WANDB_PROJECT"] = config.wandb_project

    records = _load_records(config)
    train_recs, eval_recs = train_test_split(
        records,
        test_size=0.2,
        random_state=42,
        stratify=[r["label"] for r in records],
    )
    print(f"[calibrate] train={len(train_recs)}  eval={len(eval_recs)}")

    use_fast  = "deberta-v3" not in config.checkpoint.lower()
    tokenizer = AutoTokenizer.from_pretrained(config.checkpoint, use_fast=use_fast)

    ds_config = FigurativeConfig(encoder=config.checkpoint,
                                 max_length=config.max_length,
                                 batch_size=config.batch_size)
    train_ds = FigurativeDataset(train_recs, tokenizer, ds_config)
    eval_ds  = FigurativeDataset(eval_recs,  tokenizer, ds_config)
    weights  = class_weights_from(train_ds)

    model = AutoModelForSequenceClassification.from_pretrained(
        config.checkpoint,
        num_labels=NUM_LABELS,
        ignore_mismatched_sizes=False,
        id2label={i: l for i, l in enumerate(LABEL_NAMES)},
        label2id=LABEL2ID,
        torch_dtype=torch.float32,
    )

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
        **hub_kwargs,
    )

    trainer = _WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        class_weights=weights,
    )

    print(f"[calibrate] starting from {config.checkpoint}")
    trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print(f"[calibrate] saved → {config.output_dir}")

    if config.hub_model_id:
        trainer.push_to_hub()
        print(f"[calibrate] pushed → {config.hub_model_id}")

    return config.output_dir
