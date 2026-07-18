"""
Fine-tune a sequence classifier on the combined VUA20 + MAGPIE 3-class task.

Labels: 0=literal, 1=idiom, 2=metaphor
"""

from __future__ import annotations
import os

import pandas as pd
import torch
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

from src.figurative.config import FigurativeConfig
from src.figurative.data import build_datasets, class_weights_from, resolve_use_fast, LABEL_NAMES, NUM_LABELS
from src.figurative.evaluate import compute_metrics
from src.device import get_precision_kwargs

RESULTS_FILE = "data/figurative/english_finetune_results.parquet"


def _save_metrics(config: FigurativeConfig, metrics: dict) -> None:
    """Persist this run's final (best-checkpoint) eval metrics, keyed by
    experiment_name — otherwise these numbers only ever exist in wandb (if
    logged there) or the Trainer's in-memory state, easy to lose track of
    (see e.g. the deberta_teacher checkpoint, trained with no metrics saved
    anywhere). One row per experiment_name; reruns overwrite their own row."""
    row = {
        "experiment_name": config.experiment_name,
        "encoder":         config.encoder,
        "hub_model_id":    config.hub_model_id,
        "epochs":          config.epochs,
        "learning_rate":   config.learning_rate,
        "macro_f1":        metrics.get("eval_macro_f1"),
        **{f"{name}_f1": metrics.get(f"eval_{name}_f1") for name in LABEL_NAMES},
    }
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    if os.path.exists(RESULTS_FILE):
        df = pd.read_parquet(RESULTS_FILE)
        df = df[df["experiment_name"] != config.experiment_name]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_parquet(RESULTS_FILE, index=False)
    print(f"[train] final macro_f1={row['macro_f1']:.4f} — saved → {RESULTS_FILE}")


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits

        loss_fn = nn.CrossEntropyLoss(
            weight=self._class_weights.to(logits.device) if self._class_weights is not None else None,
        )
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


def train(config: FigurativeConfig) -> str:
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    print(f"[train] encoder  : {config.encoder}")
    print(f"[train] output   : {config.checkpoint_dir}")

    if config.wandb_project:
        os.environ["WANDB_PROJECT"] = config.wandb_project

    tokenizer = AutoTokenizer.from_pretrained(config.encoder, use_fast=resolve_use_fast(config.encoder))

    model = AutoModelForSequenceClassification.from_pretrained(
        config.encoder,
        num_labels=NUM_LABELS,
        ignore_mismatched_sizes=True,
        id2label={i: l for i, l in enumerate(LABEL_NAMES)},
        label2id={l: i for i, l in enumerate(LABEL_NAMES)},
        torch_dtype=torch.float32,
    )

    if config.freeze_encoder:
        for param in model.base_model.parameters():
            param.requires_grad = False
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[train] encoder frozen — trainable params: {n_trainable:,}")

    use_gradient_checkpointing = (
        config.gradient_checkpointing and not config.freeze_encoder and model.supports_gradient_checkpointing
    )
    if config.gradient_checkpointing and not config.freeze_encoder and not model.supports_gradient_checkpointing:
        print(f"[train] {model.__class__.__name__} does not support gradient checkpointing — disabling it "
              f"(e.g. XLMForSequenceClassification/FacebookAI/xlm-mlm-100-1280)")

    train_ds, eval_ds = build_datasets(tokenizer, config)
    weights = class_weights_from(train_ds) if config.class_weights else None

    args = TrainingArguments(
        output_dir=config.checkpoint_dir,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_accum,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=2,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=100,
        report_to="wandb" if config.wandb_project else "none",
        gradient_checkpointing=use_gradient_checkpointing,
        eval_accumulation_steps=8,
        **get_precision_kwargs(),
        **({"push_to_hub": True, "hub_model_id": config.hub_model_id} if config.hub_model_id else {}),
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        class_weights=weights,
    )

    trainer.train()
    tokenizer.save_pretrained(config.checkpoint_dir)
    trainer.save_model(config.checkpoint_dir)
    print(f"[train] saved to {config.checkpoint_dir}")

    final_metrics = trainer.evaluate()
    _save_metrics(config, final_metrics)

    if config.hub_model_id:
        trainer.push_to_hub()
        print(f"[train] pushed to hub: {config.hub_model_id}")

    return config.checkpoint_dir
