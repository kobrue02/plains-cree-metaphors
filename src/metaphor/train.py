"""
Fine-tune any HuggingFace token classifier on VUA20.

The Trainer subclass adds optional class-weighted cross-entropy so the
minority metaphor class is not swamped by the majority literal class.
"""

from __future__ import annotations
import os

import torch
import torch.nn as nn
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

from src.metaphor.model import XLMRobertaLayerSelectForTokenClassification

from src.metaphor.config import ExperimentConfig
from src.metaphor.data import build_datasets, class_weights_from
from src.metaphor.evaluate import compute_metrics
from src.device import get_precision_kwargs


class WeightedTrainer(Trainer):
    """Trainer that applies per-class loss weights."""

    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits                           # (B, L, num_labels)

        loss_fn = nn.CrossEntropyLoss(
            weight=self._class_weights.to(logits.device) if self._class_weights is not None else None,
            ignore_index=-100,
        )
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def train(config: ExperimentConfig) -> str:
    """Fine-tune the encoder specified in config on VUA20.

    Returns the path to the saved checkpoint directory.
    """
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    print(f"[train] encoder  : {config.encoder}")
    print(f"[train] output   : {config.checkpoint_dir}")

    if config.wandb_project:
        os.environ["WANDB_PROJECT"] = config.wandb_project

    tokenizer = AutoTokenizer.from_pretrained(config.encoder)

    if config.hidden_layer is not None:
        model = XLMRobertaLayerSelectForTokenClassification.from_pretrained(
            config.encoder,
            num_labels=2,
            ignore_mismatched_sizes=True,
        )
        model.config.hidden_layer = config.hidden_layer
        print(f"[train] hidden layer : {config.hidden_layer}")
    else:
        model = AutoModelForTokenClassification.from_pretrained(
            config.encoder,
            num_labels=2,
            ignore_mismatched_sizes=True,   # classification head is always reinitialised
        )

    train_ds, eval_ds = build_datasets(tokenizer, config)
    weights = class_weights_from(train_ds) if config.class_weights else None
    if weights is not None:
        print(f"[train] class weights: literal={weights[0]:.3f}, metaphor={weights[1]:.3f}")

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
        metric_for_best_model="metaphor_f1",
        greater_is_better=True,
        logging_steps=100,
        report_to="wandb" if config.wandb_project else "none",
        gradient_checkpointing=True,
        eval_accumulation_steps=8,   # flush eval logits to CPU every 8 steps — prevents OOM on large models
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
    trainer.save_model(config.checkpoint_dir)
    tokenizer.save_pretrained(config.checkpoint_dir)
    print(f"[train] saved to {config.checkpoint_dir}")

    if config.hub_model_id:
        trainer.push_to_hub()
        print(f"[train] pushed to hub: {config.hub_model_id}")

    return config.checkpoint_dir
