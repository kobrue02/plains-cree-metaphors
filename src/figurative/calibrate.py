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

from src.device import get_trainer_device_kwargs
from src.figurative.data import FigurativeDataset, class_weights_from, resolve_use_fast, LABEL_NAMES, NUM_LABELS
from src.figurative.evaluate import compute_metrics
from src.figurative.config import FigurativeConfig

LABEL2ID = {l: i for i, l in enumerate(LABEL_NAMES)}

CV_FOLDS_FILE = "data/figurative/cv_folds.parquet"
GOLD_FILE     = "data/figurative/bloomfield_annotated.parquet"
RESULTS_FILE  = "data/figurative/calibrate_results.parquet"

LABEL_MAP = {
    "literal": "literal", "none": "literal",
    "idiom": "idiom", "proverb": "idiom",
    "metaphor": "metaphor",
    "simile": "simile",
}


@dataclass
class CalibrateConfig:
    """
    Held-out policy: outside CV mode (holdout_fold=None), the 219
    footnote-verified sentences in GOLD_FILE are the fixed evaluation set for
    every run, regardless of what annot_file trains on (gold, silver, or
    anything else) — they are always excluded from training and are the only
    thing calibrate() reports eval metrics against. eval_file lets you point
    at a different file for eval instead (rare — e.g. a smoke test); it does
    NOT change what gets excluded from training, which is always GOLD_FILE's
    footnoted sentences.

    In CV mode (holdout_fold set) this doesn't apply: that's a separate,
    already-honest k-fold rotation (see scripts/data/build_cv_folds.py /
    scripts/evals/eval_cv.py) where every sentence is trained on in 4/5 folds
    and scored only in the fold where it's held out.
    """
    checkpoint:   str            # CLKD model to start from
    annot_file:   str  = "data/figurative/bloomfield_annotated.parquet"
    label_col:    str  = "label"  # column in annot_file holding the label string
    eval_file:    str | None = None  # override the fixed gold test set (rare)
    output_dir:   str  = "data/calibrated"
    hub_model_id: str | None = None
    wandb_project: str | None = None
    epochs:        int   = 10
    batch_size:    int   = 8
    learning_rate: float = 5e-6
    max_length:    int   = 128
    literal_ratio: int   = 3     # literals per figurative sentence
    holdout_fold:  int | None = None  # exclude this cv_folds.parquet fold from training


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


def _read_labeled(path: str, label_col: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["label"] = (df[label_col].str.strip().str.lower()
                   .map(lambda x: LABEL_MAP.get(x, "literal")))
    return df.dropna(subset=["text_cree", "label"])


def _to_records(df: pd.DataFrame) -> list[dict]:
    return [
        {"text": row["text_cree"], "label": LABEL2ID.get(row["label"], 0)}
        for _, row in df.iterrows()
    ]


def _load_gold_test_set() -> pd.DataFrame:
    """The fixed, footnote-verified held-out test set (n=219). Never used for
    training outside CV mode — this is what every non-CV run is scored against."""
    df = _read_labeled(GOLD_FILE, "label")
    return df[df["footnote_applies"] == True]


def _load_records(config: CalibrateConfig) -> list[dict]:
    df = _read_labeled(config.annot_file, config.label_col)

    if config.holdout_fold is not None:
        if not os.path.exists(CV_FOLDS_FILE):
            raise FileNotFoundError(
                f"{CV_FOLDS_FILE} not found — run scripts/data/build_cv_folds.py first"
            )
        folds = pd.read_parquet(CV_FOLDS_FILE).set_index("text_cree")["fold"]
        held_out = set(folds[folds == config.holdout_fold].index)
        n_before = len(df)
        df = df[~df["text_cree"].isin(held_out)]
        print(f"[calibrate] holdout_fold={config.holdout_fold} — "
              f"excluded {n_before - len(df)} sentences")
    else:
        test_texts = set(_load_gold_test_set()["text_cree"])
        n_before   = len(df)
        df = df[~df["text_cree"].isin(test_texts)]
        excluded = n_before - len(df)
        if excluded:
            print(f"[calibrate] excluded {excluded} sentences that are in the "
                  f"fixed gold test set (footnote_applies=True) from training")

    figurative = df[df["label"] != "literal"]
    literals   = df[df["label"] == "literal"]
    n_lit      = min(len(literals), len(figurative) * config.literal_ratio)
    balanced   = pd.concat([figurative,
                            literals.sample(n=n_lit, random_state=42)])

    records = _to_records(balanced)
    counts = balanced["label"].value_counts().to_dict()
    print(f"[calibrate] {len(records)} sentences — {counts}")
    return records


def _load_eval_records(config: CalibrateConfig) -> list[dict]:
    """Eval set: the fixed gold test set by default, or an explicit override."""
    if config.eval_file is None:
        df = _load_gold_test_set()
    else:
        df = _read_labeled(config.eval_file, "label")
    records = _to_records(df)
    counts = df["label"].value_counts().to_dict()
    print(f"[calibrate] eval — {len(records)} sentences — {counts}"
          + ("" if config.eval_file is None else f"  (eval_file={config.eval_file})"))
    return records


def _save_metrics(config: CalibrateConfig, metrics: dict) -> None:
    """Persist the final (best-checkpoint) eval against the fixed gold test
    set — calibrate() has always computed this every epoch (it's what
    early stopping/best-checkpoint selection uses), but never saved the final
    number anywhere durable, so it only ever existed in wandb (if logged) or
    was lost once the job finished. One row per hub_model_id (falling back to
    checkpoint+output_dir if no hub id was given); reruns overwrite their own
    row. Skipped for CV-mode runs (config.holdout_fold is not None) — that
    in-training eval is only a proxy split for early stopping, not the honest
    number (see scripts/evals/eval_cv.py for that)."""
    key = config.hub_model_id or f"{config.checkpoint}->{config.output_dir}"
    row = {
        "key":           key,
        "checkpoint":    config.checkpoint,
        "hub_model_id":  config.hub_model_id,
        "annot_file":    config.annot_file,
        "epochs":        config.epochs,
        "learning_rate": config.learning_rate,
        "macro_f1":      metrics.get("eval_macro_f1"),
        **{f"{name}_f1": metrics.get(f"eval_{name}_f1") for name in LABEL_NAMES},
    }
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    if os.path.exists(RESULTS_FILE):
        df = pd.read_parquet(RESULTS_FILE)
        df = df[df["key"] != key]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_parquet(RESULTS_FILE, index=False)
    print(f"[calibrate] final macro_f1={row['macro_f1']:.4f} (vs. fixed gold test set) — saved → {RESULTS_FILE}")


def calibrate(config: CalibrateConfig) -> str:
    os.makedirs(config.output_dir, exist_ok=True)

    if config.wandb_project:
        os.environ["WANDB_PROJECT"] = config.wandb_project

    train_recs = _load_records(config)
    if config.holdout_fold is not None:
        # CV mode: this in-training eval is only a proxy for early stopping —
        # the honest, held-out score comes later from scripts/evals/eval_cv.py
        # predicting on the fold this run actually excluded from training.
        train_recs, eval_recs = train_test_split(
            train_recs,
            test_size=0.2,
            random_state=42,
            stratify=[r["label"] for r in train_recs],
        )
    else:
        eval_recs = _load_eval_records(config)
    print(f"[calibrate] train={len(train_recs)}  eval={len(eval_recs)}")

    tokenizer = AutoTokenizer.from_pretrained(config.checkpoint, use_fast=resolve_use_fast(config.checkpoint))

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
        **get_trainer_device_kwargs(),
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
    tokenizer.save_pretrained(config.output_dir)
    trainer.save_model(config.output_dir)
    print(f"[calibrate] saved → {config.output_dir}")

    if config.holdout_fold is None:
        final_metrics = trainer.evaluate()
        _save_metrics(config, final_metrics)

    if config.hub_model_id:
        trainer.push_to_hub()
        print(f"[calibrate] pushed → {config.hub_model_id}")

    return config.output_dir
