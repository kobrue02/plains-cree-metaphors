"""
Evaluate CLKD/calibrated checkpoints against the DeepSeek-annotated validation set.

Usage:
  python scripts/evals/eval_all.py
  python scripts/evals/eval_all.py --model "TLM+CLKD"

Output files
------------
  data/figurative/eval_validation_gold.parquet — the only genuinely held-out
  evaluation set (footnote-verified gold). There is no "full set" task —
  every non-gold sentence is now part of the silver training pool, so
  scoring a silver-trained model against it would be leakage, not a
  broader validation view.
"""

from __future__ import annotations
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import pandas as pd

from src.figurative.predict import load_model, predict_sentences
from src.figurative.data import LABEL_NAMES


def metrics_for(y_true: list[str], y_pred: list[str]) -> dict:
    """Per-class and macro P/R/F1 as a flat dict. Shared with eval_cv.py."""
    from sklearn.metrics import classification_report
    report = classification_report(
        y_true, y_pred, labels=LABEL_NAMES, output_dict=True, zero_division=0
    )
    row = {}
    for label in LABEL_NAMES:
        r = report.get(label, {})
        row[f"p_{label}"]  = round(r.get("precision", 0.0), 4)
        row[f"r_{label}"]  = round(r.get("recall",    0.0), 4)
        row[f"f1_{label}"] = round(r.get("f1-score",  0.0), 4)
    macro = report.get("macro avg", {})
    row["macro_p"]  = round(macro.get("precision", 0.0), 4)
    row["macro_r"]  = round(macro.get("recall",    0.0), 4)
    row["macro_f1"] = round(macro.get("f1-score",  0.0), 4)
    row["accuracy"] = round(report.get("accuracy", 0.0), 4)
    return row


def _save_or_merge(path: str, new_rows: list[dict], key_col: str = "model") -> None:
    """Upsert new_rows into path by key_col, keeping every other existing row —
    so `--model X` only touches X's row instead of wiping out every other
    model's already-computed result (a plain overwrite did that before)."""
    new_df = pd.DataFrame(new_rows)
    if os.path.exists(path):
        old_df = pd.read_parquet(path)
        keys = set(new_df[key_col])
        old_df = old_df[~old_df[key_col].isin(keys)]
        merged = pd.concat([old_df, new_df], ignore_index=True)
    else:
        merged = new_df
    merged.to_parquet(path, index=False)


def bootstrap_ci(y_true: list[str], y_pred: list[str],
                  metric_keys: tuple[str, ...] = ("macro_f1", "f1_idiom", "f1_metaphor", "f1_simile"),
                  n_boot: int = 2000, seed: int = 42, ci: float = 0.95) -> dict:
    """Percentile bootstrap CI via sentence-level resampling (with replacement).

    Quantifies how much each metric could shift under a different draw of this
    small evaluation set — not predictor stochasticity — which is the concern
    for a 228-sentence gold set with only a handful of idiom/metaphor/simile
    instances. Shared across every script that scores predictions against the
    gold set, so every reported metric gets a CI alongside the point estimate.
    """
    import numpy as np
    y_true, y_pred = list(y_true), list(y_pred)
    n = len(y_true)
    rng = np.random.RandomState(seed)
    samples = {k: [] for k in metric_keys}
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        m = metrics_for([y_true[i] for i in idx], [y_pred[i] for i in idx])
        for k in metric_keys:
            samples[k].append(m[k])
    lo_pct, hi_pct = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    out = {}
    for k in metric_keys:
        out[f"{k}_ci_lo"] = round(float(np.percentile(samples[k], lo_pct)), 4)
        out[f"{k}_ci_hi"] = round(float(np.percentile(samples[k], hi_pct)), 4)
    return out


_VALIDATION_MODELS = [
    # ceiling baseline: the CLKD teacher scored on the English glosses rather
    # than the Cree text — bounds how much of the students' error is inherited
    # teacher weakness (idiom/metaphor/simile) vs. added cross-lingual-transfer loss
    ("Teacher-on-glosses (English ceiling)", "KonradBRG/deberta-v3-base-figurative", "text_en"),
    # TLM+CLKD vs. TLM+Silver-SFT comparison (Sections 3/4 of the paper)
    ("TLM+CLKD",                       "KonradBRG/xlm-mlm-plains-cree-en-clkd"),
    ("TLM+Silver-SFT (hierarchical)",  "KonradBRG/xlm-mlm-plains-cree-en-silver-sft-hierarchical"),
    # raw encoder ablation: silver-SFT directly from the base (non-TLM-adapted)
    # encoder — isolates whether TLM adaptation itself matters, or silver
    # labels alone carry the classifier (Section 6.2's evaluation paragraph)
    ("Silver-SFT, no TLM adaptation",  "KonradBRG/xlm-mlm-plains-cree-en-silver-sft-no-tlm"),
]
# NOTE: every other entry that used to live here (XLM-R/Glot500/XLM-V variants,
# the calibrated/ablation/contrastive-alpha checkpoints) pointed at Hub
# repos deleted during the project cleanup once the CV-on-gold/calibration
# pipeline and multi-encoder comparison were dropped from the paper's scope —
# removed rather than left to fail with 401/404s on every run.

# alpha -> (label in _VALIDATION_MODELS, calibrated repo_id) — used by
# scripts/viz/generate_figures.py to build the contrastive-alpha line plot.
ALPHA_SWEEP = [
    (0.0,  "Ablation: full",            "KonradBRG/xlm-mlm-abl-full-plains-cree-en-calibrated"),
    (0.05, "Contrastive α=0.05",        "KonradBRG/xlm-mlm-alpha-0p05-plains-cree-en-calibrated"),
    (0.1,  "Ablation: TLM+contrastive", "KonradBRG/xlm-mlm-abl-tlm-contrastive-plains-cree-en-calibrated"),
    (0.15, "Contrastive α=0.15",        "KonradBRG/xlm-mlm-alpha-0p15-plains-cree-en-calibrated"),
    (0.2,  "Contrastive α=0.2",         "KonradBRG/xlm-mlm-alpha-0p2-plains-cree-en-calibrated"),
    (0.3,  "Contrastive α=0.3",         "KonradBRG/xlm-mlm-alpha-0p3-plains-cree-en-calibrated"),
    (0.4,  "Contrastive α=0.4",         "KonradBRG/xlm-mlm-alpha-0p4-plains-cree-en-calibrated"),
    (0.5,  "Contrastive α=0.5",         "KonradBRG/xlm-mlm-alpha-0p5-plains-cree-en-calibrated"),
    (0.75, "Contrastive α=0.75",        "KonradBRG/xlm-mlm-alpha-0p75-plains-cree-en-calibrated"),
    (1.0,  "Contrastive α=1.0",         "KonradBRG/xlm-mlm-alpha-1p0-plains-cree-en-calibrated"),
]

ANNOT_FILE = "data/figurative/bloomfield_annotated.parquet"


def task_validation(model: str | None = None) -> None:
    """Evaluate models against the gold (footnote-verified) set — the only subset
    that's genuinely held out. There used to also be a "full validation set"
    (gold + the non-footnote-verified rows of bloomfield_annotated.parquet),
    but those non-gold rows are now part of the silver training pool itself
    (see silver_sft.py) — scoring Silver-SFT against them would be leakage,
    not a weaker validation view, so that task was removed rather than kept
    as a second, contaminated number alongside the honest one."""
    output_gold = "data/figurative/eval_validation_gold.parquet"

    models = _VALIDATION_MODELS if model is None else [
        c for c in _VALIDATION_MODELS if c[0] == model
    ]
    if not models:
        sys.exit(f"No model named {model!r}. Choices: {[c[0] for c in _VALIDATION_MODELS]}")

    def evaluate(df: pd.DataFrame, subset_name: str) -> list[dict]:
        y_true = df["label"].tolist()

        rows = []
        for entry in models:
            name, ckpt = entry[0], entry[1]
            text_col = entry[2] if len(entry) > 2 else "text_cree"
            texts = df[text_col].tolist()
            print(f"\n{'='*60}\n  {name}  [{subset_name}]")
            try:
                model, tok = load_model(ckpt)
                preds  = predict_sentences(texts, model, tok)
                y_pred = [p["label"] for p in preds]
                del model
                torch.cuda.empty_cache()

                m = metrics_for(y_true, y_pred)
                ci = bootstrap_ci(y_true, y_pred)
                print(f"  macro F1={m['macro_f1']:.3f} "
                      f"[{ci['macro_f1_ci_lo']:.3f}, {ci['macro_f1_ci_hi']:.3f}]  "
                      + "  ".join(f"{l}={m[f'f1_{l}']:.2f}" for l in LABEL_NAMES))
                rows.append({"model": name, "checkpoint": ckpt, **m, **ci})
            except Exception as exc:
                print(f"  SKIPPED — {exc}")
                rows.append({"model": name, "checkpoint": ckpt, "error": str(exc)})

        return rows

    annot = pd.read_parquet(ANNOT_FILE)
    annot = annot.dropna(subset=["text_cree", "label"])
    # normalise label column (in case of stray whitespace)
    annot["label"] = annot["label"].str.strip().str.lower().map(
        lambda x: x if x in LABEL_NAMES else "literal"
    )

    gold = annot[annot["footnote_applies"] == True]

    print(f"Gold subset (footnote_applies=True): {len(gold)} sentences")
    print(f"  label dist: {gold['label'].value_counts().to_dict()}")

    os.makedirs("data/figurative", exist_ok=True)

    print("\n\n── Gold subset (footnote_applies=True) ──────────────────────")
    rows_gold = evaluate(gold, "gold")
    _save_or_merge(output_gold, rows_gold)
    print(f"\nSaved → {output_gold}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=None, help="Only run this one model (by label in _VALIDATION_MODELS)")
    args = p.parse_args()
    task_validation(model=args.model)
