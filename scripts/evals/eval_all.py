"""
Evaluate CLKD/calibrated checkpoints against the DeepSeek-annotated validation set.

Usage:
  python scripts/evals/eval_all.py
  python scripts/evals/eval_all.py --model "XLM-MLM CLKD (pre-calibration)"

Output files
------------
  data/figurative/eval_validation_full.parquet — validation task (full set)
  data/figurative/eval_validation_gold.parquet — validation task (gold subset)
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


# ── Shared model list ────────────────────────────────────────────────────────

_VALIDATION_MODELS = [
    ("XLM-R base",            "KonradBRG/xlm-r-plains-cree-en-tlm-figurative"),
    ("XLM-MLM CLKD f12",      "KonradBRG/xlm-mlm-100-1280-plains-cree-en-clkd-frozen12"),
    ("XLM-MLM CLKD full",     "KonradBRG/xlm-mlm-100-1280-plains-cree-en-clkd-full"),
    ("Glot500 CLKD direct",   "KonradBRG/glot500-base-plains-cree-en-clkd-direct"),
    ("Glot500 CLKD + TLM",    "KonradBRG/glot500-base-plains-cree-en-clkd-tlm"),
    ("XLM-V CLKD direct",     "KonradBRG/xlm-v-base-plains-cree-en-clkd-direct"),
    # ── CLKD, pre-calibration — base pipeline (pipeline.py), matched lineage with
    # the calibrated entries right below, for a clean does-calibration-help check ──
    ("XLM-MLM CLKD (pre-calibration)", "KonradBRG/xlm-mlm-plains-cree-en-clkd"),
    ("XLM-V CLKD (pre-calibration)",   "KonradBRG/xlm-v-plains-cree-en-clkd"),
    # ── calibrated — base pipeline (pipeline.py) ──────────────────────────────
    ("XLM-MLM calibrated",    "KonradBRG/xlm-mlm-plains-cree-en-calibrated"),
    ("Glot500 calibrated",    "KonradBRG/glot500-plains-cree-en-calibrated"),
    ("XLM-V calibrated",      "KonradBRG/xlm-v-plains-cree-en-calibrated"),
    # ── calibrated — ablation study (jobs/ablation.sh, xlm-mlm base) ──────────
    ("Ablation: full",            "KonradBRG/xlm-mlm-abl-full-plains-cree-en-calibrated"),
    ("Ablation: no TLM",          "KonradBRG/xlm-mlm-abl-no-tlm-plains-cree-en-calibrated"),
    ("Ablation: no CLKD",         "KonradBRG/xlm-mlm-abl-no-clkd-plains-cree-en-calibrated"),
    ("Ablation: neither",         "KonradBRG/xlm-mlm-abl-neither-plains-cree-en-calibrated"),
    ("Ablation: mono-MLM warmup", "KonradBRG/xlm-mlm-abl-mono-mlm-plains-cree-en-calibrated"),
    ("Ablation: TLM+contrastive", "KonradBRG/xlm-mlm-abl-tlm-contrastive-plains-cree-en-calibrated"),
    # ── contrastive alpha sweep (jobs/alpha_sweep.sh) ─────────────────────────
    # alpha=0.0 is "Ablation: full" above; alpha=0.1 is "Ablation: TLM+contrastive"
    ("Contrastive α=0.05", "KonradBRG/xlm-mlm-alpha-0p05-plains-cree-en-calibrated"),
    ("Contrastive α=0.15", "KonradBRG/xlm-mlm-alpha-0p15-plains-cree-en-calibrated"),
    ("Contrastive α=0.2",  "KonradBRG/xlm-mlm-alpha-0p2-plains-cree-en-calibrated"),
    ("Contrastive α=0.3",  "KonradBRG/xlm-mlm-alpha-0p3-plains-cree-en-calibrated"),
    ("Contrastive α=0.4",  "KonradBRG/xlm-mlm-alpha-0p4-plains-cree-en-calibrated"),
    ("Contrastive α=0.5",  "KonradBRG/xlm-mlm-alpha-0p5-plains-cree-en-calibrated"),
    ("Contrastive α=0.75", "KonradBRG/xlm-mlm-alpha-0p75-plains-cree-en-calibrated"),
    ("Contrastive α=1.0",  "KonradBRG/xlm-mlm-alpha-1p0-plains-cree-en-calibrated"),
    # ── ceiling baseline: the CLKD teacher scored on the English glosses rather
    # than the Cree text — bounds how much of the students' error is inherited
    # teacher weakness (idiom/metaphor/simile) vs. added cross-lingual-transfer loss ──
    ("Teacher-on-glosses (English ceiling)", "KonradBRG/deberta-v3-base-figurative", "text_en"),
    # ── TLM+CLKD vs. TLM+Silver-SFT comparison (Sections 3/4 of the paper) ──
    ("TLM+CLKD",                 "KonradBRG/xlm-mlm-plains-cree-en-clkd"),
    ("TLM+Silver-SFT (hierarchical)", "KonradBRG/xlm-mlm-plains-cree-en-silver-sft-hierarchical"),
]

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


# ── Task: validation ──────────────────────────────────────────────────────────

def task_validation(model: str | None = None) -> None:
    """Evaluate CLKD/calibrated models against the DeepSeek-annotated validation set."""
    output_full = "data/figurative/eval_validation_full.parquet"
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

    # NOTE: these models were calibrated on (subsets of) this same annotation pool,
    # so this is an in-sample sanity check, not a held-out evaluation — see
    # scripts/evals/eval_cv.py for the honest cross-validated numbers.
    annot = pd.read_parquet(ANNOT_FILE)
    annot = annot.dropna(subset=["text_cree", "label"])
    # normalise label column (in case of stray whitespace)
    annot["label"] = annot["label"].str.strip().str.lower().map(
        lambda x: x if x in LABEL_NAMES else "literal"
    )

    gold = annot[annot["footnote_applies"] == True]

    print("NOTE: in-sample sanity check, not held-out — see eval_cv.py for CV numbers")
    print(f"Full validation set : {len(annot)} sentences")
    print(f"  label dist: {annot['label'].value_counts().to_dict()}")
    print(f"Gold subset (footnote_applies=True): {len(gold)} sentences")
    print(f"  label dist: {gold['label'].value_counts().to_dict()}")

    os.makedirs("data/figurative", exist_ok=True)

    print("\n\n── Full validation set ──────────────────────────────────────")
    rows_full = evaluate(annot, "full")
    pd.DataFrame(rows_full).to_parquet(output_full, index=False)
    print(f"\nSaved → {output_full}")

    print("\n\n── Gold subset (footnote_applies=True) ──────────────────────")
    rows_gold = evaluate(gold, "gold")
    pd.DataFrame(rows_gold).to_parquet(output_gold, index=False)
    print(f"\nSaved → {output_gold}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=None, help="Only run this one model (by label in _VALIDATION_MODELS)")
    args = p.parse_args()
    task_validation(model=args.model)
