"""
Honest, held-out evaluation via k-fold cross-validation.

Unlike scripts/evals/eval_all.py's validation task (which scores the Hub-hosted
*production* calibrated models — trained on the entire annotation pool, so
scoring them against that same pool is in-sample, not held-out), this script
uses the per-fold calibration checkpoints from jobs/calibrate_cv.sh. For each
condition, each of the 5 fold models predicts only on the fold it never
trained on; concatenating all 5 folds' predictions covers the entire
annotation pool with a genuinely held-out prediction for every sentence.

Requires (per condition, once): scripts/data/build_cv_folds.py, then
jobs/calibrate_cv.sh to produce data/calibrated_{model_id}-fold{0..4}/ locally
(these are never pushed to the Hub — see pipeline.py --holdout-fold).

Usage:
  python scripts/evals/eval_cv.py
  python scripts/evals/eval_cv.py --condition "Ablation: full"
"""

from __future__ import annotations
import argparse, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import pandas as pd

from src.figurative.predict import load_model, predict_sentences
from src.figurative.data import LABEL_NAMES
from scripts.evals.eval_all import metrics_for

ANNOT_FILE    = "data/figurative/bloomfield_annotated.parquet"
CV_FOLDS_FILE = "data/figurative/cv_folds.parquet"
N_FOLDS       = 5

# label -> model_id (matches the --model-id used for both the original run and
# jobs/calibrate_cv.sh). Local fold checkpoints live at
# data/calibrated_{model_id}-fold{0..N_FOLDS-1}/.
CV_CONDITIONS = [
    ("Ablation: full",            "xlm-mlm-abl-full"),
    ("Ablation: no TLM",          "xlm-mlm-abl-no-tlm"),
    ("Ablation: no CLKD",         "xlm-mlm-abl-no-clkd"),
    ("Ablation: neither",         "xlm-mlm-abl-neither"),
    ("Ablation: mono-MLM warmup", "xlm-mlm-abl-mono-mlm"),
    ("Ablation: TLM+contrastive", "xlm-mlm-abl-tlm-contrastive"),
    ("XLM-MLM (base pipeline)",   "xlm-mlm"),
    ("XLM-V (base pipeline)",     "xlm-v"),
    ("Glot500 (base pipeline)",     "glot500"),
    ("XLM-R (base pipeline)",       "xlm-r"),
    ("XLM-R-large (base pipeline)", "xlm-r-large"),
    ("mBERT (base pipeline)",       "mbert"),
    ("mDeBERTa-v3 (base pipeline)", "mdeberta"),
    ("mDistilBERT (base pipeline)", "mdistilbert"),
    # ── contrastive alpha sweep (jobs/alpha_sweep.sh) ─────────────────────────
    # alpha=0.0 is "Ablation: full" above; alpha=0.1 is "Ablation: TLM+contrastive"
    ("Contrastive α=0.05", "xlm-mlm-alpha-0p05"),
    ("Contrastive α=0.15", "xlm-mlm-alpha-0p15"),
    ("Contrastive α=0.2",  "xlm-mlm-alpha-0p2"),
    ("Contrastive α=0.3",  "xlm-mlm-alpha-0p3"),
    ("Contrastive α=0.4",  "xlm-mlm-alpha-0p4"),
    ("Contrastive α=0.5",  "xlm-mlm-alpha-0p5"),
    ("Contrastive α=0.75", "xlm-mlm-alpha-0p75"),
    ("Contrastive α=1.0",  "xlm-mlm-alpha-1p0"),
]


def fold_dir(model_id: str, fold: int) -> str:
    return f"data/calibrated_{model_id}-fold{fold}"


def predict_condition(model_id: str, folds: pd.DataFrame, batch_size: int = 32, max_length: int = 128) -> pd.DataFrame | None:
    """Return one row per sentence (text_cree, label, pred) using each fold's
    own held-out model, or None if any of the 5 fold checkpoints is missing."""
    missing = [f for f in range(N_FOLDS) if not os.path.isdir(fold_dir(model_id, f))]
    if missing:
        print(f"  SKIPPED — missing fold checkpoint(s) {missing} "
              f"(run: bash jobs/calibrate_cv.sh --model-id {model_id} ...)")
        return None

    all_preds = []
    for f in range(N_FOLDS):
        sub = folds[folds["fold"] == f]
        if sub.empty:
            continue
        ckpt = fold_dir(model_id, f)
        print(f"  fold {f}: {len(sub)} sentences  ({ckpt})")
        model, tok = load_model(ckpt)
        preds = predict_sentences(sub["text_cree"].tolist(), model, tok,
                                   batch_size=batch_size, max_length=max_length)
        del model
        torch.cuda.empty_cache()

        out = sub[["text_cree", "label", "footnote_applies"]].copy()
        out["pred"] = [p["label"] for p in preds]
        all_preds.append(out)

    return pd.concat(all_preds, ignore_index=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--condition", default=None, help="Only run this one condition (by label)")
    args = p.parse_args()

    if not os.path.exists(CV_FOLDS_FILE):
        sys.exit(f"{CV_FOLDS_FILE} not found — run scripts/data/build_cv_folds.py first")

    folds = pd.read_parquet(CV_FOLDS_FILE)
    annot = pd.read_parquet(ANNOT_FILE).dropna(subset=["text_cree", "label"])
    annot["label"] = annot["label"].str.strip().str.lower().map(
        lambda x: x if x in LABEL_NAMES else "literal"
    )
    folds = folds.merge(
        annot[["text_cree", "label", "footnote_applies"]].drop_duplicates("text_cree"),
        on="text_cree", how="left",
    )
    print(f"Annotation pool: {len(folds):,} sentences across {N_FOLDS} folds")

    conditions = CV_CONDITIONS if args.condition is None else [
        c for c in CV_CONDITIONS if c[0] == args.condition
    ]
    if not conditions:
        sys.exit(f"No condition named {args.condition!r}. Choices: {[c[0] for c in CV_CONDITIONS]}")

    rows_full, rows_gold = [], []
    for name, model_id in conditions:
        print(f"\n{'='*60}\n  {name}  ({model_id})")
        result = predict_condition(model_id, folds)
        if result is None:
            rows_full.append({"model": name, "model_id": model_id, "error": "missing fold checkpoint(s)"})
            rows_gold.append({"model": name, "model_id": model_id, "error": "missing fold checkpoint(s)"})
            continue

        m_full = metrics_for(result["label"].tolist(), result["pred"].tolist())
        print(f"  [full] macro F1={m_full['macro_f1']:.3f}  "
              + "  ".join(f"{l}={m_full[f'f1_{l}']:.2f}" for l in LABEL_NAMES))
        rows_full.append({"model": name, "model_id": model_id, "n": len(result), **m_full})

        gold_result = result[result["footnote_applies"] == True]
        m_gold = metrics_for(gold_result["label"].tolist(), gold_result["pred"].tolist())
        print(f"  [gold] macro F1={m_gold['macro_f1']:.3f}  "
              + "  ".join(f"{l}={m_gold[f'f1_{l}']:.2f}" for l in LABEL_NAMES))
        rows_gold.append({"model": name, "model_id": model_id, "n": len(gold_result), **m_gold})

    os.makedirs("data/figurative", exist_ok=True)
    pd.DataFrame(rows_full).to_parquet("data/figurative/eval_cv_full.parquet", index=False)
    pd.DataFrame(rows_gold).to_parquet("data/figurative/eval_cv_gold.parquet", index=False)
    print("\nSaved → data/figurative/eval_cv_full.parquet")
    print("Saved → data/figurative/eval_cv_gold.parquet")


if __name__ == "__main__":
    main()
