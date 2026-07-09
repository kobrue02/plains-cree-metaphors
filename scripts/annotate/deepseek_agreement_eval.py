"""
Compare DeepSeek labels against a model's own predictions.

Reads the outputs of the two independent steps:
  1. scripts/annotate/deepseek_label_pool.py  → data/figurative/deepseek_labels.parquet
  2. scripts/annotate/predict_pool.py         → data/figurative/model_predictions.parquet
joins them on text_cree, and reports the agreement rate (overall, per
DeepSeek-label, and a full confusion matrix). Run both of those first.

Usage:
  python scripts/annotate/deepseek_agreement_eval.py
  python scripts/annotate/deepseek_agreement_eval.py --deepseek-file <path> --model-file <path>
"""

from __future__ import annotations
import argparse, os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd

DEEPSEEK_FILE = "data/figurative/deepseek_labels.parquet"
MODEL_FILE    = "data/figurative/model_predictions.parquet"
OUTPUT_FILE   = "data/figurative/deepseek_agreement_eval.parquet"

LABELS = ["literal", "idiom", "metaphor", "simile"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--deepseek-file", default=DEEPSEEK_FILE)
    p.add_argument("--model-file",    default=MODEL_FILE)
    p.add_argument("--out",           default=OUTPUT_FILE)
    args = p.parse_args()

    for path, label in [(args.deepseek_file, "DeepSeek labels"), (args.model_file, "model predictions")]:
        if not os.path.exists(path):
            sys.exit(f"{label} file not found: {path}\n"
                     f"Run deepseek_label_pool.py / predict_pool.py first.")

    deepseek_df = pd.read_parquet(args.deepseek_file)[["text_cree", "text_en", "deepseek_label"]]
    model_df    = pd.read_parquet(args.model_file)[["text_cree", "model_label"]]

    merged = deepseek_df.merge(model_df, on="text_cree", how="inner")
    dropped_ds  = len(deepseek_df) - len(merged)
    dropped_mdl = len(model_df) - len(merged)
    if dropped_ds or dropped_mdl:
        print(f"[warn] {dropped_ds:,} DeepSeek-only and {dropped_mdl:,} model-only "
              f"sentences dropped (pool filters didn't match — check --exclude-source "
              f"was consistent between the two runs)")

    merged["agree"] = merged["deepseek_label"] == merged["model_label"]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    merged.to_parquet(args.out, index=False)

    overall = merged["agree"].mean()
    print(f"\n{'='*60}")
    print(f"Overall agreement: {overall:.1%}  ({merged['agree'].sum():,}/{len(merged):,})")
    print(f"{'='*60}")

    print("\nDeepSeek label distribution:")
    print(merged["deepseek_label"].value_counts().to_string())
    print("\nModel label distribution:")
    print(merged["model_label"].value_counts().to_string())

    print("\nPer-DeepSeek-label agreement (model matches DeepSeek | DeepSeek said X):")
    for label in LABELS:
        subset = merged[merged["deepseek_label"] == label]
        if len(subset):
            print(f"  {label:10s}: {subset['agree'].mean():.1%}  (n={len(subset):,})")

    print("\nConfusion (rows=DeepSeek, cols=model):")
    print(pd.crosstab(merged["deepseek_label"], merged["model_label"]).to_string())

    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
