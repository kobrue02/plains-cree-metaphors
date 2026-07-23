"""
Compares DeepSeek's pool labels against a trained model's own predictions on
the same sentences and reports overall/per-label agreement plus a confusion
matrix.
"""

from __future__ import annotations
import argparse, os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
from sklearn.metrics import cohen_kappa_score

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
    # Raw agreement is misleadingly inflated by the literal-heavy class skew
    # (both sources tend to agree just by both saying "literal" often) — Cohen's
    # kappa corrects for that chance agreement, and is the number worth leading
    # with in the paper.
    kappa = cohen_kappa_score(merged["deepseek_label"], merged["model_label"], labels=LABELS)
    print(f"\n{'='*60}")
    print(f"Overall agreement: {overall:.1%}  ({merged['agree'].sum():,}/{len(merged):,})")
    print(f"Cohen's kappa    : {kappa:.3f}")
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

    # Small dedicated summary — exactly the numbers the paper's agreement
    # table needs, so filling it in later is a lookup, not a re-derivation.
    summary_rows = [{"scope": "overall", "n": len(merged),
                      "agreement": round(overall, 4), "kappa": round(kappa, 4)}]
    for label in LABELS:
        subset = merged[merged["deepseek_label"] == label]
        summary_rows.append({
            "scope": label,
            "n": len(subset),
            "agreement": round(subset["agree"].mean(), 4) if len(subset) else None,
            "kappa": None,
        })
    summary_path = args.out.replace(".parquet", "_summary.parquet")
    pd.DataFrame(summary_rows).to_parquet(summary_path, index=False)
    print(f"Saved summary → {summary_path}")

    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
