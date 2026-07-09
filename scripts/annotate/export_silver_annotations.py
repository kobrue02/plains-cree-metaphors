"""
Export silver annotations into the same column layout as the gold
annotations file (data/figurative/bloomfield_annotated.parquet), as a separate
parquet — NOT merged with gold, just structurally consistent with it so the
two are easy to compare or load side by side later.

Usage:
  python scripts/annotate/export_silver_annotations.py
"""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd

SILVER_FILE = "data/figurative/deepseek_labels.parquet"
POOL_FILE   = "data/bloomfield_texts_sentences.parquet"
OUTPUT_FILE = "data/figurative/silver_annotated.parquet"

# Gold's columns (data/figurative/bloomfield_annotated.parquet), for reference:
#   paragraph_id, sentence_id, source_file, text_cree, text_en, label,
#   label_raw, footnote_applies, rationale, footnote_en


def main() -> None:
    silver = pd.read_parquet(SILVER_FILE)
    silver["text_cree"] = silver["text_cree"].astype(str).str.strip()

    pool = pd.read_parquet(POOL_FILE)[["text_cree", "source_file"]].copy()
    pool["text_cree"] = pool["text_cree"].astype(str).str.strip()
    pool = pool.drop_duplicates("text_cree")

    df = silver.merge(pool, on="text_cree", how="left")

    out = pd.DataFrame({
        "paragraph_id":     df["paragraph_id"],
        "sentence_id":      df["sentence_id"],
        "source_file":      df["source_file"],
        "text_cree":        df["text_cree"],
        "text_en":          df["text_en"],
        "label":            df["deepseek_label"],
        # no separate raw/normalised distinction for silver — deepseek_label_pool.py
        # already restricts responses to the 4 canonical labels
        "label_raw":        df["deepseek_label"],
        "footnote_applies": False,
        "rationale":        df["reasoning"],
        "footnote_en":      "",
    })

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    out.to_parquet(OUTPUT_FILE, index=False)

    print(f"Saved {len(out):,} silver annotations (gold-matching layout) → {OUTPUT_FILE}")
    print(out["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
