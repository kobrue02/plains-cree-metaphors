"""
Carve out a fixed, held-out test split from the annotation pool, once.

calibrate.py excludes these sentences from every calibration run's training
pool (regardless of which checkpoint/architecture/ablation condition), and
eval_all.py's validation task evaluates exclusively against them. Without
this, calibration trains on effectively all annotated figurative examples
(there are so few that the class-balancing step already includes ~all of
them), and the "validation" task re-scores the same examples — the reported
macro F1 is train-set performance, not held-out performance.

Run once, before (re-)calibrating anything:
  python scripts/data/build_test_split.py

Refuses to overwrite an existing split — delete data/figurative/test_split.parquet
manually first if you actually intend to redraw it (doing so silently would
change what "held-out" means for any model already calibrated against it).
"""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd
from sklearn.model_selection import train_test_split

ANNOT_FILE      = "data/figurative/annotations.parquet"
TEST_SPLIT_FILE = "data/figurative/test_split.parquet"
LABEL_NAMES     = ["literal", "idiom", "metaphor", "simile"]
TEST_SIZE       = 0.2
SEED            = 42


def main() -> None:
    if os.path.exists(TEST_SPLIT_FILE):
        sys.exit(f"{TEST_SPLIT_FILE} already exists — refusing to overwrite. "
                 "Delete it manually first if you really mean to redraw the split.")

    df = pd.read_parquet(ANNOT_FILE)
    df = df.dropna(subset=["text_cree", "label"])
    df = df.drop_duplicates(subset=["text_cree"], keep="first")
    df["label"] = df["label"].str.strip().str.lower().map(
        lambda x: x if x in LABEL_NAMES else "literal"
    )

    _, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=SEED, stratify=df["label"],
    )

    os.makedirs(os.path.dirname(TEST_SPLIT_FILE), exist_ok=True)
    test_df[["text_cree"]].to_parquet(TEST_SPLIT_FILE, index=False)

    print(f"Held out {len(test_df):,} / {len(df):,} sentences as the fixed test split.")
    print(f"Label distribution (held-out): {test_df['label'].value_counts().to_dict()}")
    print(f"Gold (footnote_applies=True) in held-out set: "
          f"{int((test_df['footnote_applies'] == True).sum())}")
    print(f"Saved -> {TEST_SPLIT_FILE}")


if __name__ == "__main__":
    main()
