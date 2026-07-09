"""
Assign every annotated sentence to one of K stratified cross-validation folds, once.

calibrate.py can then train with any single fold excluded (--holdout-fold), and
scripts/evals/eval_cv.py aggregates all K folds' held-out predictions into one
evaluation covering the entire annotation pool: every sentence gets a genuinely
held-out prediction, and every sentence is still available for training in the
K-1 folds where it isn't held out.

This exists because a single fixed train/test split doesn't work well here —
there are only ~60 figurative examples (idiom/metaphor/simile) total, so any
split ratio either leaves too few in the test set to mean anything, or leaves
too few in training for calibration to learn the class at all. K-fold avoids
that trade-off by using every example for both roles, just never at the same
time.

Run once:
  python scripts/data/build_cv_folds.py

Refuses to overwrite an existing fold assignment — delete
data/figurative/cv_folds.parquet manually first if you really mean to redraw it
(models already calibrated against specific folds would silently become
inconsistent with a new assignment).
"""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd
from sklearn.model_selection import StratifiedKFold

ANNOT_FILE    = "data/figurative/bloomfield_annotated.parquet"
CV_FOLDS_FILE = "data/figurative/cv_folds.parquet"
LABEL_NAMES   = ["literal", "idiom", "metaphor", "simile"]
N_FOLDS       = 5
SEED          = 42


def main() -> None:
    if os.path.exists(CV_FOLDS_FILE):
        sys.exit(f"{CV_FOLDS_FILE} already exists — refusing to overwrite. "
                 "Delete it manually first if you really mean to redraw the folds.")

    df = pd.read_parquet(ANNOT_FILE)
    df = df.dropna(subset=["text_cree", "label"])
    df = df.drop_duplicates(subset=["text_cree"], keep="first").reset_index(drop=True)
    df["label"] = df["label"].str.strip().str.lower().map(
        lambda x: x if x in LABEL_NAMES else "literal"
    )

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold = pd.Series(-1, index=df.index)
    for i, (_, held_out_idx) in enumerate(skf.split(df, df["label"])):
        fold.iloc[held_out_idx] = i
    df["fold"] = fold

    os.makedirs(os.path.dirname(CV_FOLDS_FILE), exist_ok=True)
    df[["text_cree", "fold"]].to_parquet(CV_FOLDS_FILE, index=False)

    print(f"Assigned {len(df):,} sentences to {N_FOLDS} folds.")
    for i in range(N_FOLDS):
        sub = df[df["fold"] == i]
        n_gold = int((sub["footnote_applies"] == True).sum())
        print(f"  fold {i}: {len(sub):>4} sentences  "
              f"{sub['label'].value_counts().to_dict()}  (gold={n_gold})")
    print(f"Saved -> {CV_FOLDS_FILE}")


if __name__ == "__main__":
    main()
