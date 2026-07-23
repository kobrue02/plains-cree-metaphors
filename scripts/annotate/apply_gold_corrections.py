"""
Applies the human-verified corrections from verify_gold.py's review cache
onto the canonical gold file, preserving DeepSeek's original label for
provenance so downstream scripts pick up the corrected labels.
"""

from __future__ import annotations
import argparse, json, os, sys, shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd

GOLD_FILE  = "data/figurative/bloomfield_annotated.parquet"
CACHE_JSONL = "data/figurative/gold_verification_cache.jsonl"
BACKUP_FILE = "data/figurative/bloomfield_annotated.pre_human_review.parquet"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gold-file", default=GOLD_FILE)
    p.add_argument("--cache",     default=CACHE_JSONL)
    p.add_argument("--backup",    default=BACKUP_FILE)
    p.add_argument("--dry-run",   action="store_true")
    args = p.parse_args()

    df = pd.read_parquet(args.gold_file)
    if "human_verified" in df.columns:
        sys.exit(f"{args.gold_file} already has a human_verified column — "
                 f"corrections look already applied. Nothing to do.")

    recs = [json.loads(l) for l in open(args.cache)]
    rev = pd.DataFrame(recs).drop_duplicates(subset=["paragraph_id", "sentence_id"], keep="last")
    rev = rev.set_index(["paragraph_id", "sentence_id"])

    df["label_deepseek"]  = df["label"]
    df["human_verified"]  = df.set_index(["paragraph_id", "sentence_id"]).index.isin(rev.index)
    df["human_corrected"] = False

    n_corrected = 0
    for (pid, sid), row in rev.iterrows():
        mask = (df["paragraph_id"] == pid) & (df["sentence_id"] == sid)
        if row["human_label"] != row["deepseek_label"]:
            df.loc[mask, "label"] = row["human_label"]
            df.loc[mask, "human_corrected"] = True
            n_corrected += 1

    n_reviewed = int(df["human_verified"].sum())
    print(f"Reviewed: {n_reviewed}  |  Corrected: {n_corrected}  |  "
          f"Agreement: {(n_reviewed - n_corrected) / n_reviewed:.1%}")

    if args.dry_run:
        print("[dry-run] not writing any files.")
        return

    if os.path.exists(args.backup):
        sys.exit(f"Backup {args.backup} already exists — refusing to overwrite. "
                 f"Delete it manually first if you really mean to redo this.")
    shutil.copy(args.gold_file, args.backup)
    print(f"Backed up original -> {args.backup}")

    df.to_parquet(args.gold_file, index=False)
    print(f"Saved corrected gold labels -> {args.gold_file}")


if __name__ == "__main__":
    main()
