"""
Adds EdTeKLA Cree-English sentence pairs to the active annotation pool so
they flow through the same silver-annotation pipeline as the Bloomfield
material.
"""

from __future__ import annotations
import argparse, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd

SRC_FILE   = "data/sentences_edtekla.txt"
POOL_FILE  = "data/bloomfield_texts_sentences.parquet"
GOLD_FILE  = "data/figurative/bloomfield_annotated.parquet"
SOURCE_ID  = "edtekla"


def load_pairs(path: str) -> list[tuple[str, str]]:
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "|||" not in line:
                continue
            cree, en = line.split("|||", 1)
            pairs.append((cree.strip(), en.strip()))
    return pairs


def add_to_pool(
    src: str = SRC_FILE,
    pool_path: str = POOL_FILE,
    dry_run: bool = False,
) -> int:
    """Add EdTeKLA sentences to the annotation pool. Returns count of new sentences."""
    if not os.path.exists(src):
        print(f"  [skip] {src} not found")
        return 0
    pairs = load_pairs(src)
    print(f"Loaded {len(pairs):,} sentence pairs from {src}")

    if not os.path.exists(pool_path):
        print(f"  [skip] pool not found at {pool_path}")
        return 0

    pool = pd.read_parquet(pool_path)
    known = set(pool["text_cree"].dropna().str.strip().tolist())
    print(f"Pool: {len(pool):,} existing sentences")

    # Exclude sentences already in the gold set so silver never re-adds/re-annotates them
    if os.path.exists(GOLD_FILE):
        gold = pd.read_parquet(GOLD_FILE)
        known |= set(gold["text_cree"].dropna().str.strip().tolist())

    next_para_id = int(pool["paragraph_id"].max()) + 1 if len(pool) > 0 else 0

    new_rows = []
    for cree, en in pairs:
        cree = cree.strip()
        if not cree or cree in known:
            continue
        new_rows.append({
            "paragraph_id": next_para_id,
            "sentence_id":  0,
            "text_cree":    cree,
            "text_en":      en,
            "confidence":   None,
            "source_file":  SOURCE_ID,
        })
        known.add(cree)
        next_para_id += 1

    print(f"\nExtracted {len(new_rows):,} new sentences (after dedup)")

    if dry_run or not new_rows:
        if new_rows:
            print("\nSample (dry run):")
            for r in new_rows[:3]:
                print(f"  CREE: {r['text_cree'][:80]}")
                print(f"  EN:   {r['text_en'][:80]}")
        return len(new_rows)

    new_df = pd.DataFrame(new_rows)
    merged = pd.concat([pool, new_df], ignore_index=True)
    merged.to_parquet(pool_path, index=False)
    print(f"Pool: {len(pool):,} → {len(merged):,} sentences → {pool_path}")
    return len(new_rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src",     default=SRC_FILE,  help="EdTeKLA cree|||english pair file")
    p.add_argument("--pool",    default=POOL_FILE, help="Pool parquet to append to")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    add_to_pool(src=args.src, pool_path=args.pool, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
