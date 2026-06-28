"""
Scan the unlabeled pool for sentences containing known Plains Cree idioms
and write them as high-confidence 'idiom' annotations.

Usage:
  python scripts/annotate_idioms.py [--pool data/figurative/active_pool.csv]
  python scripts/annotate_idioms.py --dry-run
"""

from __future__ import annotations
import os, sys, argparse, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

IDIOMS_FILE  = "data/idioms.txt"
POOL_FILE    = "data/figurative/active_pool.csv"
ANNOT_FILE   = "data/figurative/bloomfield_annotated.csv"
ACTIVE_ANNOT = "data/figurative/active_annotations.csv"


def load_idioms(path: str) -> list[tuple[str, str]]:
    """Return list of (cree_phrase, english_meaning) from idioms.txt."""
    idioms = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "|||" not in line:
                continue
            cree, english = line.split("|||", 1)
            cree    = cree.strip()
            english = english.strip()
            idioms.append((cree, english))
    return idioms


def _normalise(text: str) -> str:
    """Lowercase and collapse whitespace for matching."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def scan(pool: pd.DataFrame, idioms: list[tuple[str, str]]) -> list[dict]:
    rows = []
    for cree_phrase, english_meaning in idioms:
        pattern = _normalise(cree_phrase)
        mask = pool["text_cree"].apply(
            lambda t: pattern in _normalise(str(t))
        )
        matches = pool[mask]
        print(f"  {cree_phrase!r:50s} → {len(matches)} match(es)")
        for _, row in matches.iterrows():
            rows.append({
                "text_cree":   row["text_cree"],
                "text_en":     row["text_en"],
                "label":       "idiom",
                "source":      "idiom_match",
                "model_label": row.get("label", ""),
                "confidence":  row.get("confidence", ""),
                "idiom":       cree_phrase,
                "idiom_en":    english_meaning,
            })
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool",    default=POOL_FILE)
    p.add_argument("--dry-run", action="store_true",
                   help="Print matches without writing to active_annotations.csv")
    args = p.parse_args()

    idioms = load_idioms(IDIOMS_FILE)
    print(f"Loaded {len(idioms)} idioms (above {MIN_PHRASE_LEN}-char threshold)\n")

    if not os.path.exists(args.pool):
        sys.exit(f"Pool file not found: {args.pool}\nRun 'scripts/active_loop.py infer' first.")

    pool = pd.read_csv(args.pool, encoding="utf-8-sig")

    # Exclude sentences already in gold or active annotations
    known: set[str] = set()
    for f in [ANNOT_FILE, ACTIVE_ANNOT]:
        if os.path.exists(f):
            df = pd.read_csv(f, encoding="utf-8-sig")
            known |= set(df["text_cree"].dropna().str.strip().tolist())

    pool_clean = pool[~pool["text_cree"].str.strip().isin(known)]
    print(f"Pool: {len(pool):,} total, {len(pool_clean):,} not yet annotated\n")

    print("Scanning for idiom matches:")
    matched = scan(pool_clean, idioms)

    # Deduplicate (a sentence can match multiple idioms)
    seen: set[str] = set()
    unique = []
    for row in matched:
        if row["text_cree"] not in seen:
            seen.add(row["text_cree"])
            unique.append(row)

    print(f"\nFound {len(unique)} unique sentences containing known idioms")

    if args.dry_run or not unique:
        if unique:
            print("\nMatched sentences (dry run):")
            for r in unique:
                print(f"  [{r['idiom']}] {r['text_cree'][:80]}")
        return

    new_df = pd.DataFrame(unique)
    if os.path.exists(ACTIVE_ANNOT):
        existing = pd.read_csv(ACTIVE_ANNOT, encoding="utf-8-sig")
        # Drop existing entries for the same sentences (avoid duplicates)
        existing = existing[~existing["text_cree"].isin(seen)]
        new_df = pd.concat([existing, new_df], ignore_index=True)

    new_df.to_csv(ACTIVE_ANNOT, index=False, encoding="utf-8-sig")
    print(f"Saved → {ACTIVE_ANNOT}")
    for r in unique:
        print(f"  idiom: {r['text_cree'][:70]}")


if __name__ == "__main__":
    main()
