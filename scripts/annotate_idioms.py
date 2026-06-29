"""
Scan the unlabeled pool for sentences containing known Plains Cree idioms
and append them to bloomfield_annotated.csv as gold idiom labels.

Usage:
  python scripts/annotate_idioms.py [--pool data/figurative/active_pool.csv]
  python scripts/annotate_idioms.py --dry-run
"""

from __future__ import annotations
import os, sys, argparse, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

IDIOMS_FILE = "data/idioms.txt"
POOL_FILE   = "data/figurative/active_pool.csv"
ANNOT_FILE  = "data/figurative/bloomfield_annotated.csv"


def load_idioms(path: str) -> list[tuple[str, str]]:
    idioms = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "|||" not in line:
                continue
            cree, english = line.split("|||", 1)
            idioms.append((cree.strip(), english.strip()))
    return idioms


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def scan(pool: pd.DataFrame, idioms: list[tuple[str, str]]) -> list[dict]:
    rows = []
    for cree_phrase, english_meaning in idioms:
        pattern = _normalise(cree_phrase)
        mask    = pool["text_cree"].apply(lambda t: pattern in _normalise(str(t)))
        matches = pool[mask]
        print(f"  {cree_phrase!r:50s} → {len(matches)} match(es)")
        for _, row in matches.iterrows():
            rows.append({
                "text_cree":      row["text_cree"],
                "text_en":        row["text_en"],
                "label":          "idiom",
                "idiom_phrase":   cree_phrase,
                "idiom_en":       english_meaning,
            })
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool",    default=POOL_FILE)
    p.add_argument("--dry-run", action="store_true",
                   help="Print matches without writing to bloomfield_annotated.csv")
    args = p.parse_args()

    idioms = load_idioms(IDIOMS_FILE)
    print(f"Loaded {len(idioms)} idioms\n")

    if not os.path.exists(args.pool):
        sys.exit(f"Pool file not found: {args.pool}\nRun 'scripts/active_loop.py infer' first.")

    pool = pd.read_csv(args.pool, encoding="utf-8-sig")

    gold     = pd.read_csv(ANNOT_FILE, encoding="utf-8-sig")
    known    = set(gold["text_cree"].dropna().str.strip().tolist())
    pool_new = pool[~pool["text_cree"].str.strip().isin(known)]
    print(f"Pool: {len(pool):,} total, {len(pool_new):,} not yet in gold set\n")

    print("Scanning for idiom matches:")
    matched = scan(pool_new, idioms)

    # Deduplicate (sentence can match multiple idioms — keep first match)
    seen:   set[str]  = set()
    unique: list[dict] = []
    for row in matched:
        if row["text_cree"] not in seen:
            seen.add(row["text_cree"])
            unique.append(row)

    print(f"\nFound {len(unique)} unique sentences containing known idioms")

    if args.dry_run or not unique:
        if unique:
            print("\nMatched sentences (dry run):")
            for r in unique:
                print(f"  [{r['idiom_phrase']}] {r['text_cree'][:80]}")
        return

    # Build rows that match bloomfield_annotated.csv schema
    new_rows = pd.DataFrame([{
        "paragraph_id":    None,
        "sentence_id":     None,
        "source_file":     "idioms.txt",
        "text_cree":       r["text_cree"],
        "text_en":         r["text_en"],
        "label":           "idiom",
        "label_raw":       r["idiom_phrase"],
        "footnote_applies": False,
        "rationale":       f"Matched idiom: {r['idiom_phrase']} = {r['idiom_en']}",
        "footnote_en":     "",
    } for r in unique])

    merged = pd.concat([gold, new_rows], ignore_index=True)
    merged.to_csv(ANNOT_FILE, index=False, encoding="utf-8-sig")

    print(f"\nAppended {len(unique)} idiom sentences → {ANNOT_FILE}")
    print(f"Gold set: {len(gold):,} → {len(merged):,} sentences")
    print(f"Label dist: {merged['label'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
