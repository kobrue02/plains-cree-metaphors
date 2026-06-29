"""
Split Bloomfield (1930) paragraph pairs into Cree sentences and append to
the active annotation pool (data/bloomfield_texts_sentences.csv).

The 1930 "Sacred Stories" corpus contains oral narrative — trickster tales and
sacred stories — which are figurative-language-rich compared to the 1934
interlinear gloss corpus. Adding these sentences extends the pool with data
that is more likely to yield idiom/metaphor/simile instances.

Sentence splitting uses `. ` as a boundary; minimum length and alpha-ratio
filters remove OCR noise. The English paragraph is kept as a loose reference
gloss for all sentences extracted from it.

Usage:
  python scripts/add_bloomfield_1930_to_pool.py
  python scripts/add_bloomfield_1930_to_pool.py --dry-run
  python scripts/add_bloomfield_1930_to_pool.py --src data/sentences_bloomfield_1930.txt
"""

from __future__ import annotations
import argparse, re, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd

SRC_FILE  = "data/sentences_bloomfield_1930.txt"
POOL_FILE = "data/bloomfield_texts_sentences.parquet"
SOURCE_ID = "bloomfield_1930"

# Minimum characters for a Cree sentence to be kept
_MIN_LEN   = 20
# Minimum ratio of alphabetic characters (filters OCR garbage)
_MIN_ALPHA = 0.45


def split_cree_sentences(para: str) -> list[str]:
    """
    Split a Cree paragraph into individual sentences.

    Bloomfield 1930 uses `.` to end sentences (same as modern SRO). We split
    on `. ` (period + space) and also on period-at-end. Each fragment is
    stripped and filtered for minimum quality.
    """
    # Normalise whitespace first
    para = re.sub(r"\s+", " ", para).strip()
    # Split on period followed by space or end-of-string
    parts = re.split(r"\.\s+", para)
    sentences = []
    for part in parts:
        part = part.strip().strip('"').strip("'").strip()
        # Re-add the period (split consumed it) unless part already ends with punct
        if part and not part[-1] in ".?!":
            part = part + "."
        if _keep(part):
            sentences.append(part)
    return sentences


def _keep(text: str) -> bool:
    if len(text) < _MIN_LEN:
        return False
    alpha = sum(1 for c in text if c.isalpha())
    if alpha / len(text) < _MIN_ALPHA:
        return False
    # Skip lines that are only English (translator notes that slipped through)
    _EN_WORDS = frozenset({
        "the","he","she","was","had","his","her","and","then","to","of",
        "that","this","it","in","for","on","at","by","with","as","said",
        "not","but","they","them","were","from","when","all","one","have",
    })
    words = re.findall(r"\b[a-z]+\b", text.lower())
    if len(words) >= 6:
        en_ratio = sum(1 for w in words if w in _EN_WORDS) / len(words)
        if en_ratio >= 0.40:
            return False
    return True


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
    src: "str | list[tuple[str, str]]" = SRC_FILE,
    pool_path: str = POOL_FILE,
    dry_run: bool = False,
) -> int:
    """Add Bloomfield 1930 sentences to the annotation pool. Returns count of new sentences.

    Parameters
    ----------
    src:
        Either a file path to a Cree ||| English text file, or a list of
        (cree, english) tuples returned directly by scrape_bloomfield_1930.extract().
    pool_path:
        Path to the parquet pool file to append to.
    dry_run:
        If True, report what would be added without writing.
    """
    if isinstance(src, str):
        if not os.path.exists(src):
            print(f"  [skip] {src} not found")
            return 0
        pairs = load_pairs(src)
        print(f"Loaded {len(pairs)} paragraph pairs from {src}")
    else:
        pairs = src
        print(f"Loaded {len(pairs)} paragraph pairs (in-memory)")

    if not os.path.exists(pool_path):
        print(f"  [skip] pool not found at {pool_path}")
        return 0

    pool = pd.read_parquet(pool_path)
    known = set(pool["text_cree"].dropna().str.strip().tolist())
    print(f"Pool: {len(pool):,} existing sentences")

    gold_file = "data/figurative/annotations.parquet"
    if os.path.exists(gold_file):
        gold = pd.read_parquet(gold_file)
        known |= set(gold["text_cree"].dropna().str.strip().tolist())

    next_para_id = int(pool["paragraph_id"].max()) + 1 if len(pool) > 0 else 0

    new_rows = []
    for para_cree, para_en in pairs:
        sentences = split_cree_sentences(para_cree)
        for sent_id, sent in enumerate(sentences):
            if sent.strip() in known:
                continue
            new_rows.append({
                "paragraph_id": next_para_id,
                "sentence_id":  sent_id,
                "text_cree":    sent,
                "text_en":      para_en,
                "confidence":   None,
                "source_file":  SOURCE_ID,
            })
            known.add(sent.strip())
        next_para_id += 1

    print(f"\nExtracted {len(new_rows)} new sentences (after dedup)")

    if dry_run or not new_rows:
        if new_rows:
            print("\nSample (dry run):")
            for r in new_rows[:3]:
                print(f"  CREE: {r['text_cree'][:80]}")
                print(f"  EN:   {r['text_en'][:80]}")
        return len(new_rows)

    new_df = pd.DataFrame(new_rows)
    merged = pd.concat([pool, new_df], ignore_index=True)

    if "source_file" not in pool.columns:
        merged.loc[merged.index < len(pool), "source_file"] = "bloomfield_1934"

    merged.to_parquet(pool_path, index=False)
    print(f"Pool: {len(pool):,} → {len(merged):,} sentences → {pool_path}")
    return len(new_rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src",     default=SRC_FILE,  help="Bloomfield 1930 pair file")
    p.add_argument("--pool",    default=POOL_FILE,  help="Pool parquet to append to")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    add_to_pool(src=args.src, pool_path=args.pool, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
