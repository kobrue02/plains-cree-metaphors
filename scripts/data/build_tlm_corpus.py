"""
Regenerate all TLM training data sources as separate files, then concatenate.

Output files:
  data/sentences_bloomfield.txt      — Plains Cree-English (Bloomfield 1934)
  data/sentences_edtekla.txt         — Plains Cree-English (Teodorescu et al. 2022)
  data/sentences_ojibwe.txt          — Ojibwe-English (Jones & Michelson 1917)
  data/sentences_okimasis.txt        — Plains Cree-English (Okimāsis 2018 textbook)
  data/sentences_bloomfield_1930.txt — Plains Cree-English (Bloomfield 1930, paragraph-aligned)
  data/sentences_combined.txt        — all five concatenated

Usage:
  python scripts/build_tlm_corpus.py
  python scripts/build_tlm_corpus.py --skip-bloomfield   # if CSV already exists
  python scripts/build_tlm_corpus.py --skip-okimasis     # skip PDF extraction
  python scripts/build_tlm_corpus.py --skip-bloomfield-1930
"""

from __future__ import annotations
import argparse
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


BLOOMFIELD_CSV  = "data/bloomfield_texts.parquet"
BLOOMFIELD_OUT  = "data/sentences_bloomfield.txt"
EDTEKLA_OUT     = "data/sentences_edtekla.txt"
OJIBWE_TXT      = "data/ojibwatextscoll07jonerich_djvu.txt"
OJIBWE_CSV      = "data/ojibwe_texts_aligned.parquet"
OJIBWE_OUT      = "data/sentences_ojibwe.txt"
OKIMASIS_PDF       = "data/creelanguageoftheplainstextbook.pdf"
OKIMASIS_OUT       = "data/sentences_okimasis.txt"
BLOOMFIELD_1930_PDF = "data/sacred-stories-bloomfield-1930.pdf"
BLOOMFIELD_1930_OUT = "data/sentences_bloomfield_1930.txt"
COMBINED_OUT       = "data/sentences_combined.txt"


def build_bloomfield(skip_scrape: bool = False) -> int:
    import pandas as pd
    from src.mt.sentence_splitter import ParallelSentenceSplitter

    if not skip_scrape:
        from src.scrapers.scrape_bloomfield import BloomfieldScraper
        print("Scraping Bloomfield texts ...")
        BloomfieldScraper().scrape(output=BLOOMFIELD_CSV)

    df = pd.read_parquet(BLOOMFIELD_CSV)
    splitter = ParallelSentenceSplitter(df)
    splitter.write(BLOOMFIELD_OUT)
    return sum(1 for _ in open(BLOOMFIELD_OUT))


def build_edtekla() -> int:
    from src.scrapers.scrape_edtekla import EdTeKLAScraper
    print("\nScraping EdTeKLA corpus ...")
    pairs = EdTeKLAScraper().scrape(output=EDTEKLA_OUT)
    return len(pairs)


def build_okimasis() -> int:
    import subprocess, sys
    print("\nExtracting Okimāsis textbook pairs ...")
    result = subprocess.run(
        [sys.executable, "scripts/data/scrape_okimasis.py",
         "--pdf", OKIMASIS_PDF, "--out", OKIMASIS_OUT],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  WARNING: scrape_okimasis.py failed: {result.stderr.strip()}")
        return 0
    print(result.stdout.strip())
    return sum(1 for _ in open(OKIMASIS_OUT))


def build_bloomfield_1930() -> int:
    import subprocess, sys
    print("\nExtracting Bloomfield (1930) paragraph pairs ...")
    result = subprocess.run(
        [sys.executable, "scripts/data/scrape_bloomfield_1930.py",
         "--pdf", BLOOMFIELD_1930_PDF, "--out", BLOOMFIELD_1930_OUT],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  WARNING: scrape_bloomfield_1930.py failed: {result.stderr.strip()}")
        return 0
    print(result.stdout.strip())
    return sum(1 for _ in open(BLOOMFIELD_1930_OUT))


def build_ojibwe() -> int:
    from src.scrapers.scrape_ojibwe import parse, to_parquet, to_parallel
    print("\nParsing Ojibwe texts ...")
    stories = parse(OJIBWE_TXT)
    to_parquet(stories, OJIBWE_CSV)
    to_parallel(stories, OJIBWE_OUT)
    return sum(1 for _ in open(OJIBWE_OUT))


def concatenate(paths: list[str], out: str) -> int:
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    total = 0
    with open(out, "w", encoding="utf-8") as fout:
        for path in paths:
            with open(path, encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line)
                    total += 1
    return total


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--skip-bloomfield-scrape", action="store_true",
                   help=f"Skip re-scraping; use existing {BLOOMFIELD_CSV}")
    p.add_argument("--skip-edtekla",          action="store_true")
    p.add_argument("--skip-ojibwe",           action="store_true")
    p.add_argument("--skip-okimasis",         action="store_true")
    p.add_argument("--skip-bloomfield-1930",  action="store_true")
    args = p.parse_args()

    counts = {}

    if not args.skip_bloomfield_scrape or not os.path.exists(BLOOMFIELD_OUT):
        counts["bloomfield"] = build_bloomfield(skip_scrape=args.skip_bloomfield_scrape)
    else:
        counts["bloomfield"] = sum(1 for _ in open(BLOOMFIELD_OUT))
        print(f"Bloomfield: using existing {BLOOMFIELD_OUT} ({counts['bloomfield']:,} pairs)")

    if not args.skip_edtekla:
        counts["edtekla"] = build_edtekla()
    else:
        counts["edtekla"] = sum(1 for _ in open(EDTEKLA_OUT))
        print(f"EdTeKLA: using existing {EDTEKLA_OUT} ({counts['edtekla']:,} pairs)")

    if not args.skip_ojibwe:
        counts["ojibwe"] = build_ojibwe()
    else:
        counts["ojibwe"] = sum(1 for _ in open(OJIBWE_OUT))
        print(f"Ojibwe: using existing {OJIBWE_OUT} ({counts['ojibwe']:,} pairs)")

    if not args.skip_okimasis:
        counts["okimasis"] = build_okimasis()
    else:
        counts["okimasis"] = sum(1 for _ in open(OKIMASIS_OUT)) if os.path.exists(OKIMASIS_OUT) else 0
        print(f"Okimāsis: using existing {OKIMASIS_OUT} ({counts['okimasis']:,} pairs)")

    if not args.skip_bloomfield_1930:
        if os.path.exists(BLOOMFIELD_1930_PDF):
            counts["bloomfield_1930"] = build_bloomfield_1930()
        else:
            counts["bloomfield_1930"] = 0
            print(f"Bloomfield 1930: PDF not found at {BLOOMFIELD_1930_PDF}, skipping")
    else:
        counts["bloomfield_1930"] = (
            sum(1 for _ in open(BLOOMFIELD_1930_OUT))
            if os.path.exists(BLOOMFIELD_1930_OUT) else 0
        )
        print(f"Bloomfield 1930: using existing {BLOOMFIELD_1930_OUT} ({counts['bloomfield_1930']:,} pairs)")

    sources = [BLOOMFIELD_OUT, EDTEKLA_OUT, OJIBWE_OUT]
    if counts.get("okimasis", 0) > 0 and os.path.exists(OKIMASIS_OUT):
        sources.append(OKIMASIS_OUT)
    if counts.get("bloomfield_1930", 0) > 0 and os.path.exists(BLOOMFIELD_1930_OUT):
        sources.append(BLOOMFIELD_1930_OUT)

    total = concatenate(sources, COMBINED_OUT)
    counts["combined"] = total

    print(f"\n{'─'*50}")
    print(f"  Bloomfield (1934)          {counts['bloomfield']:>6,}")
    print(f"  EdTeKLA / Teodorescu 2022  {counts['edtekla']:>6,}")
    print(f"  Ojibwe / Jones 1917        {counts['ojibwe']:>6,}")
    print(f"  Okimāsis (2018)            {counts.get('okimasis', 0):>6,}")
    print(f"  Bloomfield (1930)          {counts.get('bloomfield_1930', 0):>6,}")
    print(f"  {'─'*30}")
    print(f"  Combined                   {counts['combined']:>6,}")
    print(f"  → {COMBINED_OUT}")


if __name__ == "__main__":
    main()
