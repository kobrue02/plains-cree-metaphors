"""
Regenerate all TLM training data sources and consolidate into a single parquet file.

Output files:
  data/sentences.parquet                 — columns: text_cree, text_en, source
  data/bloomfield_texts_sentences.parquet — Bloomfield 1934 sentence pool (CLKD + active loop)
    source values:
      "bloomfield_1934"   — Plains Cree-English (Bloomfield 1934)
      "edtekla"           — Plains Cree-English (Teodorescu et al. 2022)
      "ojibwe"            — Ojibwe-English (Jones & Michelson 1917)
      "okimasis"          — Plains Cree-English (Okimāsis 2018 textbook)
      "bloomfield_1930"   — Plains Cree-English (Bloomfield 1930, paragraph-aligned)

Usage:
  python scripts/build_tlm_corpus.py
  python scripts/build_tlm_corpus.py --skip-bloomfield-scrape   # if parquet already exists
  python scripts/build_tlm_corpus.py --skip-okimasis     # skip PDF extraction
  python scripts/build_tlm_corpus.py --skip-bloomfield-1930
"""

from __future__ import annotations
import argparse
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd

BLOOMFIELD_CSV   = "data/bloomfield_texts.parquet"
BLOOMFIELD_SENTS = "data/bloomfield_texts_sentences.parquet"
OJIBWE_TXT       = "data/ojibwatextscoll07jonerich_djvu.txt"
OKIMASIS_PDF        = "data/creelanguageoftheplainstextbook.pdf"
BLOOMFIELD_1930_PDF = "data/sacred-stories-bloomfield-1930.pdf"
SENTENCES_OUT       = "data/sentences.parquet"

# Internal temp paths used by subprocess-based scrapers (not exported constants)
_OKIMASIS_OUT       = "data/sentences_okimasis.txt"
_BLOOMFIELD_1930_OUT = "data/sentences_bloomfield_1930.txt"


def build_bloomfield_df(skip_scrape: bool = False) -> pd.DataFrame:
    from src.mt.sentence_splitter import ParallelSentenceSplitter

    if not skip_scrape:
        from src.scrapers.scrape_bloomfield import BloomfieldScraper
        print("Scraping Bloomfield texts ...")
        BloomfieldScraper().scrape(output=BLOOMFIELD_CSV)

    df = pd.read_parquet(BLOOMFIELD_CSV)
    splitter = ParallelSentenceSplitter(df)
    sent_df = splitter.split()
    # Write sentence pool used by CLKD and the active annotation loop
    sent_df.to_parquet(BLOOMFIELD_SENTS, index=False)
    print(f"  → {BLOOMFIELD_SENTS} ({len(sent_df):,} sentences)")
    result = sent_df[["text_cree", "text_en"]].copy()
    result["source"] = "bloomfield_1934"
    return result


def build_edtekla_df() -> pd.DataFrame:
    import tempfile
    from src.scrapers.scrape_edtekla import EdTeKLAScraper
    print("\nScraping EdTeKLA corpus ...")
    # scrape() must write to a file; use a temp path and discard after
    tmp = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
    tmp.close()
    try:
        pairs = EdTeKLAScraper().scrape(output=tmp.name)
    finally:
        if os.path.exists(tmp.name):
            os.remove(tmp.name)
    # pairs is a list of (cree, english) tuples returned by scrape()
    df = pd.DataFrame(pairs, columns=["text_cree", "text_en"])
    df["source"] = "edtekla"
    return df


def build_ojibwe_df() -> pd.DataFrame:
    from src.scrapers.scrape_ojibwe import parse, to_parallel_df
    print("\nParsing Ojibwe texts ...")
    stories = parse(OJIBWE_TXT)
    df = to_parallel_df(stories)
    df["source"] = "ojibwe"
    return df


def build_okimasis_df() -> pd.DataFrame:
    import subprocess
    print("\nExtracting Okimāsis textbook pairs ...")
    result = subprocess.run(
        [sys.executable, "scripts/data/scrape_okimasis.py",
         "--pdf", OKIMASIS_PDF, "--out", _OKIMASIS_OUT],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  WARNING: scrape_okimasis.py failed: {result.stderr.strip()}")
        return pd.DataFrame(columns=["text_cree", "text_en", "source"])
    print(result.stdout.strip())
    df = pd.read_csv(
        _OKIMASIS_OUT,
        sep=r"\s*\|\|\|\s*",
        header=None,
        names=["text_cree", "text_en"],
        engine="python",
    )
    df["source"] = "okimasis"
    return df


def build_bloomfield_1930_df() -> pd.DataFrame:
    import subprocess
    print("\nExtracting Bloomfield (1930) paragraph pairs ...")
    result = subprocess.run(
        [sys.executable, "scripts/data/scrape_bloomfield_1930.py",
         "--pdf", BLOOMFIELD_1930_PDF, "--out", _BLOOMFIELD_1930_OUT],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  WARNING: scrape_bloomfield_1930.py failed: {result.stderr.strip()}")
        return pd.DataFrame(columns=["text_cree", "text_en", "source"])
    print(result.stdout.strip())
    df = pd.read_csv(
        _BLOOMFIELD_1930_OUT,
        sep=r"\s*\|\|\|\s*",
        header=None,
        names=["text_cree", "text_en"],
        engine="python",
    )
    df["source"] = "bloomfield_1930"
    return df


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

    dfs: list[pd.DataFrame] = []

    # Bloomfield 1934
    dfs.append(build_bloomfield_df(skip_scrape=args.skip_bloomfield_scrape))

    # EdTeKLA
    if not args.skip_edtekla:
        dfs.append(build_edtekla_df())
    else:
        print("EdTeKLA: skipped (--skip-edtekla)")
        dfs.append(pd.DataFrame(columns=["text_cree", "text_en", "source"]))

    # Ojibwe
    if not args.skip_ojibwe:
        dfs.append(build_ojibwe_df())
    else:
        print("Ojibwe: skipped (--skip-ojibwe)")
        dfs.append(pd.DataFrame(columns=["text_cree", "text_en", "source"]))

    # Okimasis
    if not args.skip_okimasis:
        dfs.append(build_okimasis_df())
    else:
        print("Okimāsis: skipped (--skip-okimasis)")
        dfs.append(pd.DataFrame(columns=["text_cree", "text_en", "source"]))

    # Bloomfield 1930
    if not args.skip_bloomfield_1930:
        if os.path.exists(BLOOMFIELD_1930_PDF):
            dfs.append(build_bloomfield_1930_df())
        else:
            print(f"Bloomfield 1930: PDF not found at {BLOOMFIELD_1930_PDF}, skipping")
            dfs.append(pd.DataFrame(columns=["text_cree", "text_en", "source"]))
    else:
        print("Bloomfield 1930: skipped (--skip-bloomfield-1930)")
        dfs.append(pd.DataFrame(columns=["text_cree", "text_en", "source"]))

    combined = pd.concat(dfs, ignore_index=True)
    os.makedirs(os.path.dirname(SENTENCES_OUT) or ".", exist_ok=True)
    combined.to_parquet(SENTENCES_OUT, index=False)

    counts = {src: int((combined["source"] == src).sum())
              for src in ["bloomfield_1934", "edtekla", "ojibwe", "okimasis", "bloomfield_1930"]}

    print(f"\n{'─'*50}")
    print(f"  Bloomfield (1934)          {counts['bloomfield_1934']:>6,}")
    print(f"  EdTeKLA / Teodorescu 2022  {counts['edtekla']:>6,}")
    print(f"  Ojibwe / Jones 1917        {counts['ojibwe']:>6,}")
    print(f"  Okimāsis (2018)            {counts['okimasis']:>6,}")
    print(f"  Bloomfield (1930)          {counts['bloomfield_1930']:>6,}")
    print(f"  {'─'*30}")
    print(f"  Combined                   {len(combined):>6,}")
    print(f"  → {SENTENCES_OUT}")


if __name__ == "__main__":
    main()
