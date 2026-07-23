"""Print summary statistics for the final figurative-language annotated dataset (gold + silver, deduplicated) for the writeup."""

from __future__ import annotations
import argparse, os, re, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd

GOLD_FILE   = "data/figurative/bloomfield_annotated.parquet"
SILVER_FILE = "data/figurative/deepseek_labels.parquet"
POOL_FILE   = "data/bloomfield_texts_sentences.parquet"

LABELS = ["literal", "idiom", "metaphor", "simile"]

_WORD_RE = re.compile(r"[^\s]+")


def _word_count(text: str) -> int:
    return len(text.split())


def _vocab(texts: "pd.Series") -> set[str]:
    vocab: set[str] = set()
    for t in texts.dropna():
        for w in _WORD_RE.findall(str(t).lower().strip(".,;:!?\"'()")):
            vocab.add(w.strip(".,;:!?\"'()"))
    return vocab


def _load_manuscript_lookup() -> pd.Series:
    """The authoritative Cree-sentence -> manuscript (bloomfield_1934/1930) map.

    NOTE: bloomfield_annotated.parquet's own 'source_file' column is a story-title
    slug (e.g. 'sacred-stories-14-...', 'pct-08-...') from bloomfield_texts.parquet
    — NOT a reliable proxy for which manuscript a sentence is from. Some stories
    *within* Bloomfield's 1934 published Plains Cree Texts are themselves titled
    "Sacred Stories of X", which collides with the unrelated, separately-scraped
    1930 unpublished manuscript "Sacred Stories of the Sweet Grass Cree". Always
    resolve manuscript from data/bloomfield_texts_sentences.parquet's source_file
    instead, which is the plain bloomfield_1934/bloomfield_1930 label used
    everywhere else in the pipeline (e.g. deepseek_label_pool.py's exclusion filter).
    """
    pool = pd.read_parquet(POOL_FILE)[["text_cree", "source_file"]].copy()
    pool["text_cree"] = pool["text_cree"].astype(str).str.strip()
    return pool.drop_duplicates("text_cree").set_index("text_cree")["source_file"]


def load_gold(footnoted_only: bool) -> pd.DataFrame:
    if not os.path.exists(GOLD_FILE):
        print(f"[warn] gold file not found: {GOLD_FILE}")
        return pd.DataFrame(columns=["text_cree", "text_en", "label", "source_file", "manuscript"])
    df = pd.read_parquet(GOLD_FILE)
    if footnoted_only:
        df = df[df["footnote_applies"] == True]
    df = df[["text_cree", "text_en", "label", "source_file"]].copy()
    df["annotation_type"] = "gold"
    df["text_cree"] = df["text_cree"].astype(str).str.strip()
    lookup = _load_manuscript_lookup()
    df["manuscript"] = df["text_cree"].map(lookup)
    unresolved = df["manuscript"].isna().sum()
    if unresolved:
        print(f"[warn] {unresolved} gold sentences not found in {POOL_FILE} — "
              f"manuscript left blank for those")
    return df


def load_silver(silver_file: str = SILVER_FILE) -> pd.DataFrame:
    if not os.path.exists(silver_file):
        print(f"[warn] silver file not found (deepseek_label_pool.py hasn't finished/run yet): {silver_file}")
        return pd.DataFrame(columns=["text_cree", "text_en", "label", "source_file", "manuscript"])
    df = pd.read_parquet(silver_file).rename(columns={"deepseek_label": "label"})
    df = df[["text_cree", "text_en", "label"]].copy()
    df["annotation_type"] = "silver"
    df["text_cree"] = df["text_cree"].astype(str).str.strip()
    # Look up the real source rather than assuming — silver's coverage isn't
    # fixed to one source (e.g. it now also covers bloomfield_1930 and edtekla
    # once deepseek_label_pool.py is rerun with a broader --exclude-source).
    lookup = _load_manuscript_lookup()
    df["source_file"] = df["text_cree"].map(lookup)
    df["manuscript"]  = df["source_file"]
    return df


def build_final(footnoted_only: bool, silver_file: str = SILVER_FILE) -> pd.DataFrame:
    gold   = load_gold(footnoted_only)
    silver = load_silver(silver_file)
    combined = pd.concat([gold, silver], ignore_index=True)
    combined["text_cree"] = combined["text_cree"].astype(str).str.strip()
    # gold appears first -> keep="first" makes gold win on overlap
    combined = combined.drop_duplicates(subset=["text_cree"], keep="first")
    return combined


def report(name: str, df: pd.DataFrame) -> None:
    print(f"\n{'='*64}")
    print(f"  {name}  (n={len(df):,})")
    print(f"{'='*64}")
    if df.empty:
        print("  (empty — nothing to report)")
        return

    print("\nLabel distribution:")
    counts = df["label"].value_counts()
    for label in LABELS:
        n = int(counts.get(label, 0))
        print(f"  {label:10s}: {n:5,}  ({n/len(df):.1%})")

    if "annotation_type" in df.columns and df["annotation_type"].nunique() > 1:
        print("\nSource distribution (annotation type):")
        for k, v in df["annotation_type"].value_counts().items():
            print(f"  {k:10s}: {v:5,}  ({v/len(df):.1%})")

    if "manuscript" in df.columns:
        print("\nSource distribution (text source):")
        for k, v in df["manuscript"].value_counts().items():
            print(f"  {k:20s}: {v:5,}  ({v/len(df):.1%})")

    cree_lens = df["text_cree"].dropna().apply(_word_count)
    en_lens   = df["text_en"].dropna().apply(_word_count)
    print("\nAverage sentence length (words):")
    print(f"  Cree   : {cree_lens.mean():.2f}  (median {cree_lens.median():.0f})")
    print(f"  English: {en_lens.mean():.2f}  (median {en_lens.median():.0f})")

    cree_vocab = _vocab(df["text_cree"])
    en_vocab   = _vocab(df["text_en"])
    print("\nVocabulary size (unique word types):")
    print(f"  Cree   : {len(cree_vocab):,}")
    print(f"  English: {len(en_vocab):,}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gold-footnoted-only", action="store_true",
                   help="Restrict gold to footnote_applies=True (219 rows) instead of "
                        "all footnoted-paragraph sentences (1,225 rows)")
    p.add_argument("--out", default="data/figurative/final_dataset_stats.parquet",
                   help="Where to save the combined final dataset's per-label/source counts")
    p.add_argument("--silver-file", default=SILVER_FILE,
                   help="Silver annotation file to use (default: %(default)s)")
    args = p.parse_args()

    gold   = load_gold(args.gold_footnoted_only)
    silver = load_silver(args.silver_file)
    final  = build_final(args.gold_footnoted_only, args.silver_file)

    report(f"Gold ({'footnote-verified only' if args.gold_footnoted_only else 'all footnoted-paragraph sentences'})", gold)
    report("Silver (dictionary-grounded, un-footnoted majority)", silver)
    report("FINAL (gold + silver, deduplicated, gold wins on overlap)", final)

    if not final.empty:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        summary = final.groupby(["annotation_type", "label"]).size().reset_index(name="count")
        summary.to_parquet(args.out, index=False)
        print(f"\nSaved label/source breakdown → {args.out}")


if __name__ == "__main__":
    main()
