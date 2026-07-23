"""
Build the Plains Cree figurative-language dataset and push it to the Hub.

Splits:
  gold   — data/figurative/bloomfield_annotated.parquet rows where Bloomfield's own
           footnote commentary directly applies (footnote_applies == True), plus a
           handful of supplementary idiom/metaphor examples from external sources
           (see the `source_file` column). This is the closest thing to a
           human-verified label this project has.
  silver — everything else in the corpus (Bloomfield 1934, Bloomfield 1930, and EdTeKLA;
           Ojibwe is excluded as a different language): each sentence's English gloss is
           read against itwêwina dictionary entries for its content words by an LLM
           (data/figurative/deepseek_labels_qwen_qwen3.5-122b-a10b.parquet).

Usage:
  python scripts/hub/push_dataset.py --dry-run
  python scripts/hub/push_dataset.py
  python scripts/hub/push_dataset.py --out-dir data/figurative/dataset

Requires: huggingface-cli login (or HF_TOKEN env var set), unless --dry-run.
"""

from __future__ import annotations
import argparse, os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd

from src.figurative.data import LABEL_NAMES

ANNOT_FILE  = "data/figurative/bloomfield_annotated.parquet"
SILVER_FILE = "data/figurative/deepseek_labels_qwen_qwen3.5-122b-a10b.parquet"
REPO_ID     = "KonradBRG/plains-cree-figurative"

GOLD_COLUMNS = [
    "paragraph_id", "sentence_id", "source_file", "text_cree", "text_en",
    "label", "footnote_en", "rationale",
]
SILVER_COLUMNS = [
    "paragraph_id", "sentence_id", "text_cree", "text_en", "label",
    "rationale",
]


def build_gold() -> pd.DataFrame:
    df = pd.read_parquet(ANNOT_FILE)
    gold = df[df["footnote_applies"] == True]
    return gold[GOLD_COLUMNS].reset_index(drop=True)


def build_silver(gold: pd.DataFrame) -> pd.DataFrame:
    # Every non-gold Cree sentence (Bloomfield 1934/1930 + EdTeKLA) is labeled by
    # the same dictionary-grounded LLM procedure.
    silver = pd.read_parquet(SILVER_FILE)[
        ["paragraph_id", "sentence_id", "text_cree", "text_en", "deepseek_label", "reasoning"]
    ].rename(columns={"deepseek_label": "label", "reasoning": "rationale"})

    # safety net: silver should never overlap gold, but check anyway.
    gold_keys = set(zip(gold["paragraph_id"], gold["sentence_id"]))
    keep = [
        (pid, sid) not in gold_keys
        for pid, sid in zip(silver["paragraph_id"], silver["sentence_id"])
    ]
    n_dropped = len(silver) - sum(keep)
    if n_dropped:
        print(f"  [warn] dropped {n_dropped} silver rows that overlap gold — check upstream exclusion logic")
    silver = silver[keep].reset_index(drop=True)

    return silver[SILVER_COLUMNS]


def build_card(repo_id: str, gold: pd.DataFrame, silver: pd.DataFrame) -> "DatasetCard":
    from huggingface_hub import DatasetCard, DatasetCardData

    def dist(df: pd.DataFrame) -> str:
        counts = df["label"].value_counts()
        return "  \n".join(f"`{label}`: {counts.get(label, 0):,}" for label in LABEL_NAMES)

    card_data = DatasetCardData(
        language              = ["crk", "en"],
        license               = "cc-by-4.0",
        annotations_creators  = ["machine-generated"],
        multilinguality       = "translation",
        task_categories       = ["text-classification"],
        pretty_name           = "Plains Cree Figurative Language Detection",
        # gold/silver have different schemas, so they're pushed as separate
        # configs (see main()) rather than splits of one config — this section
        # is what tells load_dataset(repo_id, "gold"/"silver") where to look.
        # Must be declared explicitly: pushing this card overwrites the
        # auto-generated README from Dataset.push_to_hub(), which would
        # otherwise carry this same section automatically.
        configs=[
            {"config_name": "gold",   "data_files": [{"split": "gold",   "path": "gold/gold-*.parquet"}]},
            {"config_name": "silver", "data_files": [{"split": "silver", "path": "silver/silver-*.parquet"}]},
        ],
    )

    content = f"""\
---
{ card_data.to_yaml() }
---

# Plains Cree Figurative Language Detection

Sentence-level 4-class figurative language labels (`literal` / `idiom` / `metaphor` / `simile`)
for Plains Cree (crk), paired with English translations, drawn from Leonard Bloomfield's
1934 *Plains Cree Texts* plus a handful of supplementary examples (see `gold`'s `source_file`
column).

There is no native figurative-language annotation for Plains Cree, so the two splits reflect
two different provenances rather than a conventional train/test split:

## Splits

### `gold` ({len(gold):,} sentences)

Sentences whose figurative-language label is directly supported by Bloomfield's own footnote
commentary, plus a small number of supplementary idiom/metaphor examples from other sources
(distinguished via `source_file`). This is the closest thing to a human-verified label in this
dataset.

**Label distribution:**
{dist(gold)}

**Columns:** `paragraph_id`, `sentence_id`, `source_file`, `text_cree`, `text_en`,
`label`, `footnote_en` (Bloomfield's original footnote, where applicable), `rationale`.

### `silver` ({len(silver):,} sentences)

Every other Cree sentence in the corpus (Bloomfield's 1934 and 1930 texts, plus the EdTeKLA
Cree Corpus). Each sentence is labeled by an LLM reading its English gloss against itwêwina
dictionary entries for its content words — not a verified label; `rationale` carries the
model's justification.

**Label distribution:**
{dist(silver)}

**Columns:** `paragraph_id`, `sentence_id`, `text_cree`, `text_en`, `label`, `rationale`.

## Usage

```python
from datasets import load_dataset

gold   = load_dataset("{repo_id}", split="gold")
silver = load_dataset("{repo_id}", split="silver")
```

## Limitations

- `gold` is small and drawn mostly from footnoted paragraphs, which are not a random sample of
  the corpus — Bloomfield tended to footnote passages he found linguistically noteworthy.
- `silver` labels are not human-verified.

## Citation

If you use this dataset, please cite the associated thesis/paper (TBD), and the
underlying sources it draws on:

- Bloomfield, L. (1934). *Plains Cree Texts*. American Ethnological Society.
- Bloomfield, L. (1930). *Sacred Stories of the Sweet Grass Cree*. Unpublished manuscript.
- Teodorescu, D., Matalski, J., Lothian, D., Barbosa, D., & Demmans Epp, C. (2022). "Cree
  Corpus: A Collection of nêhiyawêwin Resources." In *Proceedings of the 60th Annual Meeting
  of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 6354–6364.
  https://aclanthology.org/2022.acl-long.440/
- Napoleon, A. (2014). *Key Terms and Concepts for Exploring Nîhiyaw Tâpisinowin, the Cree
  Worldview*. Master's thesis, University of Victoria. http://hdl.handle.net/1828/5820
- Ogg, A. (2024). "Beginning a collection of Cree idioms." Cree Literacy Network.
  https://creeliteracy.org/2024/04/01/beginning-a-collection-of-cree-idioms/
- Alberta Language Technology Lab. *itwêwina: Plains Cree Dictionary*. University of
  Alberta. https://itwewina.altlab.app/ (used for silver-label dictionary grounding;
  no single canonical citation exists for the dictionary itself)
- Qwen Team. (2026). *Qwen3.5-Omni Technical Report*. arXiv:2604.15804. (used to produce
  the silver labels)
"""
    return DatasetCard(content)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", default=REPO_ID)
    p.add_argument("--out-dir", default=None,
                   help="Also save gold.parquet / silver.parquet here")
    p.add_argument("--dry-run", action="store_true",
                   help="Build both splits and print summary stats without pushing to the Hub")
    args = p.parse_args()

    gold = build_gold()
    print(f"gold split   : {len(gold):,} sentences  |  labels: {gold['label'].value_counts().to_dict()}")

    silver = build_silver(gold)
    print(f"silver split : {len(silver):,} sentences  |  labels: {silver['label'].value_counts().to_dict()}")

    card = build_card(args.repo_id, gold, silver)

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        gold.to_parquet(f"{args.out_dir}/gold.parquet", index=False)
        silver.to_parquet(f"{args.out_dir}/silver.parquet", index=False)
        card.save(f"{args.out_dir}/README.md")
        print(f"Saved parquet + README → {args.out_dir}/")

    if args.dry_run:
        print("\n--dry-run: not pushing to the Hub.")
        return

    # gold and silver intentionally have different schemas (distinct provenance
    # columns, not a conventional train/test split), so DatasetDict.push_to_hub
    # (which requires identical features across splits) doesn't apply — push each
    # as its own config instead, which the datasets library supports natively.
    from datasets import Dataset
    Dataset.from_pandas(gold, preserve_index=False).push_to_hub(
        args.repo_id, config_name="gold", split="gold"
    )
    Dataset.from_pandas(silver, preserve_index=False).push_to_hub(
        args.repo_id, config_name="silver", split="silver"
    )
    card.push_to_hub(args.repo_id, repo_type="dataset")
    print(f"\nPushed → https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
