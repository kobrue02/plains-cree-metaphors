"""
Build the Plains Cree figurative-language dataset and push it to the Hub.

Splits:
  gold      — data/figurative/bloomfield_annotated.parquet rows where Bloomfield's own
              footnote commentary applies (footnote_applies == True). DeepSeek
              read the footnote and assigned the label; this is the closest
              thing to a human-verified label this project has.
  synthetic — every other sentence in the Bloomfield corpus (data/bloomfield_
              texts_sentences.parquet), i.e. anything not already in `gold`,
              labeled by --checkpoint. Includes per-class probabilities and a
              top-label confidence score so downstream users can filter by how
              sure the model was.

--checkpoint should be whichever model the ablation study picks as best —
there's no built-in default since that's still being decided.

Usage:
  python scripts/hub/push_dataset.py --checkpoint KonradBRG/xlm-mlm-plains-cree-en-calibrated
  python scripts/hub/push_dataset.py --checkpoint <best-ablation-checkpoint> --dry-run
  python scripts/hub/push_dataset.py --checkpoint ... --out-dir data/figurative/dataset

Requires: huggingface-cli login (or HF_TOKEN env var set), unless --dry-run.
"""

from __future__ import annotations
import argparse, os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd

from src.figurative.predict import load_model, predict_sentences
from src.figurative.data import LABEL_NAMES

ANNOT_FILE  = "data/figurative/bloomfield_annotated.parquet"
CORPUS_FILE = "data/bloomfield_texts_sentences.parquet"
REPO_ID     = "KonradBRG/plains-cree-figurative"

GOLD_COLUMNS = [
    "paragraph_id", "sentence_id", "source_file", "text_cree", "text_en",
    "label", "footnote_en", "rationale",
]
SYNTHETIC_COLUMNS = [
    "paragraph_id", "sentence_id", "text_cree", "text_en", "label", "confidence",
    "prob_literal", "prob_idiom", "prob_metaphor", "prob_simile",
    "alignment_confidence", "model_checkpoint",
]


def build_gold() -> pd.DataFrame:
    df = pd.read_parquet(ANNOT_FILE)
    gold = df[df["footnote_applies"] == True]
    return gold[GOLD_COLUMNS].reset_index(drop=True)


def build_card(repo_id: str, checkpoint: str, gold: pd.DataFrame, synthetic: pd.DataFrame) -> "DatasetCard":
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
    )

    content = f"""\
---
{ card_data.to_yaml() }
---

# Plains Cree Figurative Language Detection

Sentence-level 4-class figurative language labels (`literal` / `idiom` / `metaphor` / `simile`)
for Plains Cree (crk), paired with English translations, drawn from Leonard Bloomfield's
1934 *Plains Cree Texts*.

There is no native figurative-language annotation for Plains Cree, so the two splits reflect
two different provenances rather than a conventional train/test split:

## Splits

### `gold` ({len(gold):,} sentences)

Sentences from paragraphs where Bloomfield's own footnote commentary discusses figurative
language. An LLM (DeepSeek) read each footnote and translated it into this dataset's 4-class
scheme, then labeled every sentence in that paragraph accordingly. This is the closest thing
to a human-verified label in this dataset — the label traces back to an expert linguist's
own commentary, not to model inference.

**Label distribution:**
{dist(gold)}

**Columns:** `paragraph_id`, `sentence_id`, `source_file` (source text file), `text_cree`, `text_en`,
`label`, `footnote_en` (Bloomfield's original footnote), `rationale` (DeepSeek's justification).

### `synthetic` ({len(synthetic):,} sentences)

Every other sentence in the Bloomfield corpus (i.e. not in `gold`), labeled by
[`{checkpoint}`](https://huggingface.co/{checkpoint}) — a classifier trained via cross-lingual
knowledge distillation (CLKD) from an English figurative-language teacher, then calibrated on
`gold` (see the [KonradBRG](https://huggingface.co/KonradBRG) Hub profile for the full set of
checkpoints and collections). These are model predictions, not verified labels — use
`confidence` / `prob_*` to filter.

**Label distribution:**
{dist(synthetic)}

**Columns:** `paragraph_id`, `sentence_id`, `text_cree`, `text_en`, `label`, `confidence`
(top-label softmax probability), `prob_literal`/`prob_idiom`/`prob_metaphor`/`prob_simile`,
`alignment_confidence` (sentence-pair alignment quality from the Cree/English splitter —
unrelated to the classifier), `model_checkpoint`.

## Usage

```python
from datasets import load_dataset

gold      = load_dataset("{repo_id}", split="gold")
synthetic = load_dataset("{repo_id}", split="synthetic")

# e.g. only confident figurative predictions
confident_figurative = synthetic.filter(lambda r: r["label"] != "literal" and r["confidence"] > 0.8)
```

## Limitations

- `gold` is small and drawn only from footnoted paragraphs, which are not a random sample of
  the corpus — Bloomfield tended to footnote passages he found linguistically noteworthy.
- `synthetic` labels are model predictions from a single checkpoint and inherit its biases;
  softmax confidence is not calibrated to be a true probability.

## Citation

If you use this dataset, please cite the associated thesis/paper (TBD), and
Bloomfield, L. (1934). *Plains Cree Texts*. American Ethnological Society.
"""
    return DatasetCard(content)


def build_synthetic(checkpoint: str, gold: pd.DataFrame, batch_size: int, max_length: int) -> pd.DataFrame:
    corpus = pd.read_parquet(CORPUS_FILE).dropna(subset=["text_cree", "text_en"])
    corpus = corpus.rename(columns={"confidence": "alignment_confidence"})

    gold_keys = set(zip(gold["paragraph_id"], gold["sentence_id"]))
    keep = [
        (pid, sid) not in gold_keys
        for pid, sid in zip(corpus["paragraph_id"], corpus["sentence_id"])
    ]
    corpus = corpus[keep].reset_index(drop=True)
    print(f"Labeling {len(corpus):,} sentences (gold sentences excluded) with {checkpoint} ...")

    model, tokenizer = load_model(checkpoint)
    preds = predict_sentences(
        corpus["text_cree"].tolist(), model, tokenizer,
        batch_size=batch_size, max_length=max_length,
    )

    out = corpus.copy()
    out["label"]      = [p["label"] for p in preds]
    out["confidence"] = [p["confidence"] for p in preds]
    for name in LABEL_NAMES:
        out[f"prob_{name}"] = [p[f"prob_{name}"] for p in preds]
    out["model_checkpoint"] = checkpoint
    return out[SYNTHETIC_COLUMNS]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True,
                   help="Best trained model checkpoint to label the synthetic split with")
    p.add_argument("--repo-id", default=REPO_ID)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--out-dir", default=None,
                   help="Also save gold.parquet / synthetic.parquet here")
    p.add_argument("--dry-run", action="store_true",
                   help="Build both splits and print summary stats without pushing to the Hub")
    args = p.parse_args()

    gold = build_gold()
    print(f"gold split      : {len(gold):,} sentences  |  labels: {gold['label'].value_counts().to_dict()}")

    synthetic = build_synthetic(args.checkpoint, gold, args.batch_size, args.max_length)
    print(f"synthetic split : {len(synthetic):,} sentences  |  labels: {synthetic['label'].value_counts().to_dict()}")

    card = build_card(args.repo_id, args.checkpoint, gold, synthetic)

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        gold.to_parquet(f"{args.out_dir}/gold.parquet", index=False)
        synthetic.to_parquet(f"{args.out_dir}/synthetic.parquet", index=False)
        card.save(f"{args.out_dir}/README.md")
        print(f"Saved parquet + README → {args.out_dir}/")

    if args.dry_run:
        print("\n--dry-run: not pushing to the Hub.")
        return

    from datasets import Dataset, DatasetDict
    ds = DatasetDict({
        "gold":      Dataset.from_pandas(gold, preserve_index=False),
        "synthetic": Dataset.from_pandas(synthetic, preserve_index=False),
    })
    ds.push_to_hub(args.repo_id)
    card.push_to_hub(args.repo_id, repo_type="dataset")
    print(f"\nPushed → https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
