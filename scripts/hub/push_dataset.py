"""
Build the Plains Cree figurative-language dataset and push it to the Hub.

Splits:
  gold   — data/figurative/bloomfield_annotated.parquet rows where Bloomfield's own
           footnote commentary directly applies (footnote_applies == True), plus a
           handful of supplementary idiom/metaphor examples from external sources
           (see the `source_file` column). This is the closest thing to a
           human-verified label this project has, and is never touched by --checkpoint.
  silver — everything else: (a) the remaining Bloomfield sentences whose label was
           inferred from paragraph context rather than a directly-applicable footnote,
           and (b) the un-footnoted majority of the corpus, labeled via DeepSeek +
           itwêwina dictionary evidence (data/figurative/deepseek_labels.parquet).
           The DeepSeek/context-inferred label is always the published `label` column.
           If --checkpoint is given, the classifier's own prediction on the same
           sentences is added as `classifier_label`/`classifier_confidence`/`prob_*`,
           plus an `agree_with_classifier` flag — this is left null/False if no
           checkpoint is given, e.g. before the classifier has been trained.

--checkpoint should be whichever model the ablation study picks as best — there's no
built-in default since that's still being decided. It's optional: the silver split can
be published from DeepSeek's labels alone, with the classifier-agreement columns added
in a later run once a checkpoint exists.

Usage:
  python scripts/hub/push_dataset.py --dry-run                       # gold + silver, no agreement columns yet
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
SILVER_FILE = "data/figurative/deepseek_labels.parquet"
REPO_ID     = "KonradBRG/plains-cree-figurative"

GOLD_COLUMNS = [
    "paragraph_id", "sentence_id", "source_file", "text_cree", "text_en",
    "label", "footnote_en", "rationale",
]
SILVER_COLUMNS = [
    "paragraph_id", "sentence_id", "text_cree", "text_en", "label",
    "annotation_source", "rationale",
    "classifier_label", "classifier_confidence",
    "prob_literal", "prob_idiom", "prob_metaphor", "prob_simile",
    "agree_with_classifier", "model_checkpoint",
]


def build_gold() -> pd.DataFrame:
    df = pd.read_parquet(ANNOT_FILE)
    gold = df[df["footnote_applies"] == True]
    return gold[GOLD_COLUMNS].reset_index(drop=True)


def build_card(repo_id: str, checkpoint: str | None, gold: pd.DataFrame, silver: pd.DataFrame) -> "DatasetCard":
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

    if checkpoint:
        agree_rate = silver["agree_with_classifier"].mean()
        classifier_note = (
            f"[`{checkpoint}`](https://huggingface.co/{checkpoint}) — a classifier trained via "
            f"cross-lingual knowledge distillation (CLKD) from an English figurative-language "
            f"teacher, then calibrated on `gold` — was additionally run on every silver sentence. "
            f"It agrees with the DeepSeek/context-inferred label {agree_rate:.1%} of the time "
            f"(`agree_with_classifier`); disagreement doesn't mean a sentence is unusable, only "
            f"that the two independent label sources diverge on it, which is itself informative — "
            f"we deliberately did not drop these rows (see Limitations)."
        )
    else:
        classifier_note = (
            "No classifier has been run against this split yet, so `classifier_label`, "
            "`classifier_confidence`, `prob_*`, and `agree_with_classifier` are all null. "
            "A later release will add these once a trained classifier is available."
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
dataset — never touched by any classifier.

**Label distribution:**
{dist(gold)}

**Columns:** `paragraph_id`, `sentence_id`, `source_file`, `text_cree`, `text_en`,
`label`, `footnote_en` (Bloomfield's original footnote, where applicable), `rationale`.

### `silver` ({len(silver):,} sentences)

Everything else, from two sources distinguished via `annotation_source`: (a) Bloomfield
sentences whose label was inferred from surrounding paragraph context rather than a directly
applicable footnote (`bloomfield-context`), and (b) the un-footnoted majority of the corpus,
labeled by DeepSeek reading each sentence's English gloss against itwêwina dictionary entries
for its content words (`deepseek-dictionary`). Neither is a verified label — `rationale`
carries the justification for either source.

{classifier_note}

**Label distribution:**
{dist(silver)}

**Columns:** `paragraph_id`, `sentence_id`, `text_cree`, `text_en`, `label`, `annotation_source`,
`rationale`, `classifier_label`, `classifier_confidence`, `prob_literal`/`prob_idiom`/
`prob_metaphor`/`prob_simile`, `agree_with_classifier`, `model_checkpoint`.

## Usage

```python
from datasets import load_dataset

gold   = load_dataset("{repo_id}", split="gold")
silver = load_dataset("{repo_id}", split="silver")

# e.g. only sentences where the classifier and DeepSeek agree
high_confidence = silver.filter(lambda r: r["agree_with_classifier"] == True)
```

## Limitations

- `gold` is small and drawn mostly from footnoted paragraphs, which are not a random sample of
  the corpus — Bloomfield tended to footnote passages he found linguistically noteworthy.
- `silver` labels are not verified. `agree_with_classifier` flags where an independently-trained
  classifier and the DeepSeek/context-inferred label coincide, but we do not filter the dataset
  to agreement-only: disagreement is concentrated on the rare idiom/metaphor classes precisely
  because those are hardest, so dropping it would gut the classes this dataset is meant to help
  with. Use the flag as a confidence signal, not a cleaning step.

## Citation

If you use this dataset, please cite the associated thesis/paper (TBD), and
Bloomfield, L. (1934). *Plains Cree Texts*. American Ethnological Society.
"""
    return DatasetCard(content)


def build_silver(gold: pd.DataFrame, checkpoint: str | None,
                  batch_size: int, max_length: int) -> pd.DataFrame:
    annot = pd.read_parquet(ANNOT_FILE)

    # (a) Bloomfield sentences whose label came from the footnote-reading procedure,
    # but which the footnote doesn't directly confirm (footnote_applies == False).
    context_inferred = annot[annot["footnote_applies"] == False][
        ["paragraph_id", "sentence_id", "text_cree", "text_en", "label", "rationale"]
    ].copy()
    context_inferred["annotation_source"] = "bloomfield-context"

    # (b) the un-footnoted majority, labeled via DeepSeek + itwêwina dictionary evidence.
    deepseek = pd.read_parquet(SILVER_FILE)[
        ["paragraph_id", "sentence_id", "text_cree", "text_en", "deepseek_label", "reasoning"]
    ].rename(columns={"deepseek_label": "label", "reasoning": "rationale"})
    deepseek["annotation_source"] = "deepseek-dictionary"

    silver = pd.concat([context_inferred, deepseek], ignore_index=True)

    # safety net: neither source should overlap gold (deepseek_label_pool.py already
    # excludes all of ANNOT_FILE, and context_inferred is the complementary
    # footnote_applies filter on the same file gold comes from), but check anyway.
    gold_keys = set(zip(gold["paragraph_id"], gold["sentence_id"]))
    keep = [
        (pid, sid) not in gold_keys
        for pid, sid in zip(silver["paragraph_id"], silver["sentence_id"])
    ]
    n_dropped = len(silver) - sum(keep)
    if n_dropped:
        print(f"  [warn] dropped {n_dropped} silver rows that overlap gold — check upstream exclusion logic")
    silver = silver[keep].reset_index(drop=True)

    if checkpoint:
        print(f"Labeling {len(silver):,} silver sentences with {checkpoint} for agreement scoring...")
        model, tokenizer = load_model(checkpoint)
        preds = predict_sentences(
            silver["text_cree"].tolist(), model, tokenizer,
            batch_size=batch_size, max_length=max_length,
        )
        silver["classifier_label"]      = [p["label"] for p in preds]
        silver["classifier_confidence"] = [p["confidence"] for p in preds]
        for name in LABEL_NAMES:
            silver[f"prob_{name}"] = [p[f"prob_{name}"] for p in preds]
        silver["agree_with_classifier"] = silver["classifier_label"] == silver["label"]
        silver["model_checkpoint"] = checkpoint
    else:
        print("No --checkpoint given — publishing silver labels without classifier agreement columns.")
        silver["classifier_label"]      = None
        silver["classifier_confidence"] = None
        for name in LABEL_NAMES:
            silver[f"prob_{name}"] = None
        silver["agree_with_classifier"] = None
        silver["model_checkpoint"]      = None

    return silver[SILVER_COLUMNS]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=None,
                   help="Trained classifier checkpoint to score against the silver labels "
                        "for agreement (optional — omit to publish silver labels alone, "
                        "e.g. before the classifier has been trained)")
    p.add_argument("--repo-id", default=REPO_ID)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--out-dir", default=None,
                   help="Also save gold.parquet / silver.parquet here")
    p.add_argument("--dry-run", action="store_true",
                   help="Build both splits and print summary stats without pushing to the Hub")
    args = p.parse_args()

    gold = build_gold()
    print(f"gold split   : {len(gold):,} sentences  |  labels: {gold['label'].value_counts().to_dict()}")

    silver = build_silver(gold, args.checkpoint, args.batch_size, args.max_length)
    print(f"silver split : {len(silver):,} sentences  |  labels: {silver['label'].value_counts().to_dict()}")
    if args.checkpoint:
        print(f"  agreement with classifier: {silver['agree_with_classifier'].mean():.1%}")

    card = build_card(args.repo_id, args.checkpoint, gold, silver)

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        gold.to_parquet(f"{args.out_dir}/gold.parquet", index=False)
        silver.to_parquet(f"{args.out_dir}/silver.parquet", index=False)
        card.save(f"{args.out_dir}/README.md")
        print(f"Saved parquet + README → {args.out_dir}/")

    if args.dry_run:
        print("\n--dry-run: not pushing to the Hub.")
        return

    from datasets import Dataset, DatasetDict
    ds = DatasetDict({
        "gold":   Dataset.from_pandas(gold, preserve_index=False),
        "silver": Dataset.from_pandas(silver, preserve_index=False),
    })
    ds.push_to_hub(args.repo_id)
    card.push_to_hub(args.repo_id, repo_type="dataset")
    print(f"\nPushed → https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
