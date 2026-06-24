"""
Build a combined sentence-level dataset from VUA20, MAGPIE, and FLUTE.

Labels
------
  0 = literal  — no figurative usage (VUA20 sentences with no metaphor token;
                 MAGPIE sentences used literally)
  1 = idiom    — idiomatic MWE usage (MAGPIE figurative; FLUTE Idiom)
  2 = metaphor — at least one metaphorical token (VUA20); FLUTE Metaphor
  3 = simile   — explicit simile construction (FLUTE Simile)

Sources
-------
  VUA20  — token-level → aggregated to sentence level (any metaphor ⇒ class 2)
  MAGPIE — sentence-level idiom detection (figurative ⇒ class 1)
  FLUTE  — NLI dataset; we use the *hypothesis* of Entailment rows for
           Idiom/Metaphor/Simile types only.  Sarcasm and CreativeParaphrase
           are excluded (sarcasm is absent from Bloomfield annotations).
"""

from __future__ import annotations
from collections import defaultdict

import pandas as pd
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast

from src.figurative.config import FigurativeConfig

LITERAL  = 0
IDIOM    = 1
METAPHOR = 2
SIMILE   = 3

LABEL_NAMES = ["literal", "idiom", "metaphor", "simile"]
NUM_LABELS  = len(LABEL_NAMES)


# ── raw loaders ───────────────────────────────────────────────────────────────

def _load_vua20_sentences(split: str) -> list[dict]:
    rows = load_dataset("CreativeLang/vua20_metaphor")[split]
    buckets: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        buckets[row["sentence"]].append(row["label"])
    result = []
    for sentence, labels in buckets.items():
        label = METAPHOR if any(l == 1 for l in labels) else LITERAL
        result.append({"text": sentence, "label": label})
    return result


def _load_flute() -> tuple[list[dict], list[dict]]:
    """Extract figurative sentences from FLUTE (Entailment rows only).

    Hypothesis of Entailment rows is the correctly-used figurative expression.
    Contradiction rows are skipped (hypothesis misuses figurative language).
    """
    _type_map = {"Idiom": IDIOM, "Metaphor": METAPHOR, "Simile": SIMILE}
    ds = load_dataset("ColumbiaNLP/FLUTE")["train"]
    records = []
    for row in ds:
        if row["label"] != "Entailment":
            continue
        label = _type_map.get(row["type"])
        if label is None:
            continue
        records.append({"text": row["hypothesis"], "label": label})
    train_data, test_data = train_test_split(
        records,
        test_size=0.2,
        random_state=42,
        stratify=[r["label"] for r in records],
    )
    return train_data, test_data


def _load_magpie() -> tuple[list[dict], list[dict]]:
    path = hf_hub_download("gsarti/magpie", "magpie.tsv", repo_type="dataset")
    df = pd.read_csv(path, sep="\t")
    records = [
        {
            "text":  row["sentence"],
            "label": IDIOM if row["usage"] == "figurative" else LITERAL,
        }
        for _, row in df.iterrows()
    ]
    train_data, test_data = train_test_split(
        records,
        test_size=0.2,
        random_state=42,
        stratify=[r["label"] for r in records],
    )
    return train_data, test_data


# ── dataset ───────────────────────────────────────────────────────────────────

class FigurativeDataset(Dataset):
    """Sentence-level 3-class dataset tokenised for sequence classification."""

    def __init__(
        self,
        records: list[dict],
        tokenizer: PreTrainedTokenizerFast,
        config: FigurativeConfig,
    ):
        self.encodings = []
        for r in records:
            enc = tokenizer(
                r["text"],
                truncation=True,
                max_length=config.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            self.encodings.append({
                "input_ids":      enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels":         torch.tensor(r["label"], dtype=torch.long),
            })

    def __len__(self) -> int:
        return len(self.encodings)

    def __getitem__(self, idx: int) -> dict:
        return self.encodings[idx]


def build_datasets(
    tokenizer: PreTrainedTokenizerFast,
    config: FigurativeConfig,
) -> tuple[FigurativeDataset, FigurativeDataset]:
    """Return (train_dataset, test_dataset) combining VUA20, MAGPIE, and FLUTE."""
    vua20_train               = _load_vua20_sentences("train")
    vua20_test                = _load_vua20_sentences("test")
    magpie_train, magpie_test = _load_magpie()
    flute_train,  flute_test  = _load_flute()

    train_records = vua20_train + magpie_train + flute_train
    test_records  = vua20_test  + magpie_test  + flute_test

    print(f"[data] train: {len(train_records):,} sentences "
          f"(VUA20={len(vua20_train):,}, MAGPIE={len(magpie_train):,}, "
          f"FLUTE={len(flute_train):,})")
    print(f"[data] test : {len(test_records):,} sentences "
          f"(VUA20={len(vua20_test):,}, MAGPIE={len(magpie_test):,}, "
          f"FLUTE={len(flute_test):,})")

    return (
        FigurativeDataset(train_records, tokenizer, config),
        FigurativeDataset(test_records,  tokenizer, config),
    )


def class_weights_from(dataset: FigurativeDataset) -> torch.Tensor:
    """Inverse-frequency weights for the 4-class imbalance."""
    counts = torch.zeros(NUM_LABELS)
    for item in dataset:
        counts[item["labels"].item()] += 1
    total = counts.sum()
    weights = total / (NUM_LABELS * counts)
    parts = ", ".join(f"{LABEL_NAMES[i]}={weights[i]:.3f}" for i in range(NUM_LABELS))
    print(f"[data] class weights — {parts}")
    return weights
