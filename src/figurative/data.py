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
           Idiom/Metaphor/Simile types only, plus each such row's *premise*
           as a literal example (kept in the same train/test split as its
           hypothesis to avoid the paraphrase pair leaking across splits).
           Sarcasm and CreativeParaphrase are excluded (sarcasm is absent
           from Bloomfield annotations).
"""

from __future__ import annotations
import random
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


def resolve_use_fast(checkpoint: str) -> bool:
    """deberta-v3 fast tokenizers are broken in recent transformers (tiktoken misparses the spm file as bpe)."""
    return "deberta-v3" not in checkpoint.lower()


# ── raw loaders ───────────────────────────────────────────────────────────────

def _load_vua20_sentences(split: str, min_metaphor_tokens: int = 2) -> list[dict]:
    """min_metaphor_tokens=2, not VUA20's original "any single token" rule —
    an isolated metaphorical token is more likely a token-level annotation
    artifact than a sentence a person would actually judge as metaphorical;
    requiring at least two shows sustained metaphorical use instead. This
    (plus the metaphor-count cap in build_datasets()) addresses the teacher
    over-predicting metaphor, traced back to VUA20 supplying ~92% of its
    metaphor training examples via this aggregation rule."""
    rows = load_dataset("CreativeLang/vua20_metaphor")[split]
    buckets: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        buckets[row["sentence"]].append(row["label"])
    result = []
    for sentence, labels in buckets.items():
        label = METAPHOR if sum(l == 1 for l in labels) >= min_metaphor_tokens else LITERAL
        result.append({"text": sentence, "label": label})
    return result


def _load_flute() -> tuple[list[dict], list[dict]]:
    """Extract figurative sentences from FLUTE; uses hypothesis of Entailment rows only
    (Contradiction rows misuse the figure). Each such row's premise is the figure's literal
    counterpart, so we add it as a literal example — kept in the same split as its hypothesis
    to avoid the paraphrase pair leaking across train/test."""
    _type_map = {"Idiom": IDIOM, "Metaphor": METAPHOR, "Simile": SIMILE}
    ds = load_dataset("ColumbiaNLP/FLUTE")["train"]
    pairs = []
    for row in ds:
        if row["label"] != "Entailment":
            continue
        label = _type_map.get(row["type"])
        if label is None:
            continue
        pairs.append({"hypothesis": row["hypothesis"], "premise": row["premise"], "label": label})
    train_pairs, test_pairs = train_test_split(
        pairs,
        test_size=0.2,
        random_state=42,
        stratify=[p["label"] for p in pairs],
    )

    def _to_records(pair_split: list[dict]) -> list[dict]:
        records = []
        for p in pair_split:
            records.append({"text": p["hypothesis"], "label": p["label"]})
            records.append({"text": p["premise"], "label": LITERAL})
        return records

    return _to_records(train_pairs), _to_records(test_pairs)


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

    # Even requiring >=2 metaphor tokens, VUA20 still supplies far more
    # metaphor examples than FLUTE's cleaner, holistically-judged ones —
    # capping at 3x FLUTE's count keeps a strong signal without letting
    # VUA20's token-triggered labels dominate the class (train split only;
    # test keeps its natural distribution for an honest eval).
    flute_metaphor_count = sum(1 for r in flute_train if r["label"] == METAPHOR)
    vua20_metaphor = [r for r in vua20_train if r["label"] == METAPHOR]
    vua20_literal  = [r for r in vua20_train if r["label"] == LITERAL]
    cap = flute_metaphor_count * 3
    if len(vua20_metaphor) > cap:
        vua20_metaphor = random.Random(42).sample(vua20_metaphor, cap)
        vua20_train = vua20_literal + vua20_metaphor
        print(f"[data] VUA20 train metaphor capped to {cap:,} (3x FLUTE's {flute_metaphor_count:,})")

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
