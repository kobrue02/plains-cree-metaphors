"""
Load and preprocess the VUA20 metaphor dataset into a HuggingFace-compatible
token-classification format.

VUA20 schema (one row per token):
    index     — sentence ID  (e.g. "a1e-fragment01 1")
    sentence  — full sentence text
    w_index   — 0-based word position in sentence.split()
    label     — 0 = literal, 1 = metaphor
    POS       — coarse POS tag  (VERB / NOUN / ADJ / ADV / ...)
    FGPOS     — fine-grained POS
"""

from __future__ import annotations
from collections import defaultdict

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast

from src.metaphor.config import ExperimentConfig


def load_vua20_sentences(split: str = "train") -> list[dict]:
    """Return a list of sentence dicts, each containing aligned word/label lists.

    {
        'index':    sentence ID string,
        'sentence': raw sentence text,
        'words':    ['word0', 'word1', ...],   # sentence.split()
        'labels':   [0, -100, 1, ...],         # -100 for unlabelled positions
        'pos':      ['VERB', None, 'NOUN', ...],
    }
    """
    rows = load_dataset("CreativeLang/vua20_metaphor")[split]

    # Group by sentence index
    buckets: dict[str, dict] = defaultdict(lambda: {"tokens": {}})
    for row in rows:
        idx = row["index"]
        buckets[idx]["index"]    = idx
        buckets[idx]["sentence"] = row["sentence"]
        buckets[idx]["tokens"][row["w_index"]] = {
            "label": row["label"],
            "pos":   row["POS"],
        }

    sentences = []
    for idx, bucket in buckets.items():
        words = bucket["sentence"].split()
        n     = len(words)
        labels = [-100] * n
        pos    = [None]  * n
        for wi, tok in bucket["tokens"].items():
            if wi < n:
                labels[wi] = tok["label"]
                pos[wi]    = tok["pos"]
        sentences.append({
            "index":    idx,
            "sentence": bucket["sentence"],
            "words":    words,
            "labels":   labels,
            "pos":      pos,
        })

    return sentences


class MetaphorDataset(Dataset):
    """Tokenised VUA20 sentences ready for AutoModelForTokenClassification.

    Subword alignment: only the *first* subword of each word carries its label;
    all subsequent pieces are masked with -100 and ignored in the loss.

    POS filtering (optional): tokens whose POS is not in config.active_pos_filter
    are also masked with -100 so the model only learns from the relevant tags.
    """

    def __init__(
        self,
        sentences: list[dict],
        tokenizer: PreTrainedTokenizerFast,
        config: ExperimentConfig,
    ):
        self.encodings = []
        pos_filter = config.active_pos_filter

        for sent in sentences:
            words  = sent["words"]
            labels = sent["labels"]   # one per word, -100 if unlabelled
            pos    = sent["pos"]

            enc = tokenizer(
                words,
                is_split_into_words=True,
                truncation=True,
                max_length=config.max_length,
                padding="max_length",
                return_tensors="pt",
            )

            word_ids     = enc.word_ids(batch_index=0)
            aligned_lbls = []
            prev_word_id = None

            for wid in word_ids:
                if wid is None:
                    # [CLS] / [SEP] / padding
                    aligned_lbls.append(-100)
                elif wid != prev_word_id:
                    # First subword of this word
                    lbl = labels[wid] if wid < len(labels) else -100
                    # Apply POS filter: mask non-target POS
                    if lbl != -100 and pos_filter and pos[wid] not in pos_filter:
                        lbl = -100
                    aligned_lbls.append(lbl)
                else:
                    # Continuation subword — ignore in loss
                    aligned_lbls.append(-100)
                prev_word_id = wid

            self.encodings.append({
                "input_ids":      enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels":         torch.tensor(aligned_lbls, dtype=torch.long),
            })

    def __len__(self) -> int:
        return len(self.encodings)

    def __getitem__(self, idx: int) -> dict:
        return self.encodings[idx]


def build_datasets(
    tokenizer: PreTrainedTokenizerFast,
    config: ExperimentConfig,
) -> tuple[MetaphorDataset, MetaphorDataset]:
    """Return (train_dataset, test_dataset) for the given config."""
    train_sents = load_vua20_sentences("train")
    test_sents  = load_vua20_sentences("test")
    return (
        MetaphorDataset(train_sents, tokenizer, config),
        MetaphorDataset(test_sents,  tokenizer, config),
    )


def class_weights_from(dataset: MetaphorDataset) -> torch.Tensor:
    """Compute inverse-frequency class weights to handle the ~10% metaphor imbalance."""
    counts = torch.zeros(2)
    for item in dataset:
        for lbl in item["labels"]:
            if lbl.item() in (0, 1):
                counts[lbl.item()] += 1
    total = counts.sum()
    weights = total / (2 * counts)
    return weights
