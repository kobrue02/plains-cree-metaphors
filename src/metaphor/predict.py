"""
Run inference on arbitrary text (English or Cree — language-agnostic).

predict() takes raw sentences and returns token-level metaphor labels
with confidence scores.  It works identically for English evaluation
and Cree zero-shot transfer.
"""

from __future__ import annotations
from dataclasses import dataclass

import torch
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer

from src.metaphor.config import ExperimentConfig
from src.metaphor.model import XLMRobertaLayerSelectForTokenClassification
from src.device import get_device


@dataclass
class TokenPrediction:
    word: str
    label: int           # 0 = literal, 1 = metaphor
    confidence: float    # probability of the predicted class


def load_model(
    config_or_path: ExperimentConfig | str,
) -> tuple[AutoModelForTokenClassification, AutoTokenizer]:
    """Load a fine-tuned checkpoint.

    Pass an ExperimentConfig (uses config.checkpoint_dir) or a raw path/HF id.
    """
    path = (
        config_or_path.checkpoint_dir
        if isinstance(config_or_path, ExperimentConfig)
        else config_or_path
    )
    tokenizer   = AutoTokenizer.from_pretrained(path)
    saved_cfg   = AutoConfig.from_pretrained(path)
    if getattr(saved_cfg, "hidden_layer", None) is not None:
        model = XLMRobertaLayerSelectForTokenClassification.from_pretrained(path)
    else:
        model = AutoModelForTokenClassification.from_pretrained(path)
    model.eval()
    return model, tokenizer


def predict(
    sentences: list[str],
    model: AutoModelForTokenClassification,
    tokenizer: AutoTokenizer,
    config: ExperimentConfig | None = None,
    batch_size: int = 32,
    max_length: int = 128,
    device: str | None = None,
) -> list[list[TokenPrediction]]:
    """Return per-token metaphor predictions for each input sentence.

    Each sentence is whitespace-tokenised first; the returned TokenPredictions
    align 1-to-1 with those whitespace tokens.

    Works for any language — pass Cree sentences for zero-shot transfer.
    """
    if device is None:
        device = get_device()

    if config is not None:
        batch_size = config.infer_batch_size
        max_length = config.max_length

    model = model.to(device)
    results: list[list[TokenPrediction]] = []

    for i in range(0, len(sentences), batch_size):
        batch_sents = sentences[i : i + batch_size]
        # Pre-split so we can recover word boundaries later
        batch_words = [s.split() for s in batch_sents]

        enc = tokenizer(
            batch_words,
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            logits = model(**enc).logits          # (B, L, 2)
        probs = torch.softmax(logits, dim=-1)     # (B, L, 2)

        for b_idx, words in enumerate(batch_words):
            word_ids     = enc.word_ids(batch_index=b_idx)
            seen_words: set[int] = set()
            sent_preds:  list[TokenPrediction] = []

            for tok_idx, wid in enumerate(word_ids):
                if wid is None or wid in seen_words:
                    continue
                seen_words.add(wid)
                if wid >= len(words):
                    continue
                p       = probs[b_idx, tok_idx]
                label   = int(p.argmax().item())
                conf    = float(p[label].item())
                sent_preds.append(TokenPrediction(
                    word=words[wid],
                    label=label,
                    confidence=conf,
                ))

            # Pad truncated sentences so len(sent_preds) == len(words) always.
            # Without this, truncated sentences shift all subsequent slices.
            for wid in range(len(sent_preds), len(words)):
                sent_preds.append(TokenPrediction(word=words[wid], label=0, confidence=0.5))

            results.append(sent_preds)

    return results


def predict_df(
    sentences: list[str],
    model: AutoModelForTokenClassification,
    tokenizer: AutoTokenizer,
    **kwargs,
):
    """Same as predict() but returns a flat pandas DataFrame.

    Columns: sentence_idx, word, label, confidence
    """
    import pandas as pd
    predictions = predict(sentences, model, tokenizer, **kwargs)
    rows = []
    for sent_idx, sent_preds in enumerate(predictions):
        for tp in sent_preds:
            rows.append({
                "sentence_idx": sent_idx,
                "word":         tp.word,
                "label":        tp.label,
                "confidence":   round(tp.confidence, 4),
            })
    return pd.DataFrame(rows)
