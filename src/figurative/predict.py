"""
Sentence-level figurative language prediction (4 classes: literal/idiom/metaphor/simile).
"""

from __future__ import annotations

import json
import os

import torch
import pandas as pd
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.device import get_device
from src.figurative.data import LABEL_NAMES, resolve_use_fast


def _model_type(checkpoint: str) -> str | None:
    """Read config.json's `model_type` without going through AutoConfig, since
    custom architectures like HierarchicalFigurativeModel aren't registered
    with the Auto* mapping."""
    config_path = os.path.join(checkpoint, "config.json") if os.path.isdir(checkpoint) else None
    if config_path is None or not os.path.exists(config_path):
        config_path = hf_hub_download(checkpoint, "config.json")
    with open(config_path) as f:
        return json.load(f).get("model_type")


def load_model(
    checkpoint: str,
):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=resolve_use_fast(checkpoint))
    if _model_type(checkpoint) == "hierarchical_figurative":
        from src.figurative.hierarchical import HierarchicalFigurativeModel
        model = HierarchicalFigurativeModel.from_pretrained(checkpoint, torch_dtype=torch.float32)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint, torch_dtype=torch.float32,
        )
    model.eval()
    model.to(get_device())
    return model, tokenizer


def predict_sentences(
    texts: list[str],
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    batch_size: int = 32,
    max_length: int = 128,
) -> list[dict]:
    """Run inference on a list of sentences. Returns one dict per sentence."""
    device = next(model.parameters()).device
    results = []

    batch_starts = range(0, len(texts), batch_size)
    for i in tqdm(batch_starts, desc="predicting", unit="batch"):
        batch_texts = texts[i : i + batch_size]
        enc = tokenizer(
            batch_texts,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu()

        for j, text in enumerate(batch_texts):
            p = probs[j]
            pred = int(p.argmax().item())
            row = {
                "text":       text,
                "label":      LABEL_NAMES[pred],
                "confidence": round(p[pred].item(), 4),
            }
            for k, name in enumerate(LABEL_NAMES):
                row[f"prob_{name}"] = round(p[k].item(), 4)
            results.append(row)

    return results


