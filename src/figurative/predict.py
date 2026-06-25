"""
Sentence-level figurative language prediction (4 classes: literal/idiom/metaphor/simile).
"""

from __future__ import annotations

import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.figurative.data import LABEL_NAMES


def load_model(
    checkpoint: str,
) -> tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    use_fast = "deberta-v3" not in checkpoint.lower()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=use_fast)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, torch_dtype=torch.float32,
    )
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
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

    for i in range(0, len(texts), batch_size):
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


def predict_idioms(
    idioms_path: str,
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    **kwargs,
) -> pd.DataFrame:
    """Run the model on both sides of an idioms file (cree ||| english).

    Returns a DataFrame comparing predictions for the Cree text vs. the
    English translation — useful for measuring cross-lingual idiom transfer.
    """
    rows = []
    with open(idioms_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "|||" not in line:
                continue
            cree, _, english = line.partition("|||")
            rows.append({"cree": cree.strip(), "english": english.strip()})

    if not rows:
        raise ValueError(f"No valid cree ||| english lines found in {idioms_path}")

    cree_texts    = [r["cree"]    for r in rows]
    english_texts = [r["english"] for r in rows if r["english"]]

    cree_preds    = predict_sentences(cree_texts,    model, tokenizer, **kwargs)
    english_preds = predict_sentences(english_texts, model, tokenizer, **kwargs)

    records = []
    en_idx = 0
    for i, row in enumerate(rows):
        cp = cree_preds[i]
        ep = english_preds[en_idx] if row["english"] else {}
        en_idx += 1 if row["english"] else 0
        records.append({
            "cree":         row["cree"],
            "english":      row["english"],
            "cree_label":   cp["label"],
            "cree_conf":    cp["confidence"],
            "cree_p_idiom": cp.get("prob_idiom", ""),
            "en_label":     ep.get("label", ""),
            "en_conf":      ep.get("confidence", ""),
            "en_p_idiom":   ep.get("prob_idiom", ""),
        })

    return pd.DataFrame(records)


def eval_idioms(
    idioms_path: str,
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    **kwargs,
) -> dict:
    """Evaluate on the Cree idiom golden test set and print a summary.

    All entries in idioms.txt are idioms (class 1), so the ground truth is
    fixed.  Reports two metrics for both Cree and English:
      - idiom accuracy   : % predicted as 'idiom'
      - figurative rate  : % predicted as any non-literal class
    """
    df = predict_idioms(idioms_path, model, tokenizer, **kwargs)

    print(f"\n{'='*60}")
    print(f"Idiom golden-set evaluation  ({len(df)} examples, all ground-truth: idiom)")
    print(f"{'='*60}")
    print(df[["cree", "english", "cree_label", "cree_p_idiom",
              "en_label", "en_p_idiom"]].to_string(index=False))
    print()

    def _metrics(label_col: str, p_idiom_col: str, side: str):
        valid = df[label_col] != ""
        labels = df.loc[valid, label_col]
        idiom_acc = (labels == "idiom").mean()
        fig_rate  = (labels != "literal").mean()
        avg_p_idiom = df.loc[valid, p_idiom_col].astype(float).mean()
        print(f"{side:8s}  idiom_accuracy={idiom_acc:.1%}  "
              f"figurative_rate={fig_rate:.1%}  "
              f"mean_p_idiom={avg_p_idiom:.3f}")
        return {"idiom_accuracy": idiom_acc, "figurative_rate": fig_rate,
                "mean_p_idiom": avg_p_idiom}

    cree_metrics = _metrics("cree_label", "cree_p_idiom", "Cree")
    en_metrics   = _metrics("en_label",   "en_p_idiom",   "English")
    print(f"{'='*60}\n")

    return {"cree": cree_metrics, "english": en_metrics, "detail": df}
