"""
Metrics for metaphor token classification.

compute_metrics() is passed directly to HuggingFace Trainer.
evaluate() is for standalone evaluation after training.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score


def compute_metrics(eval_pred) -> dict[str, float]:
    """HuggingFace Trainer-compatible metrics function.

    Returns metaphor-class P / R / F1 and macro F1.
    Tokens with label -100 (subword continuations, padding) are excluded.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)          # (B, L)

    flat_preds  = []
    flat_labels = []
    for pred_seq, label_seq in zip(preds, labels):
        for p, l in zip(pred_seq, label_seq):
            if l != -100:
                flat_preds.append(p)
                flat_labels.append(l)

    flat_preds  = np.array(flat_preds)
    flat_labels = np.array(flat_labels)

    return {
        "metaphor_precision": precision_score(flat_labels, flat_preds, pos_label=1, zero_division=0),
        "metaphor_recall":    recall_score(   flat_labels, flat_preds, pos_label=1, zero_division=0),
        "metaphor_f1":        f1_score(       flat_labels, flat_preds, pos_label=1, zero_division=0),
        "macro_f1":           f1_score(       flat_labels, flat_preds, average="macro", zero_division=0),
    }


def evaluate(
    predictions: list[list[int]],
    references:  list[list[int]],
    print_report: bool = True,
) -> dict[str, float]:
    """Standalone evaluation on flat or nested prediction/reference lists.

    Both inputs can be:
      - flat lists of int labels
      - nested lists of per-sentence labels (will be flattened, -100 excluded)
    """
    flat_preds, flat_refs = [], []

    def _is_nested(lst):
        return lst and isinstance(lst[0], list)

    if _is_nested(predictions):
        for pred_seq, ref_seq in zip(predictions, references):
            for p, r in zip(pred_seq, ref_seq):
                if r != -100:
                    flat_preds.append(p)
                    flat_refs.append(r)
    else:
        for p, r in zip(predictions, references):
            if r != -100:
                flat_preds.append(p)
                flat_refs.append(r)

    metrics = {
        "metaphor_precision": precision_score(flat_refs, flat_preds, pos_label=1, zero_division=0),
        "metaphor_recall":    recall_score(   flat_refs, flat_preds, pos_label=1, zero_division=0),
        "metaphor_f1":        f1_score(       flat_refs, flat_preds, pos_label=1, zero_division=0),
        "macro_f1":           f1_score(       flat_refs, flat_preds, average="macro", zero_division=0),
    }

    if print_report:
        print(classification_report(
            flat_refs, flat_preds,
            target_names=["literal", "metaphor"],
            zero_division=0,
        ))

    return metrics
