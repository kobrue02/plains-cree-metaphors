import numpy as np
from sklearn.metrics import classification_report

from src.figurative.data import LABEL_NAMES


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    report = classification_report(
        labels,
        preds,
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )
    return {
        "macro_f1":    report["macro avg"]["f1-score"],
        **{f"{name}_f1": report[name]["f1-score"] for name in LABEL_NAMES},
    }
