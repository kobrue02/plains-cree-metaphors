"""
Evaluate CLKD models against DeepSeek-annotated validation set.

Ground truth: data/figurative/bloomfield_annotated.csv
  - ~1,225 sentences from footnoted Bloomfield paragraphs
  - Labels assigned by DeepSeek R1 using Bloomfield's own footnotes as primary signal
  - `footnote_applies=True` marks the 35 sentences where Bloomfield directly
    flagged figurative language — the highest-confidence subset

For each model: predict on Cree text, compare to DeepSeek labels.
Reports precision / recall / F1 per class and macro average.
Two result tables:
  (a) full validation set (1,225 sentences)
  (b) gold subset only    (footnote_applies=True, ~219 sentences)

Output:
  data/figurative/eval_validation_full.csv
  data/figurative/eval_validation_gold.csv
"""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import pandas as pd
from sklearn.metrics import classification_report, f1_score

from src.figurative.predict import load_model, predict_sentences
from src.figurative.data import LABEL_NAMES

ANNOT_FILE   = "data/figurative/bloomfield_annotated.parquet"
OUTPUT_FULL  = "data/figurative/eval_validation_full.csv"
OUTPUT_GOLD  = "data/figurative/eval_validation_gold.csv"

MODELS = [
    ("XLM-R base",            "KonradBRG/xlm-r-plains-cree-en-tlm-figurative"),
    ("XLM-MLM CLKD f12",      "KonradBRG/xlm-mlm-100-1280-plains-cree-en-clkd-frozen12"),
    ("XLM-MLM CLKD full",     "KonradBRG/xlm-mlm-100-1280-plains-cree-en-clkd-full"),
    ("Glot500 CLKD direct",   "KonradBRG/glot500-base-plains-cree-en-clkd-direct"),
    ("Glot500 CLKD + TLM",    "KonradBRG/glot500-base-plains-cree-en-clkd-tlm"),
    ("XLM-V CLKD direct",     "KonradBRG/xlm-v-base-plains-cree-en-clkd-direct"),
    ("XLM-MLM calibrated",    "KonradBRG/xlm-mlm-plains-cree-en-calibrated"),
    ("XLM-V calibrated",      "KonradBRG/xlm-v-plains-cree-en-calibrated"),
]


def metrics_for(y_true: list[str], y_pred: list[str]) -> dict:
    """Per-class and macro P/R/F1 as a flat dict."""
    report = classification_report(
        y_true, y_pred, labels=LABEL_NAMES, output_dict=True, zero_division=0
    )
    row = {}
    for label in LABEL_NAMES:
        r = report.get(label, {})
        row[f"p_{label}"]  = round(r.get("precision", 0.0), 4)
        row[f"r_{label}"]  = round(r.get("recall",    0.0), 4)
        row[f"f1_{label}"] = round(r.get("f1-score",  0.0), 4)
    macro = report.get("macro avg", {})
    row["macro_p"]  = round(macro.get("precision", 0.0), 4)
    row["macro_r"]  = round(macro.get("recall",    0.0), 4)
    row["macro_f1"] = round(macro.get("f1-score",  0.0), 4)
    row["accuracy"] = round(report.get("accuracy", 0.0), 4)
    return row


def evaluate(df: pd.DataFrame, subset_name: str) -> list[dict]:
    cree_texts = df["text_cree"].tolist()
    y_true     = df["label"].tolist()

    rows = []
    for name, ckpt in MODELS:
        print(f"\n{'='*60}\n  {name}  [{subset_name}]")
        try:
            model, tok = load_model(ckpt)
            preds  = predict_sentences(cree_texts, model, tok)
            y_pred = [p["label"] for p in preds]
            del model
            torch.cuda.empty_cache()

            m = metrics_for(y_true, y_pred)
            print(f"  macro F1={m['macro_f1']:.3f}  "
                  + "  ".join(f"{l}={m[f'f1_{l}']:.2f}" for l in LABEL_NAMES))
            rows.append({"model": name, "checkpoint": ckpt, **m})
        except Exception as exc:
            print(f"  SKIPPED — {exc}")
            rows.append({"model": name, "checkpoint": ckpt, "error": str(exc)})

    return rows


def main() -> None:
    annot = pd.read_parquet(ANNOT_FILE)
    annot = annot.dropna(subset=["text_cree", "label"])
    # normalise label column (in case of stray whitespace)
    annot["label"] = annot["label"].str.strip().str.lower().map(
        lambda x: x if x in LABEL_NAMES else "literal"
    )

    gold = annot[annot["footnote_applies"] == True]

    print(f"Full validation set : {len(annot)} sentences")
    print(f"  label dist: {annot['label'].value_counts().to_dict()}")
    print(f"Gold subset (footnote_applies=True): {len(gold)} sentences")
    print(f"  label dist: {gold['label'].value_counts().to_dict()}")

    os.makedirs("data/figurative", exist_ok=True)

    print("\n\n── Full validation set ──────────────────────────────────────")
    rows_full = evaluate(annot, "full")
    pd.DataFrame(rows_full).to_csv(OUTPUT_FULL, index=False, encoding="utf-8-sig")
    print(f"\nSaved → {OUTPUT_FULL}")

    print("\n\n── Gold subset (footnote_applies=True) ──────────────────────")
    rows_gold = evaluate(gold, "gold")
    pd.DataFrame(rows_gold).to_csv(OUTPUT_GOLD, index=False, encoding="utf-8-sig")
    print(f"\nSaved → {OUTPUT_GOLD}")


if __name__ == "__main__":
    main()
