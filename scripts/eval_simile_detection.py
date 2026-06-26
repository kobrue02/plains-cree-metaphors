"""
(3) Simile Detection via Surface Markers

Plains Cree similes are marked by the particle 'tâpiskôc' (like / as if).
Sentences containing this marker are near-certain similes, giving us a
small silver-standard test set with no manual annotation.

Metric: % of tâpiskôc sentences predicted as 'simile' by each model.

Output: data/figurative/eval_simile_detection.csv
"""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import torch
import pandas as pd

from src.figurative.predict import load_model, predict_sentences
from src.figurative.data import LABEL_NAMES

CORPUS_FILE   = "data/bloomfield_texts_sentences.csv"
OUTPUT_FILE   = "data/figurative/eval_simile_detection.csv"
DETAIL_FILE   = "data/figurative/eval_simile_detection_detail.csv"

# tâpiskôc and common orthographic variants
SIMILE_RE = re.compile(r"tâpiskôc|tâpiskôt|tapiskoc|tapiskot", re.IGNORECASE)

MODELS = [
    ("XLM-R base (figurative)",          "KonradBRG/xlm-r-plains-cree-en-tlm-figurative"),
    ("XLM-R large (figurative)",         "KonradBRG/xlm-r-large-plains-cree-en-tlm-figurative"),
    ("DeBERTa-v3 (English teacher)",     "KonradBRG/deberta-v3-base-figurative"),
    ("XLM-MLM CLKD frozen-12",           "KonradBRG/xlm-mlm-100-1280-plains-cree-en-clkd-frozen12"),
    ("XLM-MLM CLKD full",                "KonradBRG/xlm-mlm-100-1280-plains-cree-en-clkd-full"),
    ("Glot500 CLKD direct",              "KonradBRG/glot500-base-plains-cree-en-clkd-direct"),
    ("Glot500 CLKD + TLM",               "KonradBRG/glot500-base-plains-cree-en-clkd-tlm"),
    ("XLM-V CLKD direct",                "KonradBRG/xlm-v-base-plains-cree-en-clkd-direct"),
    ("XLM-V CLKD + TLM",                 "KonradBRG/xlm-v-base-plains-cree-en-clkd-tlm"),
]

df = pd.read_csv(CORPUS_FILE, encoding="utf-8-sig").dropna(subset=["text_cree"])
simile_df = df[df["text_cree"].str.contains(SIMILE_RE, na=False)].reset_index(drop=True)
other_df  = df[~df["text_cree"].str.contains(SIMILE_RE, na=False)].reset_index(drop=True)

print(f"Corpus     : {len(df):,} sentences")
print(f"tâpiskôc   : {len(simile_df):,} simile candidates")
print(f"Non-simile : {len(other_df):,} sentences")

simile_texts = simile_df["text_cree"].tolist()
other_texts  = other_df["text_cree"].tolist()

summary_rows = []
detail_rows  = []

for name, ckpt in MODELS:
    print(f"\n{'='*60}\n  {name}")
    try:
        model, tok = load_model(ckpt)

        simile_preds = predict_sentences(simile_texts, model, tok)
        other_preds  = predict_sentences(other_texts,  model, tok)

        simile_labels = [p["label"] for p in simile_preds]
        other_labels  = [p["label"] for p in other_preds]

        # True positive rate: tâpiskôc sentences predicted as simile
        tp_rate = sum(l == "simile" for l in simile_labels) / len(simile_labels)
        # False positive rate: non-simile sentences predicted as simile
        fp_rate = sum(l == "simile" for l in other_labels)  / len(other_labels)
        # Figurative rate on simile sentences (any non-literal)
        fig_rate_simile = sum(l != "literal" for l in simile_labels) / len(simile_labels)

        print(f"  simile_tp_rate={tp_rate:.1%}  "
              f"simile_fp_rate={fp_rate:.1%}  "
              f"fig_rate_on_similes={fig_rate_simile:.1%}")

        summary_rows.append({
            "model":              name,
            "checkpoint":         ckpt,
            "n_simile":           len(simile_labels),
            "simile_tp_rate":     round(tp_rate,        4),
            "simile_fp_rate":     round(fp_rate,        4),
            "fig_rate_on_similes":round(fig_rate_simile,4),
        })

        for i, (row, pred) in enumerate(zip(simile_df.itertuples(), simile_preds)):
            detail_rows.append({
                "model":      name,
                "text_cree":  row.text_cree,
                "text_en":    getattr(row, "text_en", ""),
                "pred_label": pred["label"],
                "confidence": pred["confidence"],
                "prob_simile":pred.get("prob_simile", ""),
            })

        del model
        torch.cuda.empty_cache()
    except Exception as exc:
        print(f"  SKIPPED — {exc}")
        summary_rows.append({"model": name, "checkpoint": ckpt, "error": str(exc)})

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
pd.DataFrame(summary_rows).to_csv(OUTPUT_FILE,  index=False, encoding="utf-8-sig")
pd.DataFrame(detail_rows).to_csv(DETAIL_FILE,   index=False, encoding="utf-8-sig")

print(f"\nSaved summary → {OUTPUT_FILE}")
print(f"Saved details → {DETAIL_FILE}")
summary = pd.DataFrame(summary_rows)
print(summary[["model", "simile_tp_rate", "simile_fp_rate", "fig_rate_on_similes"]].to_string(index=False))
