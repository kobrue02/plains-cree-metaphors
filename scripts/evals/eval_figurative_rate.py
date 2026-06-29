"""
(2) Figurative Rate on Bloomfield Corpus

Runs each model on the full Bloomfield Cree sentence corpus and reports the
predicted label distribution. A model predicting ~99% literal is degenerate;
a reasonable model should find some figurative language in a literary text.

Output: data/figurative/eval_figurative_rate.csv
"""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import pandas as pd

from src.figurative.predict import load_model, predict_sentences
from src.figurative.data import LABEL_NAMES

CORPUS_FILE = "data/bloomfield_texts_sentences.parquet"
OUTPUT_FILE = "data/figurative/eval_figurative_rate.csv"

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

df = pd.read_parquet(CORPUS_FILE).dropna(subset=["text_cree"])
cree_texts = df["text_cree"].tolist()
print(f"Corpus: {len(cree_texts):,} Cree sentences")

rows = []
for name, ckpt in MODELS:
    print(f"\n{'='*60}\n  {name}")
    try:
        model, tok = load_model(ckpt)
        preds  = predict_sentences(cree_texts, model, tok)
        labels = [p["label"] for p in preds]
        total  = len(labels)

        counts = {l: labels.count(l) for l in LABEL_NAMES}
        rates  = {f"rate_{l}": round(counts[l] / total, 4) for l in LABEL_NAMES}
        fig_rate = round(1.0 - counts["literal"] / total, 4)

        row = {"model": name, "checkpoint": ckpt, "n_sentences": total,
               "figurative_rate": fig_rate, **rates,
               **{f"n_{l}": counts[l] for l in LABEL_NAMES}}
        rows.append(row)

        print(f"  figurative_rate={fig_rate:.1%}  "
              + "  ".join(f"{l}={counts[l]} ({rates[f'rate_{l}']:.1%})" for l in LABEL_NAMES))

        del model
        torch.cuda.empty_cache()
    except Exception as exc:
        print(f"  SKIPPED — {exc}")
        rows.append({"model": name, "checkpoint": ckpt, "error": str(exc)})

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
out = pd.DataFrame(rows)
out.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
print(f"\nSaved to {OUTPUT_FILE}")
display_cols = ["model", "figurative_rate"] + [f"rate_{l}" for l in LABEL_NAMES]
print(out[[c for c in display_cols if c in out.columns]].to_string(index=False))
