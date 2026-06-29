"""
(1) English-Cree Consistency Evaluation

For each model, measures how often its predictions on Cree sentences agree
with a DeBERTa-v3 teacher's predictions on the paired English translations.

For CLKD models this directly measures transfer quality.
For direct fine-tuned baselines it measures cross-lingual label consistency.

Output: data/figurative/eval_consistency.csv
"""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F

from src.figurative.predict import load_model, predict_sentences
from src.figurative.data import LABEL_NAMES

CORPUS_FILE = "data/bloomfield_texts_sentences.parquet"
TEACHER_ID  = "KonradBRG/deberta-v3-base-figurative"
OUTPUT_FILE = "data/figurative/eval_consistency.csv"

MODELS = [
    ("XLM-R base (figurative)",          "KonradBRG/xlm-r-plains-cree-en-tlm-figurative"),
    ("XLM-R large (figurative)",         "KonradBRG/xlm-r-large-plains-cree-en-tlm-figurative"),
    ("DeBERTa-v3 (self, English only)",  TEACHER_ID),
    ("XLM-MLM CLKD frozen-12",           "KonradBRG/xlm-mlm-100-1280-plains-cree-en-clkd-frozen12"),
    ("XLM-MLM CLKD full",                "KonradBRG/xlm-mlm-100-1280-plains-cree-en-clkd-full"),
    ("Glot500 CLKD direct",              "KonradBRG/glot500-base-plains-cree-en-clkd-direct"),
    ("Glot500 CLKD + TLM",               "KonradBRG/glot500-base-plains-cree-en-clkd-tlm"),
    ("XLM-V CLKD direct",                "KonradBRG/xlm-v-base-plains-cree-en-clkd-direct"),
    ("XLM-V CLKD + TLM",                 "KonradBRG/xlm-v-base-plains-cree-en-clkd-tlm"),
]


def get_probs(texts: list[str], model, tokenizer) -> np.ndarray:
    preds = predict_sentences(texts, model, tokenizer)
    return np.array([[p[f"prob_{l}"] for l in LABEL_NAMES] for p in preds])


df = pd.read_parquet(CORPUS_FILE).dropna(subset=["text_cree", "text_en"])
cree_texts = df["text_cree"].tolist()
en_texts   = df["text_en"].tolist()
print(f"Corpus: {len(df):,} pairs")

# Cache teacher predictions on English once
print(f"\nLoading teacher: {TEACHER_ID}")
t_model, t_tok = load_model(TEACHER_ID)
teacher_probs  = get_probs(en_texts, t_model, t_tok)
teacher_labels = teacher_probs.argmax(axis=1)
del t_model
torch.cuda.empty_cache()

rows = []
for name, ckpt in MODELS:
    print(f"\n{'='*60}\n  {name}")
    try:
        model, tok = load_model(ckpt)

        if ckpt == TEACHER_ID:
            # Self-consistency: English vs English (upper bound sanity check)
            student_probs  = teacher_probs.copy()
        else:
            student_probs  = get_probs(cree_texts, model, tok)

        student_labels = student_probs.argmax(axis=1)

        # Label agreement
        agreement = (student_labels == teacher_labels).mean()

        # Per-class agreement
        per_class = {}
        for i, label in enumerate(LABEL_NAMES):
            mask = teacher_labels == i
            if mask.sum() > 0:
                per_class[f"agree_{label}"] = (student_labels[mask] == i).mean()
            else:
                per_class[f"agree_{label}"] = float("nan")

        # Mean KL divergence: KL(teacher || student)
        t_log = np.log(teacher_probs + 1e-8)
        s_log = np.log(student_probs + 1e-8)
        kl = (teacher_probs * (t_log - s_log)).sum(axis=1).mean()

        row = {"model": name, "checkpoint": ckpt,
               "label_agreement": round(agreement, 4),
               "mean_kl_div": round(float(kl), 4),
               **{k: round(v, 4) for k, v in per_class.items()}}
        rows.append(row)
        print(f"  agreement={agreement:.1%}  kl={kl:.4f}  "
              + "  ".join(f"{l}={per_class[f'agree_{l}']:.1%}" for l in LABEL_NAMES))

        del model
        torch.cuda.empty_cache()
    except Exception as exc:
        print(f"  SKIPPED — {exc}")
        rows.append({"model": name, "checkpoint": ckpt, "error": str(exc)})

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
out = pd.DataFrame(rows)
out.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
print(f"\nSaved to {OUTPUT_FILE}")
print(out[["model", "label_agreement", "mean_kl_div"]].to_string(index=False))
