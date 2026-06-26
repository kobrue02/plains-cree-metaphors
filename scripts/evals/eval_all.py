"""
Evaluate all trained checkpoints on the Cree idiom golden set.

Loads each model in turn, frees GPU memory between runs, and writes a
single comparison table to data/figurative/results_table.csv.
"""

from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import pandas as pd

from src.figurative.predict import load_model, eval_idioms

CHECKPOINTS = [
    # ── Baselines ────────────────────────────────────────────────────────────
    ("XLM-R base (figurative)",           "KonradBRG/xlm-r-plains-cree-en-tlm-figurative"),
    ("XLM-R large (figurative)",          "KonradBRG/xlm-r-large-plains-cree-en-tlm-figurative"),
    ("XLM-MLM (figurative)",              "KonradBRG/xlm-mlm-100-1280-plains-cree-en-figurative"),
    ("DeBERTa-v3 (English teacher)",      "KonradBRG/deberta-v3-base-figurative"),
    # ── CLKD : XLM-MLM student ──────────────────────────────────────────────
    ("XLM-MLM CLKD frozen-12",            "KonradBRG/xlm-mlm-100-1280-plains-cree-en-clkd-frozen12"),
    ("XLM-MLM CLKD full",                 "KonradBRG/xlm-mlm-100-1280-plains-cree-en-clkd-full"),
    # ── CLKD : Glot500 student ───────────────────────────────────────────────
    ("Glot500 CLKD direct",               "KonradBRG/glot500-base-plains-cree-en-clkd-direct"),
    ("Glot500 CLKD + TLM",                "KonradBRG/glot500-base-plains-cree-en-clkd-tlm"),
    # ── CLKD : XLM-V student ─────────────────────────────────────────────────
    ("XLM-V CLKD direct",                 "KonradBRG/xlm-v-base-plains-cree-en-clkd-direct"),
    ("XLM-V CLKD + TLM",                  "KonradBRG/xlm-v-base-plains-cree-en-clkd-tlm"),
]

IDIOMS_FILE = "data/idioms.txt"
OUTPUT_FILE = "data/figurative/results_table.csv"


def _row(name: str, ckpt: str, result: dict) -> dict:
    c = result["cree"]
    e = result["english"]
    return {
        "model":            name,
        "checkpoint":       ckpt,
        "cree_idiom_acc":   round(c["idiom_accuracy"],  4),
        "cree_fig_rate":    round(c["figurative_rate"],  4),
        "cree_p_idiom":     round(c["mean_p_idiom"],     4),
        "en_idiom_acc":     round(e["idiom_accuracy"],   4),
        "en_fig_rate":      round(e["figurative_rate"],  4),
        "en_p_idiom":       round(e["mean_p_idiom"],     4),
    }


rows = []
for name, ckpt in CHECKPOINTS:
    print(f"\n{'='*64}")
    print(f"  {name}")
    print(f"  {ckpt}")
    print(f"{'='*64}")
    try:
        model, tokenizer = load_model(ckpt)
        result = eval_idioms(IDIOMS_FILE, model, tokenizer)
        rows.append(_row(name, ckpt, result))
    except Exception as exc:
        print(f"  SKIPPED — {exc}")
        rows.append({"model": name, "checkpoint": ckpt, "error": str(exc)})
    finally:
        try:
            del model
        except NameError:
            pass
        torch.cuda.empty_cache()

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
df = pd.DataFrame(rows)
df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print(f"\n{'='*64}")
print("FULL RESULTS TABLE")
print(f"{'='*64}")
metric_cols = ["model", "cree_idiom_acc", "cree_fig_rate", "cree_p_idiom",
               "en_idiom_acc",   "en_fig_rate",   "en_p_idiom"]
display = df[[c for c in metric_cols if c in df.columns]]
print(display.to_string(index=False))
print(f"\nSaved full table to {OUTPUT_FILE}")
