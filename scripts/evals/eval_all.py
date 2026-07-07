"""
Single eval entry point for all figurative language evaluation tasks.

Tasks
-----
  python scripts/evals/eval_all.py --task checkpoints    # checkpoint comparison table (default all)
  python scripts/evals/eval_all.py --task figurative-rate
  python scripts/evals/eval_all.py --task simile
  python scripts/evals/eval_all.py --task consistency
  python scripts/evals/eval_all.py --task validation
  python scripts/evals/eval_all.py                       # runs all tasks

Output files
------------
  data/figurative/results_table.csv           — checkpoint comparison (checkpoints task)
  data/figurative/eval_figurative_rate.csv    — figurative-rate task
  data/figurative/eval_simile_detection.csv   — simile task (summary)
  data/figurative/eval_simile_detection_detail.csv — simile task (per-sentence)
  data/figurative/eval_consistency.csv        — consistency task
  data/figurative/eval_validation_full.csv    — validation task (full set)
  data/figurative/eval_validation_gold.csv    — validation task (gold subset)
"""

from __future__ import annotations
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import pandas as pd

from src.figurative.predict import load_model, predict_sentences, eval_idioms
from src.figurative.data import LABEL_NAMES


# ── Shared model lists ────────────────────────────────────────────────────────

_CHECKPOINTS = [
    # ── baselines ────────────────────────────────────────────────────────────
    ("XLM-R base (figurative)",           "KonradBRG/xlm-r-plains-cree-en-tlm-figurative"),
    ("XLM-R large (figurative)",          "KonradBRG/xlm-r-large-plains-cree-en-tlm-figurative"),
    ("XLM-MLM (figurative)",              "KonradBRG/xlm-mlm-100-1280-plains-cree-en-tlm-figurative"),
    ("DeBERTa-v3 (English teacher)",      "KonradBRG/deberta-v3-base-figurative"),
    # ── clkd : xlm-mlm student ──────────────────────────────────────────────
    ("XLM-MLM CLKD frozen-12",            "KonradBRG/xlm-mlm-100-1280-plains-cree-en-clkd-frozen12"),
    ("XLM-MLM CLKD full",                 "KonradBRG/xlm-mlm-100-1280-plains-cree-en-clkd-full"),
    # ── clkd : glot500 student ───────────────────────────────────────────────
    ("Glot500 CLKD direct",               "KonradBRG/glot500-base-plains-cree-en-clkd-direct"),
    ("Glot500 CLKD + TLM",                "KonradBRG/glot500-base-plains-cree-en-clkd-tlm"),
    # ── clkd : xlm-v student ─────────────────────────────────────────────────
    ("XLM-V CLKD direct",                 "KonradBRG/xlm-v-base-plains-cree-en-clkd-direct"),
    ("XLM-V CLKD + TLM",                  "KonradBRG/xlm-v-base-plains-cree-en-clkd-tlm"),
]

_RATE_MODELS = [
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

_CONSISTENCY_MODELS = [
    ("XLM-R base (figurative)",          "KonradBRG/xlm-r-plains-cree-en-tlm-figurative"),
    ("XLM-R large (figurative)",         "KonradBRG/xlm-r-large-plains-cree-en-tlm-figurative"),
    ("DeBERTa-v3 (self, English only)",  "KonradBRG/deberta-v3-base-figurative"),
    ("XLM-MLM CLKD frozen-12",           "KonradBRG/xlm-mlm-100-1280-plains-cree-en-clkd-frozen12"),
    ("XLM-MLM CLKD full",                "KonradBRG/xlm-mlm-100-1280-plains-cree-en-clkd-full"),
    ("Glot500 CLKD direct",              "KonradBRG/glot500-base-plains-cree-en-clkd-direct"),
    ("Glot500 CLKD + TLM",               "KonradBRG/glot500-base-plains-cree-en-clkd-tlm"),
    ("XLM-V CLKD direct",                "KonradBRG/xlm-v-base-plains-cree-en-clkd-direct"),
    ("XLM-V CLKD + TLM",                 "KonradBRG/xlm-v-base-plains-cree-en-clkd-tlm"),
]

_VALIDATION_MODELS = [
    ("XLM-R base",            "KonradBRG/xlm-r-plains-cree-en-tlm-figurative"),
    ("XLM-MLM CLKD f12",      "KonradBRG/xlm-mlm-100-1280-plains-cree-en-clkd-frozen12"),
    ("XLM-MLM CLKD full",     "KonradBRG/xlm-mlm-100-1280-plains-cree-en-clkd-full"),
    ("Glot500 CLKD direct",   "KonradBRG/glot500-base-plains-cree-en-clkd-direct"),
    ("Glot500 CLKD + TLM",    "KonradBRG/glot500-base-plains-cree-en-clkd-tlm"),
    ("XLM-V CLKD direct",     "KonradBRG/xlm-v-base-plains-cree-en-clkd-direct"),
    # ── CLKD, pre-calibration — base pipeline (pipeline.py), matched lineage with
    # the calibrated entries right below, for a clean does-calibration-help check ──
    ("XLM-MLM CLKD (pre-calibration)", "KonradBRG/xlm-mlm-plains-cree-en-clkd"),
    ("XLM-V CLKD (pre-calibration)",   "KonradBRG/xlm-v-plains-cree-en-clkd"),
    # ── calibrated — base pipeline (pipeline.py) ──────────────────────────────
    ("XLM-MLM calibrated",    "KonradBRG/xlm-mlm-plains-cree-en-calibrated"),
    ("Glot500 calibrated",    "KonradBRG/glot500-plains-cree-en-calibrated"),
    ("XLM-V calibrated",      "KonradBRG/xlm-v-plains-cree-en-calibrated"),
    # ── calibrated — ablation study (jobs/ablation.sh, xlm-mlm base) ──────────
    ("Ablation: full",            "KonradBRG/xlm-mlm-abl-full-plains-cree-en-calibrated"),
    ("Ablation: no TLM",          "KonradBRG/xlm-mlm-abl-no-tlm-plains-cree-en-calibrated"),
    ("Ablation: no CLKD",         "KonradBRG/xlm-mlm-abl-no-clkd-plains-cree-en-calibrated"),
    ("Ablation: neither",         "KonradBRG/xlm-mlm-abl-neither-plains-cree-en-calibrated"),
    ("Ablation: mono-MLM warmup", "KonradBRG/xlm-mlm-abl-mono-mlm-plains-cree-en-calibrated"),
    ("Ablation: TLM+contrastive", "KonradBRG/xlm-mlm-abl-tlm-contrastive-plains-cree-en-calibrated"),
]

CORPUS_FILE = "data/bloomfield_texts_sentences.parquet"
IDIOMS_FILE = "data/idioms.txt"
ANNOT_FILE  = "data/figurative/annotations.parquet"
TEACHER_ID  = "KonradBRG/deberta-v3-base-figurative"


# ── Task: checkpoints ─────────────────────────────────────────────────────────

def task_checkpoints() -> None:
    """Evaluate all hub checkpoints on the idiom golden set."""
    output_file = "data/figurative/results_table.csv"

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
    for name, ckpt in _CHECKPOINTS:
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

    os.makedirs("data/figurative", exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print(f"\n{'='*64}")
    print("FULL RESULTS TABLE")
    print(f"{'='*64}")
    metric_cols = ["model", "cree_idiom_acc", "cree_fig_rate", "cree_p_idiom",
                   "en_idiom_acc",   "en_fig_rate",   "en_p_idiom"]
    display = df[[c for c in metric_cols if c in df.columns]]
    print(display.to_string(index=False))
    print(f"\nSaved full table to {output_file}")


# ── Task: figurative-rate ─────────────────────────────────────────────────────

def task_figurative_rate() -> None:
    """Figurative rate on the Bloomfield corpus."""
    output_file = "data/figurative/eval_figurative_rate.csv"

    df = pd.read_parquet(CORPUS_FILE).dropna(subset=["text_cree"])
    cree_texts = df["text_cree"].tolist()
    print(f"Corpus: {len(cree_texts):,} Cree sentences")

    rows = []
    for name, ckpt in _RATE_MODELS:
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

    os.makedirs("data/figurative", exist_ok=True)
    out = pd.DataFrame(rows)
    out.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\nSaved to {output_file}")
    display_cols = ["model", "figurative_rate"] + [f"rate_{l}" for l in LABEL_NAMES]
    print(out[[c for c in display_cols if c in out.columns]].to_string(index=False))


# ── Task: simile ──────────────────────────────────────────────────────────────

def task_simile() -> None:
    """Simile detection via surface marker tâpiskôc."""
    import re
    from collections import Counter

    output_file = "data/figurative/eval_simile_detection.csv"
    detail_file = "data/figurative/eval_simile_detection_detail.csv"

    # tâpiskôc and common orthographic variants
    simile_re = re.compile(r"tâpiskôc|tâpiskôt|tapiskoc|tapiskot", re.IGNORECASE)

    df = pd.read_parquet(CORPUS_FILE).dropna(subset=["text_cree"])
    simile_df = df[df["text_cree"].str.contains(simile_re, na=False)].reset_index(drop=True)
    other_df  = df[~df["text_cree"].str.contains(simile_re, na=False)].reset_index(drop=True)

    print(f"Corpus     : {len(df):,} sentences")
    print(f"tâpiskôc   : {len(simile_df):,} simile candidates")
    print(f"Non-simile : {len(other_df):,} sentences")

    simile_texts = simile_df["text_cree"].tolist()
    other_texts  = other_df["text_cree"].tolist()

    en_simile_texts = simile_df["text_en"].dropna().tolist()
    print(f"\n{'='*60}")
    print(f"  Teacher sanity check ({len(en_simile_texts)} English tâpiskôc translations)")
    print(f"  {TEACHER_ID}")
    t_model, t_tok = load_model(TEACHER_ID)
    t_preds  = predict_sentences(en_simile_texts, t_model, t_tok)
    t_labels = [p["label"] for p in t_preds]
    del t_model
    torch.cuda.empty_cache()

    t_dist = Counter(t_labels)
    t_simile_tp = t_dist.get("simile", 0) / len(t_labels)
    print(f"  Teacher label distribution on English tâpiskôc translations:")
    for label in ["literal", "idiom", "metaphor", "simile"]:
        n = t_dist.get(label, 0)
        print(f"    {label:10s}: {n:3d}  ({n / len(t_labels):.1%})")
    print(f"  → Teacher simile TP: {t_simile_tp:.1%}")
    if t_simile_tp < 0.10:
        print("  ! Teacher rarely labels these as simile — "
              "student soft labels are noisy for this class.")

    summary_rows = []
    detail_rows  = []

    for name, ckpt in _RATE_MODELS:
        print(f"\n{'='*60}\n  {name}")
        try:
            model, tok = load_model(ckpt)

            simile_preds = predict_sentences(simile_texts, model, tok)
            other_preds  = predict_sentences(other_texts,  model, tok)

            simile_labels = [p["label"] for p in simile_preds]
            other_labels  = [p["label"] for p in other_preds]

            tp_rate = sum(l == "simile" for l in simile_labels) / len(simile_labels)
            fp_rate = sum(l == "simile" for l in other_labels)  / len(other_labels)
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

    os.makedirs("data/figurative", exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(output_file,  index=False, encoding="utf-8-sig")
    pd.DataFrame(detail_rows).to_csv(detail_file,   index=False, encoding="utf-8-sig")

    print(f"\nSaved summary → {output_file}")
    print(f"Saved details → {detail_file}")
    summary = pd.DataFrame(summary_rows)
    print(summary[["model", "simile_tp_rate", "simile_fp_rate", "fig_rate_on_similes"]].to_string(index=False))


# ── Task: consistency ─────────────────────────────────────────────────────────

def task_consistency() -> None:
    """English-Cree label consistency evaluation."""
    import numpy as np

    output_file = "data/figurative/eval_consistency.csv"

    def get_probs(texts: list[str], model, tokenizer) -> "np.ndarray":
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
    for name, ckpt in _CONSISTENCY_MODELS:
        print(f"\n{'='*60}\n  {name}")
        try:
            model, tok = load_model(ckpt)

            if ckpt == TEACHER_ID:
                # Self-consistency: English vs English (upper bound sanity check)
                student_probs  = teacher_probs.copy()
            else:
                student_probs  = get_probs(cree_texts, model, tok)

            student_labels = student_probs.argmax(axis=1)

            agreement = (student_labels == teacher_labels).mean()

            per_class = {}
            for i, label in enumerate(LABEL_NAMES):
                mask = teacher_labels == i
                if mask.sum() > 0:
                    per_class[f"agree_{label}"] = (student_labels[mask] == i).mean()
                else:
                    per_class[f"agree_{label}"] = float("nan")

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

    os.makedirs("data/figurative", exist_ok=True)
    out = pd.DataFrame(rows)
    out.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\nSaved to {output_file}")
    print(out[["model", "label_agreement", "mean_kl_div"]].to_string(index=False))


# ── Task: validation ──────────────────────────────────────────────────────────

def task_validation() -> None:
    """Evaluate CLKD models against DeepSeek-annotated validation set."""
    from sklearn.metrics import classification_report

    output_full = "data/figurative/eval_validation_full.csv"
    output_gold = "data/figurative/eval_validation_gold.csv"

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
        for name, ckpt in _VALIDATION_MODELS:
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
    pd.DataFrame(rows_full).to_csv(output_full, index=False, encoding="utf-8-sig")
    print(f"\nSaved → {output_full}")

    print("\n\n── Gold subset (footnote_applies=True) ──────────────────────")
    rows_gold = evaluate(gold, "gold")
    pd.DataFrame(rows_gold).to_csv(output_gold, index=False, encoding="utf-8-sig")
    print(f"\nSaved → {output_gold}")


# ── CLI ───────────────────────────────────────────────────────────────────────

_ALL_TASKS = ["checkpoints", "figurative-rate", "simile", "consistency", "validation"]

_TASK_FNS = {
    "checkpoints":    task_checkpoints,
    "figurative-rate": task_figurative_rate,
    "simile":         task_simile,
    "consistency":    task_consistency,
    "validation":     task_validation,
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--task",
        choices=_ALL_TASKS,
        default=None,
        help="Task to run. Omit to run all tasks.",
    )
    args = p.parse_args()

    tasks = [args.task] if args.task else _ALL_TASKS
    for task in tasks:
        print(f"\n{'#'*70}")
        print(f"  TASK: {task}")
        print(f"{'#'*70}\n")
        _TASK_FNS[task]()


if __name__ == "__main__":
    main()
