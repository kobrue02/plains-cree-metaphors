"""
The core results table: Majority / no-adaptation / +TLM / +TLM+CLKD / +TLM+CLKD+SFT.

  | Model        | Macro F1 | Literal | Idiom | Metaphor | Simile |
  | ------------ | -------- | ------- | ----- | -------- | ------ |
  | Majority     |          |         |       |          |        |
  | No adapt.    |          |         |       |          |        |
  | +TLM         |          |         |       |          |        |
  | +TLM+CLKD    |          |         |       |          |        |
  | Full         |          |         |       |          |        |

Model-agnostic — "no adaptation" can be any multilingual encoder, not
specifically one family. Defaults below point at the FacebookAI/xlm-mlm-100-1280
lineage since that's the furthest along right now: its +TLM and +TLM+CLKD
classifier checkpoints already exist on the Hub (from earlier config.py-preset
runs, not the pipeline.py ablation runs — see --tlm-checkpoint/--clkd-checkpoint
below for exactly which). Only the "no adaptation" checkpoint is missing for
this lineage; see scripts/train/train_figurative_english.py to produce it.
To point this at a different encoder entirely, override all three
--baseline-checkpoint/--tlm-checkpoint/--clkd-checkpoint explicitly.

Evaluation protocol differs by row, because each row's checkpoint has a
different relationship to the gold annotation pool:

  - Majority, no-adaptation, +TLM, +TLM+CLKD: none of these checkpoints were
    ever trained on the Cree gold set (majority is data-free; the other three
    only ever see English VUA20+MAGPIE+FLUTE and/or the *unlabeled*
    Cree-English parallel corpus for CLKD). So it's safe to evaluate them
    directly against the WHOLE gold pool with a single checkpoint — there's
    no leakage to worry about.
  - Full (+TLM+CLKD+SFT): calibration *is* trained on (subsets of) the gold
    pool, so scoring the production calibrated model against that same pool
    would be in-sample and optimistic. This row instead uses the honest
    5-fold cross-validation protocol from scripts/evals/eval_cv.py: each
    fold's calibration checkpoint predicts only on the fold it never trained
    on, and the five out-of-fold prediction sets are concatenated to cover
    the whole pool without leakage. Requires jobs/calibrate_cv.sh to have
    been run for --model-id first (the *-fold{0..4} local checkpoints it
    produces are never pushed to the Hub, so this step needs the cluster).

Usage:
  python scripts/evals/figurative_results_table.py
  python scripts/evals/figurative_results_table.py --model-id xlm-mlm   # default, matches the CV-fold naming
  python scripts/evals/figurative_results_table.py --gold-footnoted-only
  python scripts/evals/figurative_results_table.py \
      --baseline-checkpoint KonradBRG/some-other-encoder-figurative \
      --tlm-checkpoint      KonradBRG/some-other-encoder-plains-cree-en-tlm-figurative \
      --clkd-checkpoint     KonradBRG/some-other-encoder-plains-cree-en-clkd
"""

from __future__ import annotations
import argparse, os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
import pandas as pd

from src.figurative.predict import load_model, predict_sentences
from src.figurative.data import LABEL_NAMES
from scripts.evals.eval_all import metrics_for

GOLD_FILE   = "data/figurative/bloomfield_annotated.csv"
N_FOLDS     = 5
OUTPUT_FILE = "data/figurative/figurative_results_table.csv"


def load_gold(footnoted_only: bool) -> pd.DataFrame:
    df = pd.read_csv(GOLD_FILE)
    if footnoted_only:
        df = df[df["footnote_applies"] == True]
    df = df.dropna(subset=["text_cree", "label"]).copy()
    df["label"] = df["label"].str.strip().str.lower().map(
        lambda x: x if x in LABEL_NAMES else "literal"
    )
    return df


def fold_dir(model_id: str, fold: int) -> str:
    return f"data/calibrated_{model_id}-fold{fold}"


# ── Row: Majority class ────────────────────────────────────────────────────────

def eval_majority(gold: pd.DataFrame) -> dict:
    y_true = gold["label"].tolist()
    y_pred = ["literal"] * len(gold)
    return metrics_for(y_true, y_pred)


# ── Row: Random (uniform over the 4 classes) ───────────────────────────────────

def eval_random(gold: pd.DataFrame, trials: int = 200, seed: int = 42) -> dict:
    """Uniform random guessing among the 4 classes, averaged over many trials
    (a single draw is noisy for the rare classes — idiom/metaphor/simile only
    have a handful of instances in the gold pool)."""
    import numpy as np
    rng = np.random.RandomState(seed)
    y_true = gold["label"].tolist()
    n = len(y_true)

    accum = {k: 0.0 for k in
             [f"{p}_{l}" for p in ("p", "r", "f1") for l in LABEL_NAMES]
             + ["macro_p", "macro_r", "macro_f1", "accuracy"]}
    for _ in range(trials):
        y_pred = rng.choice(LABEL_NAMES, size=n).tolist()
        m = metrics_for(y_true, y_pred)
        for k in accum:
            accum[k] += m[k]
    return {k: round(v / trials, 4) for k, v in accum.items()}


# ── Rows evaluated directly (checkpoint never trained on the gold pool) ────────

def eval_direct(checkpoint: str, gold: pd.DataFrame) -> dict:
    model, tok = load_model(checkpoint)
    preds = predict_sentences(gold["text_cree"].tolist(), model, tok,
                               batch_size=32, max_length=256)
    y_true = gold["label"].tolist()
    y_pred = [p["label"] for p in preds]
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics_for(y_true, y_pred)


# ── Row: Full (+TLM+CLKD+SFT) — honest 5-fold CV ────────────────────────────────

def eval_cv(model_id: str, gold: pd.DataFrame) -> dict | None:
    cv_folds_file = "data/figurative/cv_folds.parquet"
    if not os.path.exists(cv_folds_file):
        print(f"  [skip] {cv_folds_file} not found — run scripts/data/build_cv_folds.py first")
        return None

    folds = pd.read_parquet(cv_folds_file)
    missing = [f for f in range(N_FOLDS) if not os.path.isdir(fold_dir(model_id, f))]
    if missing:
        print(f"  [skip] missing fold checkpoint(s) {missing} for model_id={model_id!r} — "
              f"run: bash jobs/calibrate_cv.sh --base-model <hf-id> --model-id {model_id} "
              f"--calibrate-from <clkd-checkpoint>")
        return None

    merged = folds.merge(
        gold[["text_cree", "label"]].drop_duplicates("text_cree"),
        on="text_cree", how="left",
    ).dropna(subset=["label"])

    all_preds = []
    for f in range(N_FOLDS):
        sub = merged[merged["fold"] == f]
        if sub.empty:
            continue
        ckpt = fold_dir(model_id, f)
        model, tok = load_model(ckpt)
        preds = predict_sentences(sub["text_cree"].tolist(), model, tok,
                                   batch_size=32, max_length=128)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        out = sub[["text_cree", "label"]].copy()
        out["pred"] = [p["label"] for p in preds]
        all_preds.append(out)

    combined = pd.concat(all_preds, ignore_index=True)
    return metrics_for(combined["label"].tolist(), combined["pred"].tolist())


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-id", default="xlm-mlm",
                   help="model_id used for the CV-fold calibrated checkpoints in the "
                        "'Full' row (default: xlm-mlm, matching jobs/calibrate_cv.sh naming)")
    # No real checkpoint defaults here on purpose — any of these three flags
    # triggers a real multi-GB Hub download the moment the script runs, so
    # each one must be passed explicitly, never silently defaulted.
    # XLM-MLM-100-1280 already has ready checkpoints for the two TLM/CLKD
    # flags (see the module docstring above for exact IDs) — pass them
    # yourself when you're ready to spend the bandwidth:
    #   --tlm-checkpoint  KonradBRG/xlm-mlm-100-1280-plains-cree-en-tlm-figurative
    #   --clkd-checkpoint KonradBRG/xlm-mlm-100-1280-plains-cree-en-clkd-full
    p.add_argument("--baseline-checkpoint", default=None,
                   help="'No adaptation' checkpoint. No default — this row hasn't been "
                        "trained for any encoder yet; see scripts/train/train_figurative_english.py")
    p.add_argument("--tlm-checkpoint", default=None,
                   help="'+TLM' checkpoint. No default (downloading one is a multi-GB "
                        "network op) — pass explicitly, e.g. the XLM-MLM-100-1280 one "
                        "named in the module docstring")
    p.add_argument("--clkd-checkpoint", default=None,
                   help="'+TLM+CLKD' checkpoint. No default, same reasoning as --tlm-checkpoint")
    p.add_argument("--gold-footnoted-only", action="store_true",
                   help="Restrict gold to footnote_applies=True (219 rows) instead of "
                        "all footnoted-paragraph sentences (1,225 rows)")
    p.add_argument("--out", default=OUTPUT_FILE)
    args = p.parse_args()

    mid = args.model_id
    checkpoints = {
        "Majority":   None,
        "Random":     None,
        "No adapt.":  args.baseline_checkpoint,
        "+TLM":       args.tlm_checkpoint,
        "+TLM+CLKD":  args.clkd_checkpoint,
        "Full":       None,  # resolved via CV, not a single checkpoint
    }

    gold = load_gold(args.gold_footnoted_only)
    print(f"Gold pool: {len(gold):,} sentences "
          f"({'footnote-verified only' if args.gold_footnoted_only else 'all footnoted-paragraph sentences'})")

    rows = []
    for label in ["Majority", "Random", "No adapt.", "+TLM", "+TLM+CLKD"]:
        if label not in ("Majority", "Random") and not checkpoints[label]:
            flag = {"No adapt.": "--baseline-checkpoint",
                    "+TLM": "--tlm-checkpoint",
                    "+TLM+CLKD": "--clkd-checkpoint"}[label]
            print(f"\n── {label} ──\n  SKIPPED — no checkpoint given ({flag} not set)")
            rows.append({"model": label, "checkpoint": None, "error": "no checkpoint given"})
            continue
        print(f"\n── {label} ──")
        try:
            if label == "Majority":
                m = eval_majority(gold)
            elif label == "Random":
                m = eval_random(gold)
            else:
                print(f"  checkpoint: {checkpoints[label]}")
                m = eval_direct(checkpoints[label], gold)
            rows.append({"model": label, "checkpoint": checkpoints[label], **m})
        except Exception as exc:
            print(f"  SKIPPED — {exc}")
            rows.append({"model": label, "checkpoint": checkpoints[label], "error": str(exc)})

    print(f"\n── Full (+TLM+CLKD+SFT, 5-fold CV) ──")
    m = eval_cv(mid, gold)
    if m is not None:
        rows.append({"model": "Full", "checkpoint": f"data/calibrated_{mid}-fold{{0..4}}", **m})
    else:
        rows.append({"model": "Full", "checkpoint": f"data/calibrated_{mid}-fold{{0..4}}",
                     "error": "fold checkpoints not available"})

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"\n{'='*90}")
    print("| Model | Macro P | Macro R | Macro F1 | Literal | Idiom | Metaphor | Simile |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        if "error" in r:
            print(f"| {r['model']} | — | — | — | — | — | — | — |  (not available: {r['error'][:50]})")
        else:
            print(f"| {r['model']} | {r['macro_p']:.3f} | {r['macro_r']:.3f} | {r['macro_f1']:.3f} | "
                  f"{r['f1_literal']:.3f} | {r['f1_idiom']:.3f} | {r['f1_metaphor']:.3f} | {r['f1_simile']:.3f} |")
    print(f"{'='*90}")
    print(f"\nFull metrics (incl. precision/recall) saved → {args.out}")


if __name__ == "__main__":
    main()
