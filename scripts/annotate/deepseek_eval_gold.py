"""
Score the DeepSeek dictionary-grounded labeling procedure directly against
Bloomfield's gold labels — i.e. treat DeepSeek itself as a classifier and ask
how well it does, rather than only comparing it against our own model's
predictions (see deepseek_agreement_eval.py for that separate question).

This never touches label/rationale/footnote_en: the annotation call only ever
sees text_cree/text_en (same _annotate_one() used in deepseek_label_pool.py on
the silver pool), so there is no leakage of Bloomfield's justification into
the prompt.

By default this scores only the 219 footnote-verified sentences (the same
fixed held-out set src/figurative/calibrate.py evaluates against) — pass
--include-unfootnoted to score against the full 1,225-row file instead.

Runs DeepSeek by default. Pass --nvidia-model <model-id> to run a different
model hosted on NVIDIA's endpoint (https://integrate.api.nvidia.com) against
the exact same procedure instead — see src/annotate/llm.py. Needs
NVIDIA_API_KEY set (env var or .env).

Usage:
  python scripts/annotate/deepseek_eval_gold.py
  python scripts/annotate/deepseek_eval_gold.py --limit 50   # smoke test
  python scripts/annotate/deepseek_eval_gold.py --nvidia-model mistralai/mistral-medium-3.5-128b
"""

from __future__ import annotations
import argparse, os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd

from scripts.annotate.deepseek_label_pool import annotate_pool, LABELS
from scripts.evals.eval_all import metrics_for, bootstrap_ci

GOLD_FILE   = "data/figurative/bloomfield_annotated.parquet"
OUTPUT_FILE = "data/figurative/deepseek_on_gold.parquet"
CACHE_JSONL = "data/figurative/deepseek_on_gold_cache.jsonl"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gold-file", default=GOLD_FILE)
    p.add_argument("--out",       default=None)
    p.add_argument("--cache",     default=None)
    p.add_argument("--limit",     type=int, default=None, help="Cap sentences (smoke test)")
    p.add_argument("--workers",   type=int, default=8)
    p.add_argument("--include-unfootnoted", action="store_true",
                   help="Score against the full file instead of just footnote_applies=True")
    p.add_argument("--nvidia-model", default=None, metavar="MODEL_ID",
                   help="Run this NVIDIA-hosted model instead of DeepSeek "
                        "(e.g. mistralai/mistral-medium-3.5-128b)")
    p.add_argument("--no-reasoning", action="store_true",
                   help="Don't send reasoning_effort — needed for plain instruct "
                        "models that don't support it (e.g. meta/llama-3.3-70b-instruct)")
    args = p.parse_args()

    model_label = args.nvidia_model or "deepseek"
    annotate_fn = None
    if args.nvidia_model:
        from src.annotate.llm import make_annotate_fn
        annotate_fn = make_annotate_fn(args.nvidia_model, reasoning=not args.no_reasoning)

    # auto-namespace by model so a non-DeepSeek run never shares DeepSeek's
    # cache/output file unless explicitly told to (--out/--cache override this)
    if args.nvidia_model:
        slug = model_label.replace("/", "_").replace(":", "_")
        args.out   = args.out   or OUTPUT_FILE.replace(".parquet", f"_{slug}.parquet")
        args.cache = args.cache or CACHE_JSONL.replace(".jsonl", f"_{slug}.jsonl")
    else:
        args.out   = args.out   or OUTPUT_FILE
        args.cache = args.cache or CACHE_JSONL

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    gold = pd.read_parquet(args.gold_file).dropna(subset=["text_cree", "text_en", "label"]).copy()
    gold["text_cree"] = gold["text_cree"].str.strip()
    gold["text_en"]   = gold["text_en"].str.strip()
    if not args.include_unfootnoted:
        gold = gold[gold["footnote_applies"] == True]
    if args.limit:
        gold = gold.head(args.limit)
    print(f"[gold] model={model_label}  {len(gold):,} sentences  —  {gold['label'].value_counts().to_dict()}")

    annotate_kwargs = {"annotate_fn": annotate_fn} if annotate_fn else {}
    annotations = annotate_pool(gold, cache_path=args.cache, workers=args.workers, **annotate_kwargs)

    # annotate_pool() drops sentences whose calls kept failing after retries
    # (see its own comment) rather than caching a fallback label — so not
    # every gold row is guaranteed a key here. Score only what succeeded
    # instead of crashing the whole run over one persistent failure.
    n_before = len(gold)
    gold = gold[gold["text_cree"].isin(annotations)].copy()
    if len(gold) < n_before:
        print(f"  [warn] {n_before - len(gold)} sentence(s) never got a usable "
              f"annotation after retries — scoring the remaining {len(gold)}")

    gold["deepseek_label"] = gold["text_cree"].map(lambda t: annotations[t]["label"])
    gold["reasoning"]      = gold["text_cree"].map(lambda t: annotations[t]["reasoning"])

    gold.to_parquet(args.out, index=False)
    print(f"Saved → {args.out}")

    y_true = gold["label"].tolist()
    y_pred = gold["deepseek_label"].tolist()
    metrics = metrics_for(y_true, y_pred)
    ci = bootstrap_ci(y_true, y_pred)
    metrics.update(ci)

    print(f"\n{'='*60}")
    print(f"{model_label}-vs-gold accuracy : {metrics['accuracy']:.1%}")
    print(f"{model_label}-vs-gold macro F1 : {metrics['macro_f1']:.4f} "
          f"[{ci['macro_f1_ci_lo']:.4f}, {ci['macro_f1_ci_hi']:.4f}]")
    print(f"{'='*60}")
    for label in LABELS:
        print(f"  {label:10s}: P={metrics[f'p_{label}']:.3f}  "
              f"R={metrics[f'r_{label}']:.3f}  F1={metrics[f'f1_{label}']:.3f}")

    print("\nConfusion (rows=gold, cols=DeepSeek):")
    print(pd.crosstab(gold["label"], gold["deepseek_label"]).to_string())

    summary_path = args.out.replace(".parquet", "_summary.parquet")
    pd.DataFrame([metrics]).to_parquet(summary_path, index=False)
    print(f"Saved summary → {summary_path}")


if __name__ == "__main__":
    main()
