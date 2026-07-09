"""
Build the Base / +TLM / +TLM+InfoNCE comparison table for the writeup:

  | Model        | Cree PPL | English PPL | Cree->EN MRR | EN->Cree MRR |

Reuses evaluate() from scripts/evaluate/tlm_eval.py (same held-out split,
same pseudo-perplexity + bitext retrieval metrics) across a fixed list of
checkpoints instead of just a --model/--baseline pair.

Usage:
  python scripts/evaluate/tlm_eval_table.py
  python scripts/evaluate/tlm_eval_table.py --n 500
  python scripts/evaluate/tlm_eval_table.py --model "+TLM+InfoNCE=KonradBRG/xlm-mlm-plains-cree-en-tlm-infonce"
"""

from __future__ import annotations
import argparse, os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
import pandas as pd

from scripts.evaluate.tlm_eval import evaluate, SENTENCES_FILE, DEFAULT_N, SEED

# NOTE: the +TLM+InfoNCE row points at a *local* checkpoint path
# (data/tlm_xlm-mlm-abl-tlm-contrastive) that was never pushed to the Hub —
# only intermediate TLM/CLKD checkpoints get pushed with --push-intermediates,
# and this one wasn't. It only exists on the cluster right now. Either:
#   (a) push it from the cluster: from a node with the checkpoint, run
#       `python -c "from transformers import AutoModelForMaskedLM, AutoTokenizer; \
#         m=AutoModelForMaskedLM.from_pretrained('data/tlm_xlm-mlm-abl-tlm-contrastive'); \
#         t=AutoTokenizer.from_pretrained('data/tlm_xlm-mlm-abl-tlm-contrastive'); \
#         m.push_to_hub('KonradBRG/xlm-mlm-plains-cree-en-tlm-infonce'); \
#         t.push_to_hub('KonradBRG/xlm-mlm-plains-cree-en-tlm-infonce')"`
#       then swap the path below for that Hub id, or
#   (b) override it at the CLI with --model once you have a reachable path/id.
DEFAULT_MODELS = [
    ("Base XLM-R",   "FacebookAI/xlm-mlm-100-1280"),
    ("+TLM",         "KonradBRG/xlm-mlm-plains-cree-en-tlm"),
    ("+TLM+InfoNCE", "data/tlm_xlm-mlm-abl-tlm-contrastive"),
]

OUTPUT_FILE = "data/figurative/tlm_eval_table.parquet"


def _parse_model_arg(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        sys.exit(f"--model expects LABEL=CHECKPOINT, got: {spec!r}")
    label, checkpoint = spec.split("=", 1)
    return label, checkpoint


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", action="append", default=[],
                   metavar="LABEL=CHECKPOINT",
                   help="Add/override a row, e.g. --model \"+TLM+InfoNCE=KonradBRG/...\". "
                        "Repeatable. Overrides a default row with the same label.")
    p.add_argument("--sentences-file", default=SENTENCES_FILE)
    p.add_argument("--n",    type=int, default=DEFAULT_N)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--out",  default=OUTPUT_FILE)
    args = p.parse_args()

    models = dict(DEFAULT_MODELS)
    for spec in args.model:
        label, checkpoint = _parse_model_arg(spec)
        models[label] = checkpoint
    # preserve Base/+TLM/+TLM+InfoNCE ordering for whichever labels survive
    ordered_labels = [l for l, _ in DEFAULT_MODELS if l in models] + \
                     [l for l in models if l not in dict(DEFAULT_MODELS)]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    df = pd.read_parquet(args.sentences_file).dropna(subset=["text_cree", "text_en"])
    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    eval_df = df.tail(args.n)  # same held-out convention as tlm_eval.py
    cree = eval_df["text_cree"].astype(str).tolist()
    en   = eval_df["text_en"].astype(str).tolist()
    print(f"Evaluating on {len(cree)} held-out pairs from {args.sentences_file}\n")

    rows = []
    for label in ordered_labels:
        checkpoint = models[label]
        print(f"── {label} ({checkpoint}) ──")
        try:
            res = evaluate(checkpoint, cree, en, device)
            rows.append({
                "model":        label,
                "checkpoint":   checkpoint,
                "cree_ppl":     round(res["ppl_cree"], 3),
                "english_ppl":  round(res["ppl_en"], 3),
                "cree_to_en_r1":  round(res["retrieval"]["Cree→EN"]["R@1"], 4),
                "cree_to_en_r5":  round(res["retrieval"]["Cree→EN"]["R@5"], 4),
                "cree_to_en_mrr": round(res["retrieval"]["Cree→EN"]["MRR"], 4),
                "en_to_cree_r1":  round(res["retrieval"]["EN→Cree"]["R@1"], 4),
                "en_to_cree_r5":  round(res["retrieval"]["EN→Cree"]["R@5"], 4),
                "en_to_cree_mrr": round(res["retrieval"]["EN→Cree"]["MRR"], 4),
            })
        except Exception as exc:
            print(f"  SKIPPED — {exc}")
            rows.append({"model": label, "checkpoint": checkpoint, "error": str(exc)})
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    result_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    result_df.to_parquet(args.out, index=False)

    print(f"\n{'='*70}")
    print("| Model | Cree PPL | English PPL | Cree→EN MRR | EN→Cree MRR |")
    print("| --- | --- | --- | --- | --- |")
    for r in rows:
        if "error" in r:
            print(f"| {r['model']} | — | — | — | — |  (failed: {r['error'][:40]})")
        else:
            print(f"| {r['model']} | {r['cree_ppl']:.2f} | {r['english_ppl']:.2f} | "
                  f"{r['cree_to_en_mrr']:.3f} | {r['en_to_cree_mrr']:.3f} |")
    print(f"{'='*70}")
    print(f"\nFull metrics (incl. R@1/R@5) saved → {args.out}")


if __name__ == "__main__":
    main()
