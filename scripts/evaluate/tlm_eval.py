"""
TLM intrinsic evaluation: held-out perplexity + bitext retrieval.

Reports per-model:
  - MLM pseudo-perplexity on held-out Cree and English text
  - Bitext retrieval Recall@1, Recall@5, MRR  (Cree→EN and EN→Cree)

Pass --baseline to compare against the raw base model side-by-side.

Usage:
  python scripts/evaluate/tlm_eval.py \\
      --model data/tlm_xlm-mlm \\
      --baseline FacebookAI/xlm-mlm-100-1280

  # Use the Hub checkpoint instead of local dir:
  python scripts/evaluate/tlm_eval.py \\
      --model KonradBRG/xlm-mlm-plains-cree-en-tlm \\
      --baseline FacebookAI/xlm-mlm-100-1280 \\
      --n 300
"""

from __future__ import annotations
import argparse, os, sys
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForMaskedLM

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.device import get_device
from src.mt.tlm_eval import bitext_retrieval, pseudo_perplexity

SENTENCES_FILE = "data/sentences.parquet"
DEFAULT_N      = 500   # evaluation pairs (randomly sampled from held-out split)
SEED           = 42

def evaluate(model_name: str, cree: list[str], en: list[str], device: str) -> dict:
    print(f"\n  Loading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Use ForMaskedLM so we get both hidden states (for retrieval) and MLM head (for perplexity)
    mlm_model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)

    print("  Computing perplexity ...")
    ppl_cree = pseudo_perplexity(mlm_model, tokenizer, cree, device)
    ppl_en   = pseudo_perplexity(mlm_model, tokenizer, en,   device)

    # Retrieval — embeddings from the base model (strip MLM head)
    print("  Computing bitext retrieval ...")
    base_model = mlm_model.base_model
    retr = bitext_retrieval(base_model, tokenizer, cree, en, device)

    return {
        "ppl_cree": ppl_cree,
        "ppl_en":   ppl_en,
        "retrieval": retr,
    }

def print_results(label: str, res: dict) -> None:
    print(f"\n{'─'*58}")
    print(f"  {label}")
    print(f"{'─'*58}")
    print(f"  Perplexity   Cree : {res['ppl_cree']:.2f}")
    print(f"               EN   : {res['ppl_en']:.2f}")
    for direction, m in res["retrieval"].items():
        print(f"  {direction:<10}  R@1={m['R@1']:.3f}  R@5={m['R@5']:.3f}  MRR={m['MRR']:.3f}")

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model",    required=True,
                   help="TLM checkpoint path or Hub ID")
    p.add_argument("--baseline", default=None,
                   help="Base model to compare against (e.g. FacebookAI/xlm-mlm-100-1280)")
    p.add_argument("--sentences-file", default=SENTENCES_FILE)
    p.add_argument("--n",        type=int, default=DEFAULT_N,
                   help="Number of held-out pairs to evaluate on (default: 500)")
    p.add_argument("--seed",     type=int, default=SEED)
    args = p.parse_args()

    device = get_device()
    print(f"Device: {device}")

    df = pd.read_parquet(args.sentences_file)
    df = df.dropna(subset=["text_cree", "text_en"])
    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    eval_df = df.tail(args.n)  # seeded shuffle makes this a fixed, reproducible held-out slice
    cree = eval_df["text_cree"].astype(str).tolist()
    en   = eval_df["text_en"].astype(str).tolist()
    print(f"Evaluating on {len(cree)} held-out pairs from {args.sentences_file}")

    models_to_eval = [(args.model, "TLM")]
    if args.baseline:
        models_to_eval.append((args.baseline, "Baseline"))

    all_results = {}
    for model_name, label in models_to_eval:
        all_results[label] = evaluate(model_name, cree, en, device)
        print_results(label, all_results[label])

    if args.baseline and "Baseline" in all_results and "TLM" in all_results:
        base = all_results["Baseline"]
        tlm  = all_results["TLM"]
        print(f"\n{'─'*58}")
        print("  Δ TLM − Baseline")
        print(f"{'─'*58}")
        print(f"  Perplexity   Cree : {tlm['ppl_cree'] - base['ppl_cree']:+.2f}")
        print(f"               EN   : {tlm['ppl_en']   - base['ppl_en']:+.2f}")
        for direction in ("Cree→EN", "EN→Cree"):
            for k in ("R@1", "R@5", "MRR"):
                delta = tlm["retrieval"][direction][k] - base["retrieval"][direction][k]
                print(f"  {direction:<10}  {k}: {delta:+.3f}")
    print()

if __name__ == "__main__":
    main()
