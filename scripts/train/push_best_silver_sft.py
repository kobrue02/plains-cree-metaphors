"""
Read data/figurative/silver_sft_sweep_results.parquet (written by every run
of jobs/silver_sft_worker.sh via src.figurative.silver_sft._save_sweep_result),
find the freeze_n_layers config with the highest gold macro-F1, and push only
that checkpoint to the Hub — the sweep's other ~16 local checkpoints are left
on disk (delete manually once you've confirmed the winner) rather than ever
touching the Hub, so this is the only Hub push the whole sweep produces.

Usage:
  python scripts/train/push_best_silver_sft.py --dry-run
  python scripts/train/push_best_silver_sft.py --hub-model-id KonradBRG/xlm-mlm-plains-cree-en-silver-sft-hierarchical
"""

from __future__ import annotations
import argparse, os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.figurative.hierarchical import HierarchicalFigurativeConfig, HierarchicalFigurativeModel
from src.figurative.silver_sft import RESULTS_FILE


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-file",  default=RESULTS_FILE)
    p.add_argument("--hub-model-id",  default="KonradBRG/xlm-mlm-plains-cree-en-silver-sft-hierarchical")
    p.add_argument("--dry-run", action="store_true", help="Print the winner without pushing")
    args = p.parse_args()

    df = pd.read_parquet(args.results_file)
    if df.empty:
        sys.exit(f"{args.results_file} is empty — no sweep results to compare")

    df = df.sort_values("macro_f1", ascending=False)
    print(df[["freeze_n_layers", "macro_f1", "literal_f1", "idiom_f1", "metaphor_f1", "simile_f1"]]
          .to_string(index=False))

    best = df.iloc[0]
    print(f"\nBest: freeze_n_layers={best['freeze_n_layers']}  "
          f"macro_f1={best['macro_f1']:.4f}  output_dir={best['output_dir']}")

    if args.dry_run:
        print("\n--dry-run: not pushing to the Hub.")
        return

    if not os.path.isdir(best["output_dir"]):
        sys.exit(f"{best['output_dir']} not found locally — was it moved or deleted?")

    if bool(best["hierarchical"]):
        model = HierarchicalFigurativeModel.from_pretrained(best["output_dir"])
    else:
        model = AutoModelForSequenceClassification.from_pretrained(best["output_dir"])
    tokenizer = AutoTokenizer.from_pretrained(best["output_dir"])

    model.push_to_hub(args.hub_model_id)
    tokenizer.push_to_hub(args.hub_model_id)
    print(f"Pushed → https://huggingface.co/{args.hub_model_id}")


if __name__ == "__main__":
    main()
