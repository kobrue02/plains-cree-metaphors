"""
Run every table-producing evaluation step in one go, once the training/
ablation jobs have finished — a single entry point instead of remembering
which of the four separate scripts feeds which paper table.

  1. scripts/evals/eval_cv.py               -> tab:ablation-loo, and the
                                                "Full" row of Table 3
  2. scripts/evals/figurative_results_table.py -> the rest of Table 3
                                                (Majority/No adapt./+TLM/+TLM+CLKD)
  3. scripts/evaluate/tlm_eval_table.py      -> tab:tlm-eval (TLM intrinsic:
                                                pseudo-perplexity + bitext retrieval)
  4. scripts/evals/llm_annotator_comparison.py -> LLM-vs-gold comparison table

Each step runs even if an earlier one fails or a checkpoint isn't ready yet
(e.g. a training job still pending) — failures/skips are collected and
reported in a summary at the end instead of aborting the whole run, since a
missing checkpoint for one row shouldn't block the tables that don't need it.

This does real Hub checkpoint downloads and GPU inference — run it on a node
(see jobs/produce_tables.sh), not the login shell.

Usage:
  python scripts/produce_tables.py
  python scripts/produce_tables.py --skip-cv --skip-tlm-eval
  python scripts/produce_tables.py \
      --baseline-checkpoint KonradBRG/xlm-mlm-100-1280-figurative-baseline \
      --tlm-checkpoint      KonradBRG/xlm-mlm-100-1280-plains-cree-en-tlm-figurative \
      --clkd-checkpoint     KonradBRG/xlm-mlm-100-1280-plains-cree-en-clkd-full \
      --tlm-eval-infonce-checkpoint KonradBRG/xlm-mlm-abl-tlm-contrastive-plains-cree-en-tlm
"""

from __future__ import annotations
import argparse, subprocess, sys


def run_step(label: str, cmd: list[str]) -> tuple[str, str]:
    print(f"\n{'='*70}\n  {label}\n  $ {' '.join(cmd)}\n{'='*70}")
    result = subprocess.run(cmd)
    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"--- {label}: {status} ---")
    return label, status


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--skip-cv",             action="store_true")
    p.add_argument("--skip-results-table",  action="store_true")
    p.add_argument("--skip-tlm-eval",       action="store_true")
    p.add_argument("--skip-llm-comparison", action="store_true")

    p.add_argument("--baseline-checkpoint", default="KonradBRG/xlm-mlm-100-1280-figurative-baseline",
                   help="'No adaptation' row checkpoint for Table 3")
    p.add_argument("--tlm-checkpoint", default="KonradBRG/xlm-mlm-100-1280-plains-cree-en-tlm-figurative",
                   help="'+TLM' row checkpoint for Table 3")
    p.add_argument("--clkd-checkpoint", default="KonradBRG/xlm-mlm-100-1280-plains-cree-en-clkd-full",
                   help="'+TLM+CLKD' row checkpoint for Table 3")

    p.add_argument("--tlm-eval-infonce-checkpoint",
                   default="KonradBRG/xlm-mlm-abl-tlm-contrastive-plains-cree-en-tlm",
                   help="Checkpoint for tab:tlm-eval's '+TLM+InfoNCE' column. Defaults to the "
                        "ablation's alpha=0.1 contrastive checkpoint, since the production "
                        "'+TLM' checkpoint now trains with InfoNCE (alpha=0.2) baked in by "
                        "default and is no longer a distinct 'no InfoNCE' comparison point — "
                        "see the module docstring in scripts/evaluate/tlm_eval_table.py.")

    args = p.parse_args()
    py = sys.executable
    results: list[tuple[str, str]] = []

    if not args.skip_cv:
        results.append(run_step(
            "1. eval_cv.py (tab:ablation-loo + Table 3's Full row)",
            [py, "scripts/evals/eval_cv.py"],
        ))

    if not args.skip_results_table:
        results.append(run_step(
            "2. figurative_results_table.py (Table 3)",
            [py, "scripts/evals/figurative_results_table.py", "--gold-footnoted-only",
             "--baseline-checkpoint", args.baseline_checkpoint,
             "--tlm-checkpoint",      args.tlm_checkpoint,
             "--clkd-checkpoint",     args.clkd_checkpoint],
        ))

    if not args.skip_tlm_eval:
        results.append(run_step(
            "3. tlm_eval_table.py (tab:tlm-eval)",
            [py, "scripts/evaluate/tlm_eval_table.py",
             "--model", f"+TLM+InfoNCE={args.tlm_eval_infonce_checkpoint}"],
        ))

    if not args.skip_llm_comparison:
        results.append(run_step(
            "4. llm_annotator_comparison.py (LLM-vs-gold comparison)",
            [py, "scripts/evals/llm_annotator_comparison.py"],
        ))

    print(f"\n\n{'='*70}\n  SUMMARY\n{'='*70}")
    for label, status in results:
        print(f"  [{status:20s}] {label}")
    if any(status != "OK" for _, status in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
