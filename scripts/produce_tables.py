"""Run every table-producing evaluation step needed for the paper in one go; each step runs even if an earlier one fails or a checkpoint isn't ready yet, with failures/skips collected into a summary at the end."""

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

    p.add_argument("--tlm-eval-no-infonce-checkpoint",
                   default="KonradBRG/xlm-mlm-abl-no-clkd-plains-cree-en-tlm",
                   help="Checkpoint for tab:tlm-eval's '+TLM' column (no InfoNCE, alpha=0.0). "
                        "Defaults to the alpha=0.0 TLM checkpoint trained for the ablation "
                        "table's no_infonce condition — NOT tlm_eval_table.py's own default, "
                        "which points at the production (alpha=0.2) checkpoint.")
    p.add_argument("--tlm-eval-infonce-checkpoint",
                   default="KonradBRG/xlm-mlm-plains-cree-en-tlm",
                   help="Checkpoint for tab:tlm-eval's '+TLM+InfoNCE' column (alpha=0.2). "
                        "Defaults to the actual production TLM checkpoint, matching the "
                        "'-InfoNCE only' decision for tab:ablation-loo — the ablation's "
                        "alpha=0.1 tlm_contrastive checkpoint is a separate data point, not "
                        "used here.")

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
             "--model-id",            "xlm-mlm-abl-full",  # matches jobs/ablation.sh's Full-condition fold naming, not the "xlm-mlm" default
             "--baseline-checkpoint", args.baseline_checkpoint,
             "--tlm-checkpoint",      args.tlm_checkpoint,
             "--clkd-checkpoint",     args.clkd_checkpoint],
        ))

    if not args.skip_tlm_eval:
        results.append(run_step(
            "3. tlm_eval_table.py (tab:tlm-eval)",
            [py, "scripts/evaluate/tlm_eval_table.py",
             "--model", f"+TLM={args.tlm_eval_no_infonce_checkpoint}",
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
