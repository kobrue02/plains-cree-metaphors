"""
Fetch the best run from a wandb sweep and save its config.

Usage:
  python3 scripts/best_sweep_config.py <sweep_id>
  python3 scripts/best_sweep_config.py konradbrg-uni/FNLP/0cvfxt6y

Outputs:
  data/sweep_best/<sweep_id>.json   — best hyperparameters + metric
  Prints a ready-to-use pipeline.py command.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Stage → pipeline.py flag mapping ──────────────────────────────────────────

STAGE_FLAGS = {
    "calibrate": {
        "learning_rate": "--calibrate-lr",
        "epochs":        "--calibrate-epochs",
        "literal_ratio": "--literal-ratio",
    },
    "clkd": {
        "learning_rate":   "--clkd-lr",
        "epochs":          "--clkd-epochs",
        "temperature":     None,          # not yet a pipeline flag
        "freeze_n_layers": "--freeze-layers",
    },
    "tlm": {
        "learning_rate": "--tlm-lr",
        "epochs":        "--tlm-epochs",
        "batch_size":    "--batch-size",
        "grad_accum":    "--grad-accum",
    },
}

METRIC = "eval/macro_f1"   # for calibrate
METRIC_FALLBACK = {
    "calibrate": "eval/macro_f1",
    "clkd":      "eval/kl_epoch",
    "tlm":       "eval/loss",
}


def _infer_stage(run) -> str:
    """Guess the stage from the run command or config."""
    cmd = getattr(run, "metadata", {}).get("args", [])
    for c in cmd:
        if c in ("calibrate", "clkd", "tlm"):
            return c
    cfg = dict(run.config)
    if "literal_ratio" in cfg:
        return "calibrate"
    if "temperature" in cfg or "freeze_n_layers" in cfg:
        return "clkd"
    return "tlm"


def fetch_best(sweep_path: str) -> dict:
    try:
        import wandb
    except ImportError:
        sys.exit("wandb not installed — run: pip install wandb")

    api = wandb.Api()
    sweep = api.sweep(sweep_path)
    runs = [r for r in sweep.runs if r.state in ("finished", "crashed")]

    if not runs:
        sys.exit("No completed runs in this sweep yet.")

    stage = _infer_stage(runs[0])
    metric_key = METRIC_FALLBACK.get(stage, METRIC)
    higher_is_better = stage == "calibrate"

    def score(run):
        val = run.summary.get(metric_key)
        if val is None:
            return float("-inf") if higher_is_better else float("inf")
        return val

    best = max(runs, key=score) if higher_is_better else min(runs, key=score)
    best_score = score(best)

    # Extract only the parameters the sweep actually varied
    swept_params = set(sweep.config.get("parameters", {}).keys())
    swept_config = {k: v for k, v in best.config.items() if k in swept_params}

    return {
        "sweep_id":   sweep_path,
        "stage":      stage,
        "metric":     metric_key,
        "best_value": best_score,
        "run_id":     best.id,
        "run_name":   best.name,
        "config":     swept_config,
        "n_runs":     len(runs),
    }


def build_pipeline_cmd(result: dict) -> str:
    stage = result["stage"]
    cfg = result["config"]
    flags = STAGE_FLAGS.get(stage, {})

    parts = ["python3 pipeline.py"]

    # Stage skips (run only the relevant stage)
    skip_map = {
        "calibrate": ["--skip-tlm", "--skip-clkd"],
        "clkd":      ["--skip-tlm", "--skip-calibrate"],
        "tlm":       ["--skip-clkd", "--skip-calibrate"],
    }
    parts += skip_map.get(stage, [])

    # Hyperparameters
    for param, flag in flags.items():
        if flag and param in cfg:
            val = cfg[param]
            if isinstance(val, float):
                parts.append(f"{flag} {val:.2e}")
            else:
                parts.append(f"{flag} {val}")

    parts += ["--base-model FacebookAI/xlm-mlm-100-1280", "--model-id xlm-mlm"]
    return " \\\n  ".join(parts)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sweep_id", help="Sweep path, e.g. konradbrg-uni/FNLP/0cvfxt6y")
    p.add_argument("--out-dir", default="data/sweep_best")
    args = p.parse_args()

    print(f"Fetching sweep: {args.sweep_id} ...")
    result = fetch_best(args.sweep_id)

    # Save
    os.makedirs(args.out_dir, exist_ok=True)
    slug = args.sweep_id.replace("/", "_")
    out_path = os.path.join(args.out_dir, f"{slug}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    # Report
    print(f"\n{'─'*60}")
    print(f"  Stage   : {result['stage']}")
    print(f"  Metric  : {result['metric']} = {result['best_value']:.4f}")
    print(f"  Run     : {result['run_name']}  ({result['run_id']})")
    print(f"  Trials  : {result['n_runs']}")
    print(f"  Config  :")
    for k, v in result["config"].items():
        print(f"    {k:<20} {v}")
    print(f"\n  Saved → {out_path}")
    print(f"{'─'*60}")
    print(f"\nRun with best config:\n")
    print(f"  {build_pipeline_cmd(result)}")
    print()


if __name__ == "__main__":
    main()
