"""Fetch the best run from a wandb sweep and save its config."""

from __future__ import annotations
import argparse
import json
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

STAGE_FLAGS = {
    "calibrate": {
        "learning_rate": "--calibrate-lr",
        "epochs":        "--calibrate-epochs",
        "literal_ratio": "--literal-ratio",
    },
    "clkd": {
        "learning_rate":   "--clkd-lr",
        "epochs":          "--clkd-epochs",
        "temperature":     "--clkd-temperature",
        "freeze_n_layers": "--freeze-layers",
    },
    "tlm": {
        "learning_rate":  "--tlm-lr",
        "epochs":         "--tlm-epochs",
        "batch_size":     "--batch-size",
        "grad_accum":     "--grad-accum",
        "sentences_file": "--sentences-file",
    },
    "pipeline": {
        "clkd_lr":                 "--clkd-lr",
        "clkd_epochs":             "--clkd-epochs",
        "clkd_temperature":        "--clkd-temperature",
        "clkd_freeze_layers":      "--freeze-layers",
        "calibrate_lr":            "--calibrate-lr",
        "calibrate_epochs":        "--calibrate-epochs",
        "calibrate_literal_ratio": "--literal-ratio",
    },
}

METRIC_FALLBACK = {
    "calibrate": "eval/macro_f1",
    "clkd":      "eval/best_gold_macro_f1",
    "tlm":       "eval/loss",
    "pipeline":  "eval/macro_f1",
}


def _infer_stage(run) -> str:
    """Guess the stage from the run command or config."""
    cmd = (getattr(run, "metadata", None) or {}).get("args", [])
    for c in cmd:
        if c in ("calibrate", "clkd", "tlm", "pipeline"):
            return c
    cfg = dict(run.config)
    if "clkd_lr" in cfg or "calibrate_lr" in cfg:
        return "pipeline"
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
    metric_key = METRIC_FALLBACK.get(stage, "eval/macro_f1")
    higher_is_better = stage in ("calibrate", "pipeline", "clkd")

    def score(run):
        val = run.summary.get(metric_key)
        if val is None:
            return float("-inf") if higher_is_better else float("inf")
        return val

    best = max(runs, key=score) if higher_is_better else min(runs, key=score)
    best_score = score(best)

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


def build_pipeline_cmd(result: dict, use_sbatch: bool = False) -> str:
    stage = result["stage"]
    cfg = result["config"]
    flags = STAGE_FLAGS.get(stage, {})

    prefix = "sbatch jobs/pipeline.sh" if use_sbatch else "python3 pipeline.py"
    parts = [prefix]

    skip_map = {
        "calibrate": ["--skip-tlm", "--skip-clkd"],
        "clkd":      ["--skip-tlm", "--skip-calibrate"],
        "tlm":       ["--skip-clkd", "--skip-calibrate"],
        "pipeline":  ["--skip-tlm"],  # TLM fixed; sweep covers CLKD + Calibrate
    }
    parts += skip_map.get(stage, [])

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
    p.add_argument("--run",    action="store_true",
                   help="Execute pipeline.py with the best config immediately")
    p.add_argument("--sbatch", action="store_true",
                   help="Submit jobs/pipeline.sh with the best config via sbatch")
    args = p.parse_args()

    if args.run and args.sbatch:
        p.error("--run and --sbatch are mutually exclusive")

    print(f"Fetching sweep: {args.sweep_id} ...")
    result = fetch_best(args.sweep_id)

    os.makedirs(args.out_dir, exist_ok=True)
    slug = args.sweep_id.replace("/", "_")
    out_path = os.path.join(args.out_dir, f"{slug}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

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

    cmd = build_pipeline_cmd(result, use_sbatch=args.sbatch)

    if args.run or args.sbatch:
        print(f"\n{'sbatch' if args.sbatch else 'Running'}:\n\n  {cmd}\n")
        import subprocess
        subprocess.run(cmd, shell=True, check=True)
    else:
        print(f"\nRun with best config:\n\n  {cmd}\n")


if __name__ == "__main__":
    main()
