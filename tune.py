"""
Single-trial runner for wandb hyperparameter sweeps.

Each invocation runs one trial for one pipeline stage, logging the
stage-appropriate metric to wandb so the sweep controller can guide
the next trial via Bayesian optimisation.

Metrics optimised per stage:
  tlm       → eval/loss       (minimize)  — TLM held-out MLM/TLM loss
  clkd      → eval/kl_epoch   (minimize)  — held-out KL on 10% corpus split
  calibrate → eval/macro_f1   (maximize)  — macro F1 on held-out annotation split
  pipeline  → eval/macro_f1   (maximize)  — joint CLKD+Calibrate sweep (recommended)

Usage:
  # 1. Create a sweep (once per stage):
  wandb sweep sweeps/pipeline.yaml           # prints <entity/project/sweep_id>

  # 2. Submit N parallel agents on the cluster:
  sbatch jobs/sweep_agent.sh <sweep_id>      # repeat for desired parallelism

  # 3. Pick best run in wandb UI or:
  python tune.py --stage calibrate --show-best --sweep-id <sweep_id>
"""

from __future__ import annotations
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import wandb

WANDB_PROJECT  = "FNLP"
TEACHER        = "KonradBRG/deberta-v3-base-figurative"
SENTENCES_FILE = "data/sentences.parquet"
CORPUS_FILE    = "data/bloomfield_texts_sentences.parquet"
ANNOT_FILE     = "data/figurative/bloomfield_annotated.parquet"


# ── Stage runners ──────────────────────────────────────────────────────────────

def run_tlm(args: argparse.Namespace) -> None:
    from funcs import fine_tune
    wandb.init(project=WANDB_PROJECT)
    cfg = wandb.config
    output_dir = f"{args.output_dir}_{wandb.run.id}"
    fine_tune(
        sentences_file=cfg.get("sentences_file", args.sentences_file),
        model_name=args.base_model,
        output_dir=output_dir,
        epochs=cfg.get("epochs", 15),
        batch_size=cfg.get("batch_size", 16),
        learning_rate=cfg.get("learning_rate", 2e-5),
        grad_accum=cfg.get("grad_accum", 2),
        max_length=args.max_length,
        wandb_project=WANDB_PROJECT,
    )


def run_clkd(args: argparse.Namespace) -> None:
    from funcs import figurative_distill
    wandb.init(project=WANDB_PROJECT)
    cfg = wandb.config
    output_dir = f"{args.output_dir}_{wandb.run.id}"
    figurative_distill(
        checkpoint=args.tlm_ckpt,
        teacher_checkpoint=args.teacher,
        mode="clkd",
        corpus_file=args.corpus_file,
        epochs=cfg.get("epochs", 10),
        batch_size=args.batch_size,
        learning_rate=cfg.get("learning_rate", 5e-6),
        temperature=cfg.get("temperature", 2.0),
        freeze_n_layers=cfg.get("freeze_n_layers", 0),
        max_length=args.max_length,
        output_dir=output_dir,
        wandb_project=WANDB_PROJECT,
    )


def run_calibrate(args: argparse.Namespace) -> None:
    from funcs import calibrate
    wandb.init(project=WANDB_PROJECT)
    cfg = wandb.config
    output_dir = f"{args.output_dir}_{wandb.run.id}"
    calibrate(
        checkpoint=args.clkd_ckpt,
        output_dir=output_dir,
        annot_file=args.annot_file,
        epochs=cfg.get("epochs", 10),
        batch_size=args.batch_size,
        learning_rate=cfg.get("learning_rate", 5e-6),
        literal_ratio=cfg.get("literal_ratio", 3),
        max_length=args.max_length,
        wandb_project=WANDB_PROJECT,
    )


def run_pipeline(args: argparse.Namespace) -> None:
    """Joint CLKD → Calibrate sweep trial; swept jointly because freeze_n_layers couples both stages."""
    import shutil
    from funcs import figurative_distill, calibrate

    wandb.init(project=WANDB_PROJECT)
    cfg = wandb.config
    run_id = wandb.run.id

    clkd_dir = f"{args.output_dir}_clkd_{run_id}"
    cal_dir  = f"{args.output_dir}_cal_{run_id}"

    try:
        figurative_distill(
            checkpoint=args.tlm_ckpt,
            teacher_checkpoint=args.teacher,
            mode="clkd",
            corpus_file=args.corpus_file,
            epochs=int(cfg.get("clkd_epochs", 5)),
            batch_size=args.batch_size,
            learning_rate=float(cfg.get("clkd_lr", 5e-6)),
            temperature=float(cfg.get("clkd_temperature", 4.0)),
            freeze_n_layers=int(cfg.get("clkd_freeze_layers", 6)),
            max_length=args.max_length,
            output_dir=clkd_dir,
            wandb_project=WANDB_PROJECT,
        )
        calibrate(
            checkpoint=clkd_dir,
            output_dir=cal_dir,
            annot_file=args.annot_file,
            epochs=int(cfg.get("calibrate_epochs", 15)),
            batch_size=args.batch_size,
            learning_rate=float(cfg.get("calibrate_lr", 5e-6)),
            literal_ratio=int(cfg.get("calibrate_literal_ratio", 3)),
            max_length=args.max_length,
            wandb_project=WANDB_PROJECT,
        )
    finally:
        shutil.rmtree(clkd_dir, ignore_errors=True)
        shutil.rmtree(cal_dir, ignore_errors=True)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="One sweep trial for a single pipeline stage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--stage", required=True, choices=["tlm", "clkd", "calibrate", "pipeline"])

    # Input checkpoints (fixed across all trials for a given sweep)
    p.add_argument("--base-model",   default="FacebookAI/xlm-mlm-100-1280",
                   help="Base model for TLM stage")
    p.add_argument("--tlm-ckpt",     default=None,
                   help="TLM checkpoint for CLKD stage")
    p.add_argument("--clkd-ckpt",    default=None,
                   help="CLKD checkpoint for calibrate stage")

    # Fixed (non-swept) trial params
    p.add_argument("--output-dir",     default="data/sweep_trial")
    p.add_argument("--sentences-file", default=SENTENCES_FILE)
    p.add_argument("--corpus-file",    default=CORPUS_FILE)
    p.add_argument("--annot-file",     default=ANNOT_FILE)
    p.add_argument("--teacher",        default=TEACHER)
    p.add_argument("--batch-size",     type=int,  default=16)
    p.add_argument("--max-length",     type=int,  default=256)

    args = p.parse_args()

    if args.stage == "tlm":
        run_tlm(args)
    elif args.stage == "clkd":
        if not args.tlm_ckpt:
            p.error("--tlm-ckpt is required for --stage clkd")
        run_clkd(args)
    elif args.stage == "calibrate":
        if not args.clkd_ckpt:
            p.error("--clkd-ckpt is required for --stage calibrate")
        run_calibrate(args)
    elif args.stage == "pipeline":
        if not args.tlm_ckpt:
            p.error("--tlm-ckpt is required for --stage pipeline")
        run_pipeline(args)


if __name__ == "__main__":
    main()
