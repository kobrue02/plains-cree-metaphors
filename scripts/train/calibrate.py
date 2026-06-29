"""Standalone entry point for the calibration stage. See pipeline.py for the full run."""

import os, sys, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from funcs import calibrate

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",    required=True)
    p.add_argument("--output-dir",    default="data/calibrated")
    p.add_argument("--hub-model-id",  default=None)
    p.add_argument("--annot-file",    default="data/figurative/annotations.parquet")
    p.add_argument("--epochs",        type=int,   default=10)
    p.add_argument("--batch-size",    type=int,   default=8)
    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--max-length",    type=int,   default=128)
    p.add_argument("--gold-only",     action="store_true")
    p.add_argument("--wandb-project", default=None)
    args = p.parse_args()

    calibrate(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        hub_model_id=args.hub_model_id,
        annot_file=args.annot_file,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        gold_only=args.gold_only,
        wandb_project=args.wandb_project,
    )
