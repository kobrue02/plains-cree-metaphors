"""
CLI wrapper around funcs.figurative_train — fine-tune a sequence classifier on
VUA20+MAGPIE+FLUTE (English) and push it to the Hub. Used for the "no
adaptation" and "+TLM" rows of the classifier results table: both are
English-fine-tuned-then-zero-shot-transferred-to-Cree, differing only in
whether the encoder went through TLM first. Works with any multilingual
encoder — pass --encoder/--hub-model-id to point at the one you want.

Usage:
  # "no adaptation" row for XLM-MLM-100-1280 (the currently missing piece —
  # its +TLM/+TLM+CLKD checkpoints already exist, see scripts/evals/figurative_results_table.py)
  python scripts/train/train_figurative_english.py --experiment baseline \
      --encoder FacebookAI/xlm-mlm-100-1280 \
      --hub-model-id KonradBRG/xlm-mlm-100-1280-figurative

  # same pattern for a different encoder entirely, e.g. a fresh TLM checkpoint
  # you haven't fine-tuned on English data yet:
  python scripts/train/train_figurative_english.py --experiment baseline \
      --encoder <hf-id-or-local-path> --hub-model-id <KonradBRG/your-name-figurative>
"""

from __future__ import annotations
import argparse, os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from funcs import figurative_train, FIGURATIVE_PRESETS


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", required=True, choices=list(FIGURATIVE_PRESETS),
                   help="Preset from src/figurative/config.py")
    p.add_argument("--encoder",        default=None, help="Override the preset's encoder")
    p.add_argument("--epochs",         type=int,   default=None)
    p.add_argument("--batch-size",     type=int,   default=None)
    p.add_argument("--learning-rate",  type=float, default=None)
    p.add_argument("--hub-model-id",   default=None, help="Override the preset's hub_model_id")
    p.add_argument("--wandb-project",  default=None)
    p.add_argument("--freeze-encoder", action="store_true")
    args = p.parse_args()

    ckpt = figurative_train(
        experiment=args.experiment,
        encoder=args.encoder,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hub_model_id=args.hub_model_id,
        wandb_project=args.wandb_project,
        freeze_encoder=args.freeze_encoder,
    )
    print(f"\nSaved → {ckpt}")


if __name__ == "__main__":
    main()
