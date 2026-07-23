"""CLI wrapper around funcs.train_silver — fine-tune a TLM-adapted encoder directly on the full-pool LLM (Qwen) silver labels, then evaluate on the fixed 228-sentence gold set."""

from __future__ import annotations
import argparse, os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from funcs import train_silver


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint",    required=True, help="TLM-adapted encoder to start from")
    p.add_argument("--silver-file",   default="data/figurative/deepseek_labels_qwen_qwen3.5-122b-a10b.parquet")
    p.add_argument("--label-col",     default="deepseek_label")
    p.add_argument("--output-dir",    default="data/figurative/silver_sft")
    p.add_argument("--hub-model-id",  default=None)
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--epochs",        type=int,   default=5)
    p.add_argument("--batch-size",    type=int,   default=16)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--max-length",    type=int,   default=128)
    p.add_argument("--freeze-n-layers", type=int, default=0,
                    help="Freeze embeddings + first n transformer layers (0 = train all)")
    p.add_argument("--hierarchical", action="store_true",
                    help="Binary (literal/figurative) head + conditional 3-way "
                         "(idiom/metaphor/simile) head instead of a single 4-way softmax")
    args = p.parse_args()

    ckpt = train_silver(
        checkpoint=args.checkpoint,
        silver_file=args.silver_file,
        label_col=args.label_col,
        output_dir=args.output_dir,
        hub_model_id=args.hub_model_id,
        wandb_project=args.wandb_project,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        freeze_n_layers=args.freeze_n_layers,
        hierarchical=args.hierarchical,
    )
    print(f"\nSaved → {ckpt}")


if __name__ == "__main__":
    main()
