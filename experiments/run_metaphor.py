"""
CLI for running and comparing metaphor detection experiments.

Subcommands
-----------
train    Fine-tune an encoder on VUA20.
eval     Evaluate a saved checkpoint on the VUA20 test set.
predict  Run zero-shot inference on arbitrary sentences (English or Cree).
compare  Print a comparison table of all evaluated experiments.

Usage examples
--------------
# Train the baseline (base XLM-R):
    uv run python experiments/run_metaphor.py train --experiment baseline

# Train with awesome-align encoder:
    uv run python experiments/run_metaphor.py train --experiment awesome_align

# Train with content-words-only loss (mirrors MIPVU scope):
    uv run python experiments/run_metaphor.py train --experiment content_words

# Override any config field on the fly:
    uv run python experiments/run_metaphor.py train --experiment baseline --epochs 3 --batch_size 32

# Evaluate a saved checkpoint on the VUA20 test set:
    uv run python experiments/run_metaphor.py eval --checkpoint data/metaphor/baseline_xlmr

# Run zero-shot inference on Cree sentences:
    uv run python experiments/run_metaphor.py predict \\
        --checkpoint data/metaphor/awesome_align_encoder \\
        --sentences "kî-pîkiskwêw" "iskwêw ê-wîcihât"

# Compare all evaluated experiments:
    uv run python experiments/run_metaphor.py compare
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.metaphor.config import (
    ExperimentConfig,
    awesome_align_content_words,
    awesome_align_encoder,
    baseline,
    content_words_only,
)
from src.metaphor.train import train
from src.metaphor.predict import load_model, predict_df
from src.metaphor.evaluate import evaluate
from src.metaphor.data import load_vua20_sentences


PRESETS = {
    "baseline":       baseline,
    "awesome_align":  awesome_align_encoder,
    "content_words":  content_words_only,
    "awesome_align_content": awesome_align_content_words,
}


def apply_overrides(config: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    """Apply CLI field overrides to a config instance."""
    for field in ("epochs", "batch_size", "learning_rate", "max_length",
                  "grad_accum", "encoder", "experiment_name"):
        val = getattr(args, field, None)
        if val is not None:
            setattr(config, field, val)
    if getattr(args, "no_class_weights", False):
        config.class_weights = False
    if getattr(args, "content_words_only", False):
        config.content_words_only = True
    return config


def cmd_train(args: argparse.Namespace) -> None:
    preset_fn = PRESETS.get(args.experiment)
    if preset_fn is None:
        print(f"Unknown experiment '{args.experiment}'. Available: {list(PRESETS)}")
        sys.exit(1)
    config = apply_overrides(preset_fn(), args)
    print(f"\n{'='*60}")
    print(f"Experiment : {config.experiment_name}")
    print(f"Encoder    : {config.encoder}")
    print(f"Output     : {config.checkpoint_dir}")
    print(f"{'='*60}\n")
    train(config)


def cmd_eval(args: argparse.Namespace) -> None:
    checkpoint = args.eval
    model, tokenizer = load_model(checkpoint)

    test_sents = load_vua20_sentences("test")
    sentences  = [s["sentence"] for s in test_sents]
    references = [s["labels"]   for s in test_sents]

    df = predict_df(sentences, model, tokenizer)

    # Reconstruct nested prediction lists aligned to test_sents
    predictions = []
    idx = 0
    for sent in test_sents:
        n = len(sent["words"])
        predictions.append(df.iloc[idx : idx + n]["label"].tolist())
        idx += n

    print(f"\nEvaluation: {checkpoint}")
    metrics = evaluate(predictions, references, print_report=True)

    # Save metrics alongside the checkpoint
    metrics_path = os.path.join(checkpoint, "test_metrics.json")
    # if checkpoint is a hf model, save to the parent directory instead
    if os.path.exists(os.path.join(checkpoint, "pytorch_model.bin")):
        metrics_path = os.path.join(os.path.dirname(checkpoint), "test_metrics.json")
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")


def cmd_predict(args: argparse.Namespace) -> None:
    model, tokenizer = load_model(args.predict)
    sentences = args.sentences
    df = predict_df(sentences, model, tokenizer)
    print(df.to_string(index=False))


def cmd_compare(args: argparse.Namespace) -> None:
    """Print a table comparing test_metrics.json across all saved experiments."""
    root = args.output_root or "data/metaphor"
    rows = []
    for exp_dir in sorted(os.listdir(root)):
        metrics_path = os.path.join(root, exp_dir, "test_metrics.json")
        if not os.path.exists(metrics_path):
            continue
        with open(metrics_path) as f:
            m = json.load(f)
        rows.append({"experiment": exp_dir, **m})

    if not rows:
        print("No evaluated experiments found. Run --eval first.")
        return

    header = f"{'Experiment':<35} {'P':>6} {'R':>6} {'F1':>6} {'macro_F1':>9}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['experiment']:<35} "
            f"{r.get('metaphor_precision', 0):>6.3f} "
            f"{r.get('metaphor_recall',    0):>6.3f} "
            f"{r.get('metaphor_f1',        0):>6.3f} "
            f"{r.get('macro_f1',           0):>9.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Metaphor detection experiments")
    sub = parser.add_subparsers(dest="command")

    # ── train ────────────────────────────────────────────────────────────────
    p_train = sub.add_parser("train", help="Fine-tune a model")
    p_train.add_argument("--experiment", required=True, choices=list(PRESETS))
    p_train.add_argument("--encoder")
    p_train.add_argument("--epochs",        type=int)
    p_train.add_argument("--batch_size",    type=int)
    p_train.add_argument("--learning_rate", type=float)
    p_train.add_argument("--max_length",    type=int)
    p_train.add_argument("--grad_accum",    type=int)
    p_train.add_argument("--no_class_weights", action="store_true")
    p_train.add_argument("--content_words_only", action="store_true")

    # ── eval ─────────────────────────────────────────────────────────────────
    p_eval = sub.add_parser("eval", help="Evaluate a saved checkpoint on VUA20 test")
    p_eval.add_argument("--checkpoint", dest="eval", required=True)

    # ── predict ──────────────────────────────────────────────────────────────
    p_pred = sub.add_parser("predict", help="Run inference on raw sentences")
    p_pred.add_argument("--checkpoint", dest="predict", required=True)
    p_pred.add_argument("--sentences", nargs="+", required=True)

    # ── compare ──────────────────────────────────────────────────────────────
    p_cmp = sub.add_parser("compare", help="Compare all evaluated experiments")
    p_cmp.add_argument("--output_root", default="data/metaphor")

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "predict":
        cmd_predict(args)
    elif args.command == "compare":
        cmd_compare(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
