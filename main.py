import argparse
import os

import pandas as pd
from tqdm import tqdm

from src.scrapers import BloomfieldScraper, EdTeKLAScraper
from src.eda import EDA
from src.annotate import call_deepseek, format_prompt
from src.mt import TLMFinetuner, TLMConfig, ParallelSentenceSplitter
from src.figurative import config as figurative_config
from src.figurative.train import train as figurative_train_fn
from src.figurative.distill import distill as figurative_distill_fn, DistillConfig
from src.figurative.predict import (
    load_model as figurative_load_model,
    predict_sentences,
    predict_idioms,
    eval_idioms,
)

FIGURATIVE_PRESETS = {
    "tlm_base":        figurative_config.tlm_base,
    "tlm_large":       figurative_config.tlm_large,
    "tlm_xlm":         figurative_config.tlm_xlm,
    "deberta_teacher": figurative_config.deberta_teacher,
    "baseline":        figurative_config.baseline,
}


# ── pipeline steps ────────────────────────────────────────────────────────────

def scrape() -> pd.DataFrame:
    df = BloomfieldScraper().scrape(output="data/bloomfield_texts.csv")
    return df


def scrape_edtekla(output: str, append: bool = False) -> None:
    EdTeKLAScraper().scrape(output=output, append=append)


def eda(df: pd.DataFrame | None = None) -> None:
    if df is None:
        df = pd.read_csv("data/bloomfield_texts.csv", encoding="utf-8-sig")
    e = EDA(df)
    summary, figures = e.run()
    print(summary)
    os.makedirs("figures", exist_ok=True)
    for i, fig in enumerate(figures, 1):
        fig.savefig(f"figures/figure_{i}.png", dpi=300)
    e.to_tikz(figures, "figures")


def annotate(df: pd.DataFrame | None = None) -> pd.DataFrame:
    path = "data/bloomfield_texts_annotated.csv"
    if df is None:
        src = path if os.path.exists(path) else "data/bloomfield_texts.csv"
        df  = pd.read_csv(src, encoding="utf-8-sig")
    for col in ("annotation", "reasoning"):
        if col not in df.columns:
            df[col] = pd.NA
    pending   = df[df["annotation"].isna() | df["reasoning"].isna()]
    completed = len(df) - len(pending)
    for i, row in tqdm(pending.iterrows(), total=len(df), initial=completed):
        prompt = format_prompt(row["text_cree"], row["text_en"], row["footnote_en"])
        try:
            reasoning, label = call_deepseek(prompt)
            tqdm.write(f"Paragraph {i}: {label} ({reasoning})")
            df.at[i, "annotation"] = label
            df.at[i, "reasoning"]  = reasoning
        except (Exception, KeyboardInterrupt) as e:
            tqdm.write(f"Error at paragraph {i}: {e}")
            break
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Saved → {path}")
    return df


def fine_tune(
    confidence:     float = 0.0,
    sentences_file: str | None = None,
    epochs:         int  = 5,
    batch_size:     int  = 16,
    hub_model_id:   str | None = None,
    wandb_project:  str | None = None,
    model_name:     str  = "xlm-roberta-base",
    output_dir:     str  = "data/tlm_model",
    grad_accum:     int  = 2,
    max_length:     int  = 256,
) -> str:
    if sentences_file:
        sent_df = pd.read_csv(
            sentences_file,
            sep=r"\s*\|\|\|\s*",
            header=None,
            names=["text_cree", "text_en"],
            engine="python",
        )
        print(f"Loaded {len(sent_df):,} pairs from {sentences_file}")
    else:
        df      = pd.read_csv("data/bloomfield_texts.csv", encoding="utf-8-sig")
        sent_df = ParallelSentenceSplitter(df).split()
        if confidence > 0:
            sent_df = sent_df[sent_df.confidence >= confidence]
            print(f"Kept {len(sent_df):,} pairs with confidence ≥ {confidence}")
    cfg  = TLMConfig(model_name=model_name, output_dir=output_dir, epochs=epochs, batch_size=batch_size, grad_accum=grad_accum, max_length=max_length, hub_model_id=hub_model_id, wandb_project=wandb_project)
    ckpt = TLMFinetuner(cfg).fit(sent_df)
    return ckpt


def figurative_train(
    experiment:     str         = "tlm_base",
    epochs:         int | None  = None,
    batch_size:     int | None  = None,
    learning_rate:  float | None = None,
    hub_model_id:   str | None  = None,
    wandb_project:  str | None  = None,
    encoder:        str | None  = None,
    freeze_encoder: bool        = False,
) -> str:
    preset_fn = FIGURATIVE_PRESETS.get(experiment)
    if preset_fn is None:
        raise ValueError(f"Unknown experiment '{experiment}'. Choose from: {list(FIGURATIVE_PRESETS)}")
    cfg = preset_fn()
    if encoder        is not None: cfg.encoder        = encoder
    if freeze_encoder:             cfg.freeze_encoder = True
    if epochs         is not None: cfg.epochs         = epochs
    if batch_size     is not None: cfg.batch_size      = batch_size
    if learning_rate  is not None: cfg.learning_rate   = learning_rate
    if hub_model_id   is not None: cfg.hub_model_id    = hub_model_id
    if wandb_project  is not None: cfg.wandb_project   = wandb_project
    return figurative_train_fn(cfg)


def figurative_distill(
    checkpoint:          str        = "KonradBRG/xlm-mlm-100-1280-plains-cree-en-tlm",
    teacher_checkpoint:  str | None = None,
    freeze_n_layers:     int        = 0,
    mode:                str        = "clkd",
    corpus_file:         str        = "data/bloomfield_texts_sentences.csv",
    epochs:              int        = 10,
    batch_size:          int        = 16,
    learning_rate:       float      = 5e-6,
    temperature:         float      = 2.0,
    hub_model_id:        str | None = None,
    wandb_project:       str | None = None,
    output_dir:          str        = "data/figurative/distilled",
) -> str:
    cfg = DistillConfig(
        checkpoint=checkpoint,
        teacher_checkpoint=teacher_checkpoint,
        freeze_n_layers=freeze_n_layers,
        corpus_file=corpus_file,
        mode=mode,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        temperature=temperature,
        hub_model_id=hub_model_id,
        output_dir=output_dir,
        wandb_project=wandb_project,
    )
    return figurative_distill_fn(cfg)


def figurative_eval_idioms(
    checkpoint:  str = "KonradBRG/xlm-r-plains-cree-en-tlm-figurative",
    idioms_file: str = "data/idioms.txt",
) -> None:
    slug = checkpoint.replace("/", "_").replace("\\", "_")
    output_file = f"data/figurative/idiom_eval_{slug}.csv"
    os.makedirs("data/figurative", exist_ok=True)
    model, tokenizer = figurative_load_model(checkpoint)
    result = eval_idioms(idioms_file, model, tokenizer)
    result["detail"].to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"Saved idiom evaluation to {output_file}")


def predict_figurative(
    checkpoint:  str = "KonradBRG/xlm-r-plains-cree-en-tlm-figurative",
    input_file:  str = "data/bloomfield_texts_sentences.csv",
    output_file: str = "data/bloomfield_figurative.csv",
    idioms_file: str = "data/idioms.txt",
) -> pd.DataFrame:
    sentences_df = pd.read_csv(input_file, encoding="utf-8-sig")
    print(f"Loaded {len(sentences_df):,} sentences from {input_file}")

    model, tokenizer = figurative_load_model(checkpoint)
    print(f"Loaded model from {checkpoint}")

    preds = predict_sentences(sentences_df["text_cree"].tolist(), model, tokenizer)
    pred_df = pd.DataFrame(preds)
    pred_df["paragraph_id"] = sentences_df["paragraph_id"].values
    pred_df["sentence_id"]  = sentences_df["sentence_id"].values
    pred_df["text_en"]      = sentences_df["text_en"].values

    from src.figurative.data import LABEL_NAMES as _fig_labels
    prob_cols = [f"prob_{n}" for n in _fig_labels]
    cols = ["paragraph_id", "sentence_id", "text", "text_en",
            "label", "confidence"] + prob_cols
    pred_df[cols].to_csv(output_file, index=False, encoding="utf-8-sig")

    counts = pred_df["label"].value_counts()
    print(f"Saved {len(pred_df):,} predictions to {output_file}")
    print(f"  literal={counts.get('literal',0):,}  "
          f"idiom={counts.get('idiom',0):,}  "
          f"metaphor={counts.get('metaphor',0):,}")

    if os.path.exists(idioms_file):
        print(f"\nEvaluating idiom transfer on {idioms_file} ...")
        idiom_df = predict_idioms(idioms_file, model, tokenizer)
        print(idiom_df.to_string(index=False))
        idiom_out = output_file.replace(".csv", "_idioms.csv")
        idiom_df.to_csv(idiom_out, index=False, encoding="utf-8-sig")
        print(f"Saved idiom comparison to {idiom_out}")

    return pred_df


def split_sentences(output: str, confidence: float = 0.0) -> None:
    df = pd.read_csv("data/bloomfield_texts.csv", encoding="utf-8-sig")
    splitter = ParallelSentenceSplitter(df)
    splitter.write(output, min_confidence=confidence)
    sent_df = splitter.split()
    if confidence > 0:
        sent_df = sent_df[sent_df.confidence >= confidence]
    csv_path = "data/bloomfield_texts_sentences.csv"
    sent_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"Saved {len(sent_df):,} sentence pairs → {csv_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bloomfield Plains Cree FNLP pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python main.py --scrape
  python main.py --annotate
  python main.py --split-sentences --output data/sentences.txt --confidence 0.6
  python main.py --fine-tune --confidence 0.6
  python main.py --eda
""",
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    parser.add_argument("--scrape",          action="store_true",
                        help="Scrape Bloomfield texts to data/bloomfield_texts.csv")
    parser.add_argument("--scrape-edtekla",  action="store_true",
                        help="Download EdTeKLA parallel corpus and write src ||| tgt pairs")
    parser.add_argument("--edtekla-output",  default="data/sentences_edtekla.txt",
                        help="Output path for --scrape-edtekla (default: data/sentences_edtekla.txt)")
    parser.add_argument("--edtekla-append",  action="store_true",
                        help="Append EdTeKLA pairs to an existing file instead of overwriting")
    parser.add_argument("--eda",             action="store_true",
                        help="Run EDA and save figures to figures/")
    parser.add_argument("--annotate",        action="store_true",
                        help="Run LLM annotation on the Bloomfield paragraphs")
    parser.add_argument("--split-sentences", action="store_true",
                        help="Split paragraphs into sentence pairs and write src ||| tgt")
    parser.add_argument("--output",          default="data/sentences.txt",
                        help="Output path for --split-sentences (default: data/sentences.txt)")
    parser.add_argument("--confidence",      type=float, default=0.0,
                        help="Minimum alignment confidence filter for sentence pairs (0–1)")

    # ── TLM ───────────────────────────────────────────────────────────────────
    parser.add_argument("--fine-tune",       action="store_true",
                        help="TLM fine-tune a model on Cree-English sentence pairs")
    parser.add_argument("--sentences-file",  default=None,
                        help="Pre-split sentences file (cree ||| en) to use instead of the scraper pipeline")
    parser.add_argument("--model-name",      default="xlm-roberta-base",
                        help="Base model for --fine-tune (default: xlm-roberta-base)")
    parser.add_argument("--tlm-output-dir",  default="data/tlm_model",
                        help="Output directory for --fine-tune (default: data/tlm_model)")
    parser.add_argument("--epochs",          type=int,   default=5,
                        help="Number of training epochs (default: 5)")
    parser.add_argument("--batch-size",      type=int,   default=16,
                        help="Per-device train batch size (default: 16)")
    parser.add_argument("--grad-accum",      type=int,   default=2,
                        help="Gradient accumulation steps for --fine-tune (default: 2)")
    parser.add_argument("--max-length",      type=int,   default=256,
                        help="Max token length per TLM input pair (default: 256)")

    # ── Shared training args ───────────────────────────────────────────────────
    parser.add_argument("--hub-model-id",    default=None,
                        help="HuggingFace Hub repo ID to push the final model to")
    parser.add_argument("--wandb-project",   default=None,
                        help="Weights & Biases project name for logging (omit to disable)")
    parser.add_argument("--learning-rate",   type=float, default=None,
                        help="Learning rate override (default: preset value)")
    parser.add_argument("--encoder",         default=None,
                        help="Encoder checkpoint override for figurative training")
    parser.add_argument("--freeze-encoder",  action="store_true",
                        help="Freeze the encoder and only train the classification head (probing)")

    # ── Figurative detection ──────────────────────────────────────────────────
    parser.add_argument("--train-figurative",    action="store_true",
                        help="Train figurative classifier on VUA20 + MAGPIE + FLUTE")
    parser.add_argument("--figurative-experiment", default="tlm_base",
                        choices=list(FIGURATIVE_PRESETS),
                        help="Figurative experiment preset (default: tlm_base)")
    parser.add_argument("--predict-figurative",  action="store_true",
                        help="Run figurative detection on Bloomfield Cree sentences + idioms.txt")
    parser.add_argument("--figurative-checkpoint", default="KonradBRG/xlm-r-plains-cree-en-tlm-figurative",
                        help="Checkpoint for --predict-figurative / --eval-idioms")
    parser.add_argument("--figurative-output",   default="data/bloomfield_figurative.csv",
                        help="Output CSV for --predict-figurative")
    parser.add_argument("--input-file",          default="data/bloomfield_texts_sentences.csv",
                        help="Input CSV for --predict-figurative")

    # ── CLKD distillation ─────────────────────────────────────────────────────
    parser.add_argument("--distill-figurative",  action="store_true",
                        help="Cross-lingual knowledge distillation on the parallel corpus")
    parser.add_argument("--distill-mode",        default="clkd", choices=["align", "binary_kl", "clkd"],
                        help="Distillation mode (default: clkd)")
    parser.add_argument("--distill-checkpoint",  default="KonradBRG/xlm-mlm-100-1280-plains-cree-en-tlm",
                        help="Student model checkpoint")
    parser.add_argument("--distill-teacher",     default=None,
                        help="Frozen English teacher checkpoint for clkd mode")
    parser.add_argument("--distill-freeze-layers", type=int, default=0,
                        help="Freeze first N student layers during CLKD (0 = train all)")
    parser.add_argument("--distill-output",      default="data/figurative/distilled",
                        help="Output directory for distilled model")
    parser.add_argument("--distill-temperature", type=float, default=2.0,
                        help="Softening temperature for KL divergence (default: 2.0)")

    # ── Evaluation ────────────────────────────────────────────────────────────
    parser.add_argument("--eval-idioms",     action="store_true",
                        help="Evaluate a figurative model on the Cree idiom golden test set")
    parser.add_argument("--idioms-file",     default="data/idioms.txt",
                        help="Path to the cree ||| english idioms file (default: data/idioms.txt)")

    args = parser.parse_args()

    if not any([args.scrape, args.scrape_edtekla, args.eda, args.annotate,
                args.split_sentences, args.fine_tune,
                args.train_figurative, args.predict_figurative,
                args.distill_figurative, args.eval_idioms]):
        parser.print_help()
        return

    if args.scrape:
        scrape()

    if args.scrape_edtekla:
        scrape_edtekla(args.edtekla_output, append=args.edtekla_append)

    if args.eda:
        eda()

    if args.annotate:
        annotate()

    if args.split_sentences:
        split_sentences(args.output, args.confidence)

    if args.fine_tune:
        ckpt = fine_tune(
            args.confidence, args.sentences_file, args.epochs, args.batch_size,
            args.hub_model_id, args.wandb_project, args.model_name,
            args.tlm_output_dir, args.grad_accum, args.max_length,
        )
        print(f"Checkpoint: {ckpt}")

    if args.train_figurative:
        ckpt = figurative_train(
            experiment     = args.figurative_experiment,
            epochs         = args.epochs,
            batch_size     = args.batch_size,
            learning_rate  = args.learning_rate,
            hub_model_id   = args.hub_model_id,
            wandb_project  = args.wandb_project,
            encoder        = args.encoder,
            freeze_encoder = args.freeze_encoder,
        )
        print(f"Checkpoint: {ckpt}")

    if args.predict_figurative:
        predict_figurative(
            checkpoint  = args.figurative_checkpoint,
            input_file  = args.input_file,
            output_file = args.figurative_output,
        )

    if args.distill_figurative:
        ckpt = figurative_distill(
            checkpoint         = args.distill_checkpoint,
            teacher_checkpoint = args.distill_teacher,
            freeze_n_layers    = args.distill_freeze_layers,
            mode               = args.distill_mode,
            corpus_file        = args.input_file,
            epochs             = args.epochs,
            batch_size         = args.batch_size,
            learning_rate      = args.learning_rate if args.learning_rate else 5e-6,
            temperature        = args.distill_temperature,
            hub_model_id       = args.hub_model_id,
            wandb_project      = args.wandb_project,
            output_dir         = args.distill_output,
        )
        print(f"Distilled model saved to: {ckpt}")

    if args.eval_idioms:
        figurative_eval_idioms(
            checkpoint  = args.figurative_checkpoint,
            idioms_file = args.idioms_file,
        )


if __name__ == "__main__":
    main()
