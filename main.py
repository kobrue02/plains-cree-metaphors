import argparse
import os

import pandas as pd
from tqdm import tqdm

from src.scrapers import BloomfieldScraper
from src.eda import EDA
from src.annotate import call_deepseek, format_prompt
from src.mt import TLMFinetuner, TLMConfig, ParallelSentenceSplitter
from src.metaphor import config as metaphor_config
from src.metaphor.train import train as metaphor_train_fn

METAPHOR_PRESETS = {
    "baseline":              metaphor_config.baseline,
    "tlm_last_layer":        metaphor_config.tlm_last_layer,
    "tlm_layer_12":          metaphor_config.tlm_layer_12,
    "awesome_align":         metaphor_config.awesome_align_encoder,
    "content_words":         metaphor_config.content_words_only,
    "awesome_align_content": metaphor_config.awesome_align_content_words,
}


# ── pipeline steps ────────────────────────────────────────────────────────────

def scrape() -> pd.DataFrame:
    df = BloomfieldScraper().scrape(output="data/bloomfield_texts.csv")
    return df


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
    cfg  = TLMConfig(model_name=model_name, epochs=epochs, batch_size=batch_size, hub_model_id=hub_model_id, wandb_project=wandb_project)
    ckpt = TLMFinetuner(cfg).fit(sent_df)
    return ckpt


def metaphor_train(
    experiment:    str         = "tlm_last_layer",
    epochs:        int | None  = None,
    batch_size:    int | None  = None,
    learning_rate: float | None = None,
    hub_model_id:  str | None  = None,
    wandb_project: str | None  = None,
    encoder:       str | None  = None,
) -> str:
    preset_fn = METAPHOR_PRESETS.get(experiment)
    if preset_fn is None:
        raise ValueError(f"Unknown experiment '{experiment}'. Choose from: {list(METAPHOR_PRESETS)}")
    cfg = preset_fn()
    if encoder       is not None: cfg.encoder        = encoder
    if epochs        is not None: cfg.epochs         = epochs
    if batch_size    is not None: cfg.batch_size      = batch_size
    if learning_rate is not None: cfg.learning_rate   = learning_rate
    if hub_model_id  is not None: cfg.hub_model_id    = hub_model_id
    if wandb_project is not None: cfg.wandb_project   = wandb_project
    return metaphor_train_fn(cfg)


def split_sentences(output: str, confidence: float = 0.0) -> None:
    df = pd.read_csv("data/bloomfield_texts.csv", encoding="utf-8-sig")
    ParallelSentenceSplitter(df).write(output, min_confidence=confidence)


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

    parser.add_argument("--scrape",          action="store_true",
                        help="Scrape Bloomfield texts to data/bloomfield_texts.csv")
    parser.add_argument("--eda",             action="store_true",
                        help="Run EDA and save figures to figures/")
    parser.add_argument("--annotate",        action="store_true",
                        help="Run LLM annotation on the Bloomfield paragraphs")
    parser.add_argument("--split-sentences", action="store_true",
                        help="Split paragraphs into sentence pairs and write src ||| tgt")
    parser.add_argument("--fine-tune",       action="store_true",
                        help="TLM fine-tune XLM-R on Cree-English sentence pairs")
    parser.add_argument("--metaphor",        action="store_true",
                        help="Fine-tune a metaphor detector on VUA20")
    parser.add_argument("--output",          default="data/sentences.txt",
                        help="Output path for --split-sentences (default: data/sentences.txt)")
    parser.add_argument("--confidence",      type=float, default=0.0,
                        help="Minimum alignment confidence filter for sentence pairs (0–1)")
    parser.add_argument("--sentences-file", default=None,
                        help="Pre-split sentences file (cree ||| en) to use instead of the scraper pipeline")
    parser.add_argument("--epochs",         type=int,   default=5,
                        help="Number of TLM training epochs (default: 5)")
    parser.add_argument("--batch-size",     type=int,   default=16,
                        help="Per-device train batch size (default: 16)")
    parser.add_argument("--hub-model-id",   default=None,
                        help="HuggingFace Hub repo ID to push the final model to (e.g. YourName/model-tag)")
    parser.add_argument("--wandb-project",   default=None,
                        help="Weights & Biases project name for logging (omit to disable)")
    parser.add_argument("--experiment",      default="tlm_last_layer",
                        choices=list(METAPHOR_PRESETS),
                        help="Metaphor experiment preset (default: tlm_last_layer)")
    parser.add_argument("--learning-rate",   type=float, default=None,
                        help="Learning rate override for --metaphor (default: preset value)")
    parser.add_argument("--encoder",         default=None,
                        help="Encoder checkpoint for --metaphor, overrides the preset (e.g. KonradBRG/xlm-r-large-plains-cree-en-tlm)")
    parser.add_argument("--model-name",      default="xlm-roberta-base",
                        help="Base model for --fine-tune (default: xlm-roberta-base)")

    args = parser.parse_args()

    if not any([args.scrape, args.eda, args.annotate,
                args.split_sentences, args.fine_tune, args.metaphor]):
        parser.print_help()
        return

    if args.scrape:
        scrape()

    if args.eda:
        eda()

    if args.annotate:
        annotate()

    if args.split_sentences:
        split_sentences(args.output, args.confidence)

    if args.fine_tune:
        ckpt = fine_tune(args.confidence, args.sentences_file, args.epochs, args.batch_size, args.hub_model_id, args.wandb_project, args.model_name)
        print(f"Checkpoint: {ckpt}")

    if args.metaphor:
        ckpt = metaphor_train(
            experiment    = args.experiment,
            epochs        = args.epochs,
            batch_size    = args.batch_size,
            learning_rate = args.learning_rate,
            hub_model_id  = args.hub_model_id,
            wandb_project = args.wandb_project,
            encoder       = args.encoder,
        )
        print(f"Checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
