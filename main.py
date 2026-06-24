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
from src.metaphor.predict import load_model, predict_df

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
    output_dir:     str  = "data/tlm_model",
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
    cfg  = TLMConfig(model_name=model_name, output_dir=output_dir, epochs=epochs, batch_size=batch_size, hub_model_id=hub_model_id, wandb_project=wandb_project)
    ckpt = TLMFinetuner(cfg).fit(sent_df)
    return ckpt


def metaphor_train(
    experiment:     str         = "tlm_last_layer",
    epochs:         int | None  = None,
    batch_size:     int | None  = None,
    learning_rate:  float | None = None,
    hub_model_id:   str | None  = None,
    wandb_project:  str | None  = None,
    encoder:        str | None  = None,
    freeze_encoder: bool        = False,
) -> str:
    preset_fn = METAPHOR_PRESETS.get(experiment)
    if preset_fn is None:
        raise ValueError(f"Unknown experiment '{experiment}'. Choose from: {list(METAPHOR_PRESETS)}")
    cfg = preset_fn()
    if encoder       is not None: cfg.encoder        = encoder
    if freeze_encoder:            cfg.freeze_encoder = True
    if epochs        is not None: cfg.epochs         = epochs
    if batch_size    is not None: cfg.batch_size      = batch_size
    if learning_rate is not None: cfg.learning_rate   = learning_rate
    if hub_model_id  is not None: cfg.hub_model_id    = hub_model_id
    if wandb_project is not None: cfg.wandb_project   = wandb_project
    return metaphor_train_fn(cfg)


def predict_cree(
    checkpoint: str = "KonradBRG/xlm-r-plains-cree-en-tlm-metaphor-layer12",
    input_file: str = "data/bloomfield_texts_sentences.csv",
    output_file: str = "data/bloomfield_metaphors.csv",
    min_confidence: float = 0.0,
) -> pd.DataFrame:
    sentences_df = pd.read_csv(input_file, encoding="utf-8-sig")
    print(f"Loaded {len(sentences_df):,} sentences from {input_file}")

    model, tokenizer = load_model(checkpoint)
    print(f"Loaded model from {checkpoint}")

    preds = predict_df(sentences_df["text_cree"].tolist(), model, tokenizer)

    # preds has sentence_idx, word, label, confidence — join back with metadata
    meta = sentences_df[["paragraph_id", "sentence_id", "text_cree", "text_en"]].reset_index(drop=True)
    preds["paragraph_id"] = preds["sentence_idx"].map(meta["paragraph_id"])
    preds["sentence_id"]  = preds["sentence_idx"].map(meta["sentence_id"])
    preds["text_cree"]    = preds["sentence_idx"].map(meta["text_cree"])
    preds["text_en"]      = preds["sentence_idx"].map(meta["text_en"])

    cols = ["paragraph_id", "sentence_id", "text_cree", "text_en", "word", "label", "confidence", "metaphor_prob"]
    result = preds[cols]

    if min_confidence > 0:
        result = result[result["confidence"] >= min_confidence]

    result.to_csv(output_file, index=False, encoding="utf-8-sig")
    metaphors = result[result["label"] == 1]
    print(f"Saved {len(result):,} token predictions to {output_file}")
    print(f"Metaphors found: {len(metaphors):,} / {len(result):,} tokens "
          f"({100 * len(metaphors) / max(len(result), 1):.1f}%)")
    return result


def compare_annotations(
    predictions_file: str = "data/bloomfield_metaphors.csv",
    annotated_file:   str = "data/bloomfield_texts_annotated.csv",
    output_file:      str = "data/bloomfield_comparison.csv",
    figurative_only:  bool = False,
    threshold:        float | None = None,
) -> pd.DataFrame:
    """Compare paragraph-level model predictions against LLM annotations.

    A paragraph is 'predicted metaphor' if any of its tokens has label=1.
    Ground truth is the 'annotation' column in bloomfield_texts_annotated.csv.
    By default only 'metaphor' counts as positive; pass figurative_only=True
    to also include simile/idiom/proverb.
    """
    from sklearn.metrics import classification_report

    preds = pd.read_csv(predictions_file, encoding="utf-8-sig")
    annot = pd.read_csv(annotated_file,   encoding="utf-8-sig")

    # Aggregate token predictions to paragraph level using max metaphor_prob
    if "metaphor_prob" not in preds.columns:
        raise KeyError(
            "'metaphor_prob' column missing from predictions file. "
            "Re-run --predict-cree to regenerate it with the updated predict_df."
        )
    para_scores = preds.groupby("paragraph_id")["metaphor_prob"].max().reset_index()
    para_scores.rename(columns={"metaphor_prob": "max_metaphor_prob"}, inplace=True)

    if threshold is None:
        # Sweep thresholds and report F1 at each
        import numpy as np
        from sklearn.metrics import f1_score
        annot_tmp = annot.reset_index().rename(columns={"index": "paragraph_id"})
        annot_tmp["annotation"] = annot_tmp["annotation"].str.strip()
        positive_labels_tmp = (
            {"metaphor", "simile", "idiom", "proverb"} if figurative_only else {"metaphor"}
        )
        annot_tmp["true_metaphor"] = annot_tmp["annotation"].isin(positive_labels_tmp).astype(int)
        merged_tmp = annot_tmp.merge(para_scores, on="paragraph_id", how="left").fillna(0)

        print(f"\n{'Threshold':>10}  {'Precision':>10}  {'Recall':>10}  {'F1':>10}  {'Predicted+':>10}")
        best_t, best_f1 = 0.5, 0.0
        for t in np.arange(0.05, 0.55, 0.05):
            predicted = (merged_tmp["max_metaphor_prob"] >= t).astype(int)
            from sklearn.metrics import precision_score, recall_score
            p = precision_score(merged_tmp["true_metaphor"], predicted, zero_division=0)
            r = recall_score(merged_tmp["true_metaphor"], predicted, zero_division=0)
            f = f1_score(merged_tmp["true_metaphor"], predicted, zero_division=0)
            n = predicted.sum()
            print(f"{t:>10.2f}  {p:>10.3f}  {r:>10.3f}  {f:>10.3f}  {n:>10}")
            if f > best_f1:
                best_f1, best_t = f, t
        print(f"\nBest threshold: {best_t:.2f}  (F1={best_f1:.3f})")
        threshold = best_t

    para_pred = para_scores.copy()
    para_pred["predicted_metaphor"] = (para_pred["max_metaphor_prob"] >= threshold).astype(int)

    # bloomfield_texts.csv row index == paragraph_id; annotated has same ordering
    # Reset index to get a paragraph_id column for the annotated file
    annot = annot.reset_index().rename(columns={"index": "paragraph_id"})
    annot["annotation"] = annot["annotation"].str.strip()

    positive_labels = (
        {"metaphor", "simile", "idiom", "proverb"} if figurative_only
        else {"metaphor"}
    )
    annot["true_metaphor"] = annot["annotation"].isin(positive_labels).astype(int)

    merged = annot.merge(para_pred, on="paragraph_id", how="left")
    merged["predicted_metaphor"] = merged["predicted_metaphor"].fillna(0).astype(int)

    # Only evaluate paragraphs that appear in both files
    evaluated = merged.dropna(subset=["true_metaphor"])

    print(classification_report(
        evaluated["true_metaphor"],
        evaluated["predicted_metaphor"],
        target_names=["literal", "metaphor"],
        digits=3,
    ))

    out_cols = ["paragraph_id", "source_file", "paragraph_num",
                "text_cree", "text_en", "annotation",
                "true_metaphor", "predicted_metaphor"]
    result = evaluated[out_cols]
    result.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"Saved comparison to {output_file}")
    return result


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
    parser.add_argument("--predict-cree",   action="store_true",
                        help="Run metaphor detection on the Bloomfield Cree sentences")
    parser.add_argument("--compare",        action="store_true",
                        help="Compare token predictions against paragraph-level LLM annotations")
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
    parser.add_argument("--freeze-encoder",  action="store_true",
                        help="Freeze the encoder and only train the classification head (probing)")
    parser.add_argument("--checkpoint",      default="KonradBRG/xlm-r-plains-cree-en-tlm-metaphor-layer12",
                        help="Model checkpoint for --predict-cree (default: KonradBRG/xlm-r-plains-cree-en-tlm-metaphor-layer12)")
    parser.add_argument("--input-file",      default="data/bloomfield_texts_sentences.csv",
                        help="Input CSV for --predict-cree (default: data/bloomfield_texts_sentences.csv)")
    parser.add_argument("--predict-output",  default="data/bloomfield_metaphors.csv",
                        help="Output CSV for --predict-cree (default: data/bloomfield_metaphors.csv)")
    parser.add_argument("--min-confidence",  type=float, default=0.0,
                        help="Only keep predictions above this confidence for --predict-cree")
    parser.add_argument("--figurative",      action="store_true",
                        help="For --compare: count simile/idiom/proverb as positive, not just metaphor")
    parser.add_argument("--threshold",       type=float, default=None,
                        help="For --compare: fixed metaphor_prob threshold (omit to sweep 0.05–0.50 and pick best F1)")
    parser.add_argument("--model-name",      default="xlm-roberta-base",
                        help="Base model for --fine-tune (default: xlm-roberta-base)")
    parser.add_argument("--tlm-output-dir", default="data/tlm_model",
                        help="Output directory for --fine-tune (default: data/tlm_model)")

    args = parser.parse_args()

    if not any([args.scrape, args.eda, args.annotate,
                args.split_sentences, args.fine_tune, args.metaphor,
                args.predict_cree, args.compare]):
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
        ckpt = fine_tune(args.confidence, args.sentences_file, args.epochs, args.batch_size, args.hub_model_id, args.wandb_project, args.model_name, args.tlm_output_dir)
        print(f"Checkpoint: {ckpt}")

    if args.predict_cree:
        predict_cree(args.checkpoint, args.input_file, args.predict_output, args.min_confidence)

    if args.compare:
        compare_annotations(args.predict_output, figurative_only=args.figurative, threshold=args.threshold)

    if args.metaphor:
        ckpt = metaphor_train(
            experiment     = args.experiment,
            epochs         = args.epochs,
            batch_size     = args.batch_size,
            learning_rate  = args.learning_rate,
            hub_model_id   = args.hub_model_id,
            wandb_project  = args.wandb_project,
            encoder        = args.encoder,
            freeze_encoder = args.freeze_encoder,
        )
        print(f"Checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
