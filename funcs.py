import os

import pandas as pd
from tqdm import tqdm

from src.scrapers import BloomfieldScraper, EdTeKLAScraper
from src.eda import EDA
from src.mt import TLMFinetuner, TLMConfig, ParallelSentenceSplitter
from src.figurative import config as figurative_config
from src.figurative.train import train as figurative_train_fn
from src.figurative.distill import distill as figurative_distill_fn, DistillConfig
from src.figurative.calibrate import calibrate as figurative_calibrate_fn, CalibrateConfig
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


def scrape() -> pd.DataFrame:
    """Scrape Bloomfield Plains Cree texts and save to data/bloomfield_texts.csv."""
    df = BloomfieldScraper().scrape(output="data/bloomfield_texts.parquet")
    return df


def scrape_edtekla(output: str, append: bool = False) -> None:
    """Download EdTeKLA parallel corpus and write src ||| tgt pairs to output file."""
    EdTeKLAScraper().scrape(output=output, append=append)


def eda(df: pd.DataFrame | None = None) -> None:
    """Run basic EDA on the Bloomfield texts and save figures to figures/."""
    if df is None:
        df = pd.read_parquet("data/bloomfield_texts.parquet")
    e = EDA(df)
    summary, figures = e.run()
    print(summary)
    os.makedirs("figures", exist_ok=True)
    for i, fig in enumerate(figures, 1):
        fig.savefig(f"figures/figure_{i}.png", dpi=300)
    e.to_tikz(figures, "figures")


def annotate(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Run LLM annotation on the Bloomfield paragraphs and save to data/bloomfield_texts_annotated.csv."""
    from src.annotate import call_deepseek, format_prompt

    path = "data/bloomfield_texts_annotated.parquet"
    if df is None:
        src = path if os.path.exists(path) else "data/bloomfield_texts.parquet"
        df  = pd.read_parquet(src)
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
    df.to_parquet(path, index=False)
    print(f"Saved → {path}")
    return df


def fine_tune(
    confidence:     float = 0.0,
    sentences_file: str | None = None,
    epochs:         int   = 5,
    batch_size:     int   = 16,
    learning_rate:  float = 2e-5,
    hub_model_id:   str | None = None,
    wandb_project:  str | None = None,
    model_name:     str  = "xlm-roberta-base",
    output_dir:     str  = "data/tlm_model",
    grad_accum:     int  = 2,
    max_length:     int  = 256,
    src_col:                str   = "text_cree",
    tgt_col:                str   = "text_en",
    contrastive_alpha:      float = 0.0,
    contrastive_temperature: float = 0.05,
) -> str:
    """Fine-tune a masked LM on sentence pairs (TLM) or monolingual text (pass tgt_col=src_col)."""
    if sentences_file:
        sent_df = pd.read_parquet(sentences_file)
        print(f"Loaded {len(sent_df):,} pairs from {sentences_file}")
    else:
        df      = pd.read_parquet("data/bloomfield_texts.parquet")
        sent_df = ParallelSentenceSplitter(df).split()
        if confidence > 0:
            sent_df = sent_df[sent_df.confidence >= confidence]
            print(f"Kept {len(sent_df):,} pairs with confidence ≥ {confidence}")
    cfg  = TLMConfig(model_name=model_name, output_dir=output_dir, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, grad_accum=grad_accum, max_length=max_length, hub_model_id=hub_model_id, wandb_project=wandb_project, contrastive_alpha=contrastive_alpha, contrastive_temperature=contrastive_temperature)
    ckpt = TLMFinetuner(cfg).fit(sent_df, src_col=src_col, tgt_col=tgt_col)
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
    """Train a figurative classifier on VUA20 + MAGPIE + FLUTE and return the checkpoint path."""
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
    corpus_file:         str        = "data/bloomfield_texts_sentences.parquet",
    epochs:              int        = 10,
    batch_size:          int        = 16,
    learning_rate:       float      = 5e-6,
    temperature:         float      = 2.0,
    max_length:          int        = 256,
    hub_model_id:        str | None = None,
    wandb_project:       str | None = None,
    output_dir:          str        = "data/figurative/distilled",
) -> str:
    """Cross-lingual knowledge distillation on the parallel corpus and return the checkpoint path."""
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
        max_length=max_length,
        hub_model_id=hub_model_id,
        output_dir=output_dir,
        wandb_project=wandb_project,
    )
    return figurative_distill_fn(cfg)


def calibrate(
    checkpoint:   str,
    output_dir:   str        = "data/calibrated",
    hub_model_id: str | None = None,
    wandb_project: str | None = None,
    annot_file:   str        = "data/figurative/annotations.parquet",
    epochs:       int        = 10,
    batch_size:   int        = 8,
    learning_rate: float     = 5e-6,
    max_length:   int        = 128,
    literal_ratio: int       = 3,
    gold_only:    bool       = False,
    holdout_fold: int | None = None,
) -> str:
    """Calibrate a CLKD model on DeepSeek-annotated Bloomfield sentences."""
    cfg = CalibrateConfig(
        checkpoint=checkpoint,
        annot_file=annot_file,
        output_dir=output_dir,
        hub_model_id=hub_model_id,
        wandb_project=wandb_project,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_length=max_length,
        literal_ratio=literal_ratio,
        gold_only=gold_only,
        holdout_fold=holdout_fold,
    )
    return figurative_calibrate_fn(cfg)


def figurative_eval_idioms(
    checkpoint:  str = "KonradBRG/xlm-r-plains-cree-en-tlm-figurative",
    idioms_file: str = "data/idioms.txt",
) -> None:
    """Evaluate a figurative model on the Cree idiom golden test set and save results to CSV."""
    slug = checkpoint.replace("/", "_").replace("\\", "_")
    output_file = f"data/figurative/idiom_eval_{slug}.csv"
    os.makedirs("data/figurative", exist_ok=True)
    model, tokenizer = figurative_load_model(checkpoint)
    result = eval_idioms(idioms_file, model, tokenizer)
    result["detail"].to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"Saved idiom evaluation to {output_file}")


def predict_figurative(
    checkpoint:  str = "KonradBRG/xlm-r-plains-cree-en-tlm-figurative",
    input_file:  str = "data/bloomfield_texts_sentences.parquet",
    output_file: str = "data/bloomfield_figurative.csv",
    idioms_file: str = "data/idioms.txt",
) -> pd.DataFrame:
    """Run figurative detection on Bloomfield Cree sentences and save results to CSV."""
    sentences_df = pd.read_parquet(input_file)
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
    """Split Bloomfield paragraphs into sentence pairs and write src ||| tgt to output file."""
    df = pd.read_parquet("data/bloomfield_texts.parquet")
    splitter = ParallelSentenceSplitter(df)
    splitter.write(output, min_confidence=confidence)
    sent_df = splitter.split()
    if confidence > 0:
        sent_df = sent_df[sent_df.confidence >= confidence]
    csv_path = "data/bloomfield_texts_sentences.parquet"
    sent_df.to_parquet(csv_path, index=False)
    print(f"Saved {len(sent_df):,} sentence pairs → {csv_path}")