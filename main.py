import argparse
import os

import pandas as pd
from tqdm import tqdm

from src.scrapers import BloomfieldScraper
from src.eda import EDA
from src.annotate import call_deepseek, format_prompt
from src.mt import TLMFinetuner, TLMConfig, ParallelSentenceSplitter


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


def fine_tune(confidence: float = 0.0) -> str:
    df       = pd.read_csv("data/bloomfield_texts.csv", encoding="utf-8-sig")
    sent_df  = ParallelSentenceSplitter(df).split()
    if confidence > 0:
        sent_df = sent_df[sent_df.confidence >= confidence]
        print(f"Kept {len(sent_df):,} pairs with confidence ≥ {confidence}")
    ckpt = TLMFinetuner(TLMConfig(epochs=5)).fit(sent_df)
    return ckpt


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
    parser.add_argument("--output",          default="data/sentences.txt",
                        help="Output path for --split-sentences (default: data/sentences.txt)")
    parser.add_argument("--confidence",      type=float, default=0.0,
                        help="Minimum alignment confidence filter for sentence pairs (0–1)")

    args = parser.parse_args()

    if not any([args.scrape, args.eda, args.annotate,
                args.split_sentences, args.fine_tune]):
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
        ckpt = fine_tune(args.confidence)
        print(f"Checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
