"""
Active annotation loop for Plains Cree figurative language detection.

Pipeline
--------
1. infer   — run calibrated model on unlabeled pool, save confidence scores
2. annotate — present low-confidence sentences (with itwêwina lookups) for
              human or DeepSeek annotation
3. retrain  — expand labeled set and re-run calibration

Modes
-----
  python scripts/active_annotate.py infer
  python scripts/active_annotate.py annotate [--mode human|deepseek]
  python scripts/active_annotate.py retrain
  python scripts/active_annotate.py status

Files
-----
  data/figurative/bloomfield_annotated.csv     — existing gold labels
  data/bloomfield_texts_sentences.csv          — full sentence pool
  data/figurative/active_pool.csv              — inference results on unlabeled
  data/figurative/active_annotations.csv       — newly annotated sentences
"""

from __future__ import annotations
import os, sys, argparse, csv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd

CALIBRATED_MODEL = "KonradBRG/xlm-mlm-plains-cree-en-calibrated"
ANNOT_FILE       = "data/figurative/bloomfield_annotated.parquet"
POOL_FILE        = "data/bloomfield_texts_sentences.parquet"
ACTIVE_POOL      = "data/figurative/active_pool.parquet"
ACTIVE_ANNOT     = "data/figurative/active_annotations.parquet"
LABELS           = ["literal", "idiom", "metaphor", "simile"]

# Confidence thresholds
HIGH_CONF        = 0.90   # above this → pseudo-label (if figurative) or skip (if literal)
LOW_CONF         = 0.75   # below this → send to annotation queue


# ── 1. INFER ──────────────────────────────────────────────────────────────────

def cmd_infer(args) -> None:
    import torch
    from src.figurative.predict import load_model, predict_sentences

    # Load already-annotated sentence texts to exclude
    annotated = pd.read_parquet(ANNOT_FILE)
    known_texts = set(annotated["text_cree"].dropna().str.strip().tolist())

    # Load full pool
    pool = pd.read_parquet(POOL_FILE)
    pool = pool.dropna(subset=["text_cree", "text_en"])
    pool["text_cree"] = pool["text_cree"].str.strip()

    # Remove already-annotated
    unlabeled = pool[~pool["text_cree"].isin(known_texts)].copy()
    print(f"Unlabeled pool: {len(unlabeled):,} sentences "
          f"({len(pool)-len(unlabeled):,} already annotated)")

    print(f"Loading model: {CALIBRATED_MODEL}")
    model, tok = load_model(CALIBRATED_MODEL)

    print("Running inference ...")
    preds = predict_sentences(
        unlabeled["text_cree"].tolist(), model, tok,
        batch_size=32, max_length=256,
    )

    df = unlabeled.reset_index(drop=True).copy()
    for key in ["label", "confidence", "prob_literal", "prob_idiom",
                "prob_metaphor", "prob_simile"]:
        df[key] = [p[key] for p in preds]

    os.makedirs(os.path.dirname(ACTIVE_POOL), exist_ok=True)
    df.to_parquet(ACTIVE_POOL, index=False)

    # Summary
    high_fig = df[(df["confidence"] >= HIGH_CONF) & (df["label"] != "literal")]
    low      = df[df["confidence"] < LOW_CONF]
    high_lit = df[(df["confidence"] >= HIGH_CONF) & (df["label"] == "literal")]

    print(f"\nResults saved → {ACTIVE_POOL}")
    print(f"  High-conf figurative (pseudo-label ready) : {len(high_fig):>5,}")
    print(f"  Low-conf  (annotation queue)              : {len(low):>5,}")
    print(f"  High-conf literal   (skip)                : {len(high_lit):>5,}")


# ── 2. ANNOTATE ───────────────────────────────────────────────────────────────

def _build_prompt(row: pd.Series, lookups: dict) -> str:
    from src.scrapers.scrape_itwewina import format_for_prompt
    return format_for_prompt(row["text_cree"], row["text_en"], lookups)


def _deepseek_annotate(prompt: str, model_probs: dict) -> str:
    """Call DeepSeek to annotate a sentence. Returns label string."""
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com",
        )
        prob_str = "  ".join(f"{k}={v:.2f}" for k, v in model_probs.items())
        system = (
            "You are an expert in Plains Cree linguistics and figurative language. "
            "Classify sentences as: literal, metaphor, idiom, or simile. "
            "Respond with ONLY one word."
        )
        user = (
            f"{prompt}\n\n"
            f"Model probability estimates: {prob_str}\n"
            f"Your classification (literal/metaphor/idiom/simile):"
        )
        resp = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[{"role": "system", "content": system},
                      {"role": "user",   "content": user}],
            max_tokens=10,
        )
        label = resp.choices[0].message.content.strip().lower()
        return label if label in LABELS else "literal"
    except Exception as exc:
        print(f"  DeepSeek error: {exc}")
        return "literal"


def _human_annotate(prompt: str, model_probs: dict) -> str:
    """Interactive CLI annotation. Returns label string."""
    print("\n" + "─" * 60)
    print(prompt)
    prob_str = "  ".join(f"{k}={v:.2f}" for k, v in model_probs.items())
    print(f"\nModel probs: {prob_str}")
    print("\n[l]iteral  [m]etaphor  [i]diom  [s]imile  [?]skip")
    while True:
        key = input("Label: ").strip().lower()
        mapping = {"l": "literal", "m": "metaphor", "i": "idiom",
                   "s": "simile", "literal": "literal",
                   "metaphor": "metaphor", "idiom": "idiom",
                   "simile": "simile", "?": None}
        if key in mapping:
            return mapping[key]
        print("  Enter l/m/i/s or ? to skip.")


def cmd_annotate(args) -> None:
    from src.scrapers.scrape_itwewina import lookup_sentence

    if not os.path.exists(ACTIVE_POOL):
        sys.exit(f"Run 'infer' first — {ACTIVE_POOL} not found.")

    pool = pd.read_parquet(ACTIVE_POOL)

    # Load existing active annotations to skip already-done ones
    done_texts: set[str] = set()
    if os.path.exists(ACTIVE_ANNOT):
        done = pd.read_parquet(ACTIVE_ANNOT)
        done_texts = set(done["text_cree"].dropna().str.strip().tolist())

    # Build annotation queue: low confidence OR high-conf figurative
    queue = pool[
        (pool["confidence"] < LOW_CONF) |
        ((pool["confidence"] >= HIGH_CONF) & (pool["label"] != "literal"))
    ].copy()
    queue = queue[~queue["text_cree"].isin(done_texts)]

    # Sort: figurative predictions first, then by ascending confidence
    queue["_fig"] = queue["label"] != "literal"
    queue = queue.sort_values(["_fig", "confidence"], ascending=[False, True])
    queue = queue.drop(columns=["_fig"])

    print(f"Annotation queue: {len(queue):,} sentences  "
          f"(mode={args.mode}, batch={args.batch})")

    annotated_rows = []
    batch = queue.head(args.batch)

    for _, row in batch.iterrows():
        # Dictionary lookup
        lookups = lookup_sentence(row["text_cree"], verbose=False)
        prompt  = _build_prompt(row, lookups)
        probs   = {l: row[f"prob_{l}"] for l in LABELS}

        if args.mode == "human":
            label = _human_annotate(prompt, probs)
            if label is None:
                continue
        else:
            print(f"  annotating: {row['text_cree'][:60]}...")
            label = _deepseek_annotate(prompt, probs)
            print(f"    → {label}")

        annotated_rows.append({
            "text_cree":   row["text_cree"],
            "text_en":     row["text_en"],
            "label":       label,
            "source":      f"active_{args.mode}",
            "model_label": row["label"],
            "confidence":  row["confidence"],
        })

    if not annotated_rows:
        print("Nothing annotated.")
        return

    # Append to active annotations file
    new_df = pd.DataFrame(annotated_rows)
    if os.path.exists(ACTIVE_ANNOT):
        existing = pd.read_parquet(ACTIVE_ANNOT)
        new_df = pd.concat([existing, new_df], ignore_index=True)
    new_df.to_parquet(ACTIVE_ANNOT, index=False)
    print(f"\nSaved {len(annotated_rows)} annotations → {ACTIVE_ANNOT}")
    label_dist = pd.Series([r["label"] for r in annotated_rows]).value_counts()
    print(f"Distribution: {label_dist.to_dict()}")


# ── 3. RETRAIN ────────────────────────────────────────────────────────────────

def cmd_retrain(args) -> None:
    import tempfile
    from funcs import calibrate

    # Load gold labels
    gold = pd.read_parquet(ANNOT_FILE)

    # Load active annotations
    if not os.path.exists(ACTIVE_ANNOT):
        sys.exit(f"No active annotations found at {ACTIVE_ANNOT}. Run 'annotate' first.")
    active = pd.read_parquet(ACTIVE_ANNOT)

    # Optionally also add high-confidence pseudo-labels
    if os.path.exists(ACTIVE_POOL):
        pool = pd.read_parquet(ACTIVE_POOL)
        pseudo = pool[
            (pool["confidence"] >= HIGH_CONF) & (pool["label"] != "literal")
        ][["text_cree", "text_en", "label"]].copy()
        pseudo["source"] = "pseudo_label"
        print(f"Adding {len(pseudo):,} high-confidence pseudo-labels")
    else:
        pseudo = pd.DataFrame(columns=["text_cree", "text_en", "label"])

    # Merge: gold + active + pseudo
    combined = pd.concat([
        gold[["text_cree", "text_en", "label"]],
        active[["text_cree", "text_en", "label"]],
        pseudo[["text_cree", "text_en", "label"]],
    ], ignore_index=True).drop_duplicates(subset=["text_cree"])

    print(f"\nTraining set: {len(combined):,} sentences "
          f"(gold={len(gold):,}, active={len(active):,}, pseudo={len(pseudo):,})")
    print(f"Label dist: {combined['label'].value_counts().to_dict()}")

    # Write merged annotation file to temp location
    tmp = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
    tmp.close()
    combined.to_parquet(tmp.name, index=False)

    print(f"\nRetraining calibration from {args.checkpoint} ...")
    calibrate(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        annot_file=tmp.name,
        epochs=args.epochs,
        batch_size=8,
        learning_rate=args.lr,
        literal_ratio=args.literal_ratio,
        max_length=128,
        wandb_project="FNLP",
    )
    os.unlink(tmp.name)
    print(f"\nRetrained model saved → {args.output_dir}")


# ── 4. STATUS ─────────────────────────────────────────────────────────────────

def cmd_status(_args) -> None:
    gold_n = len(pd.read_parquet(ANNOT_FILE)) \
        if os.path.exists(ANNOT_FILE) else 0

    pool_n = annot_n = pseudo_n = queue_n = 0
    if os.path.exists(ACTIVE_POOL):
        pool = pd.read_parquet(ACTIVE_POOL)
        pool_n  = len(pool)
        queue_n = len(pool[pool["confidence"] < LOW_CONF])
        pseudo_n = len(pool[
            (pool["confidence"] >= HIGH_CONF) & (pool["label"] != "literal")
        ])
    if os.path.exists(ACTIVE_ANNOT):
        annot_n = len(pd.read_parquet(ACTIVE_ANNOT))

    print(f"Gold annotations        : {gold_n:>5,}  ({ANNOT_FILE})")
    print(f"Unlabeled pool (inferred): {pool_n:>5,}  ({ACTIVE_POOL})")
    print(f"  → annotation queue    : {queue_n:>5,}  (conf < {LOW_CONF})")
    print(f"  → pseudo-label ready  : {pseudo_n:>5,}  (conf ≥ {HIGH_CONF}, figurative)")
    print(f"Active annotations done : {annot_n:>5,}  ({ACTIVE_ANNOT})")
    print(f"Total for retrain       : {gold_n + annot_n + pseudo_n:>5,}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("infer",  help="Run inference on unlabeled pool")
    sub.add_parser("status", help="Show annotation progress")

    ann = sub.add_parser("annotate", help="Annotate low-confidence sentences")
    ann.add_argument("--mode",  choices=["human", "deepseek"], default="human")
    ann.add_argument("--batch", type=int, default=20,
                     help="Sentences to annotate per session (default: 20)")

    ret = sub.add_parser("retrain", help="Retrain calibration with expanded data")
    ret.add_argument("--checkpoint", default=CALIBRATED_MODEL)
    ret.add_argument("--output-dir", default="data/figurative/active_calibrated")
    ret.add_argument("--epochs",       type=int,   default=15)
    ret.add_argument("--lr",           type=float, default=3.78e-6)
    ret.add_argument("--literal-ratio",type=int,   default=5)

    args = p.parse_args()
    {"infer": cmd_infer, "annotate": cmd_annotate,
     "retrain": cmd_retrain, "status": cmd_status}[args.cmd](args)


if __name__ == "__main__":
    main()
