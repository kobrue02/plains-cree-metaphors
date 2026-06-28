"""
Active annotation loop — single job (infer → DeepSeek annotate → retrain).

All three phases run in sequence and log to a single wandb run so training
curves, annotation stats, and data provenance are all in one place.

Usage
-----
  # Full loop (infer + annotate + retrain)
  python scripts/active_loop.py

  # Skip inference if active_pool.csv already exists
  python scripts/active_loop.py --skip-infer

  # Annotate all queue items (default) or cap at N
  python scripts/active_loop.py --max-annotate 500

  # Dry run: infer + annotate only, no retraining
  python scripts/active_loop.py --no-retrain

  # Push retrained model to Hub
  python scripts/active_loop.py --push-to-hub KonradBRG/xlm-mlm-plains-cree-en-active-v1
"""

from __future__ import annotations
import os, sys, argparse, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

WANDB_PROJECT    = "FNLP"
CALIBRATED_MODEL = "KonradBRG/xlm-mlm-plains-cree-en-calibrated"
ANNOT_FILE       = "data/figurative/bloomfield_annotated.csv"
POOL_FILE        = "data/bloomfield_texts_sentences.csv"
ACTIVE_POOL      = "data/figurative/active_pool.csv"
ACTIVE_ANNOT     = "data/figurative/active_annotations.csv"
LABELS           = ["literal", "idiom", "metaphor", "simile"]

HIGH_CONF = 0.90
LOW_CONF  = 0.75


# ── Phase 1: Infer ────────────────────────────────────────────────────────────

def phase_infer(checkpoint: str) -> pd.DataFrame:
    import wandb
    from src.figurative.predict import load_model, predict_sentences

    annotated   = pd.read_csv(ANNOT_FILE, encoding="utf-8-sig")
    known_texts = set(annotated["text_cree"].dropna().str.strip().tolist())

    pool     = pd.read_csv(POOL_FILE, encoding="utf-8-sig")
    pool     = pool.dropna(subset=["text_cree", "text_en"])
    pool["text_cree"] = pool["text_cree"].str.strip()
    unlabeled = pool[~pool["text_cree"].isin(known_texts)].copy().reset_index(drop=True)

    print(f"[infer] {len(unlabeled):,} unlabeled sentences")
    model, tok = load_model(checkpoint)

    preds = predict_sentences(
        unlabeled["text_cree"].tolist(), model, tok,
        batch_size=32, max_length=256,
    )
    for key in ["label", "confidence", "prob_literal", "prob_idiom",
                "prob_metaphor", "prob_simile"]:
        unlabeled[key] = [p[key] for p in preds]

    os.makedirs(os.path.dirname(ACTIVE_POOL), exist_ok=True)
    unlabeled.to_csv(ACTIVE_POOL, index=False, encoding="utf-8-sig")

    # Summary counts
    high_fig = (unlabeled["confidence"] >= HIGH_CONF) & (unlabeled["label"] != "literal")
    low      =  unlabeled["confidence"] < LOW_CONF
    high_lit = (unlabeled["confidence"] >= HIGH_CONF) & (unlabeled["label"] == "literal")

    conf_mean = unlabeled["confidence"].mean()
    conf_hist = wandb.Histogram(unlabeled["confidence"].tolist())

    wandb.log({
        "infer/pool_size":         len(unlabeled),
        "infer/already_annotated": len(known_texts),
        "infer/high_conf_fig":     int(high_fig.sum()),
        "infer/low_conf_queue":    int(low.sum()),
        "infer/high_conf_literal": int(high_lit.sum()),
        "infer/mean_confidence":   round(conf_mean, 4),
        "infer/confidence_hist":   conf_hist,
        **{f"infer/pred_{l}": int((unlabeled["label"] == l).sum()) for l in LABELS},
    })

    print(f"[infer] high-conf figurative={high_fig.sum()}  "
          f"queue={low.sum()}  skip-literal={high_lit.sum()}")
    return unlabeled


# ── Phase 2: DeepSeek annotation ─────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are annotating Plains Cree sentences from Leonard Bloomfield's 1934 fieldwork
transcriptions for figurative language.

Critical context:
- The English gloss is a MEANING-FOR-MEANING translation, not word-for-word.
  It already conveys the intended sense, so a gloss that reads literally in
  English does NOT mean the Cree original is literal.
- Figurative language must be identified in the CREE STRUCTURE. Compare the
  word-level Cree meanings (from the dictionary entries) against the English
  gloss: if the Cree words say one thing literally but the gloss says something
  different, figurative language is likely present.
- Plains Cree oral narratives make heavy use of idioms (fixed expressions whose
  meaning cannot be composed from parts), metaphors (conceptual transfers, e.g.
  body-part terms used for landscape or emotions), and similes (explicit
  comparisons with 'like').
- Our trained cross-lingual classifier has already flagged this sentence.
  Treat its prediction as a meaningful prior — only override it if the
  word-level evidence clearly supports a different label.

Respond with EXACTLY one word: literal / metaphor / idiom / simile\
"""


def _deepseek_annotate_one(prompt: str, model_probs: dict) -> str:
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
    )
    top_label = max(model_probs, key=model_probs.__getitem__)
    top_conf  = model_probs[top_label]
    model_note = (
        f"Classifier prediction: {top_label.upper()} ({top_conf:.0%} confidence)\n"
        f"Full distribution: "
        + "  ".join(f"{k}={v:.2f}" for k, v in model_probs.items())
    )
    resp = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": f"{prompt}\n\n{model_note}\n\nYour label:"},
        ],
        max_tokens=10,
    )
    label = resp.choices[0].message.content.strip().lower()
    return label if label in LABELS else "literal"


def phase_annotate(pool: pd.DataFrame, max_annotate: int) -> pd.DataFrame:
    import wandb
    from src.scrapers.scrape_itwewina import lookup_sentence, format_for_prompt

    # Queue: only low-confidence sentences — high-conf figurative go straight to
    # pseudo-labels in phase_retrain and should not be second-guessed by DeepSeek
    queue = pool[pool["confidence"] < LOW_CONF].copy()

    # Skip already-annotated
    done_texts: set[str] = set()
    if os.path.exists(ACTIVE_ANNOT):
        done = pd.read_csv(ACTIVE_ANNOT, encoding="utf-8-sig")
        done_texts = set(done["text_cree"].dropna().str.strip().tolist())
        queue = queue[~queue["text_cree"].isin(done_texts)]

    # Figurative predictions first, then ascending confidence
    queue["_fig"] = queue["label"] != "literal"
    queue = queue.sort_values(["_fig", "confidence"], ascending=[False, True]).drop(columns=["_fig"])
    if max_annotate > 0:
        queue = queue.head(max_annotate)

    print(f"[annotate] {len(queue):,} sentences to annotate via DeepSeek")

    rows = []
    agreements = []
    label_counts = {l: 0 for l in LABELS}
    errors = 0

    for i, (_, row) in enumerate(queue.iterrows()):
        try:
            lookups = lookup_sentence(row["text_cree"], verbose=False)
            prompt  = format_for_prompt(row["text_cree"], row["text_en"], lookups)
            probs   = {l: float(row[f"prob_{l}"]) for l in LABELS}
            label   = _deepseek_annotate_one(prompt, probs)
        except Exception as exc:
            print(f"  [annotate] error on row {i}: {exc}")
            label = "literal"
            errors += 1

        agrees = label == row["label"]
        agreements.append(int(agrees))
        label_counts[label] += 1

        rows.append({
            "text_cree":   row["text_cree"],
            "text_en":     row["text_en"],
            "label":       label,
            "source":      "active_deepseek",
            "model_label": row["label"],
            "confidence":  row["confidence"],
        })

        if (i + 1) % 20 == 0:
            running_agreement = sum(agreements) / len(agreements)
            print(f"  [{i+1}/{len(queue)}] agreement={running_agreement:.2%}  "
                  f"dist={label_counts}")
            wandb.log({
                "annotate/sentences_done": i + 1,
                "annotate/running_agreement": running_agreement,
                **{f"annotate/running_{l}": label_counts[l] for l in LABELS},
            })

    new_df = pd.DataFrame(rows)

    # Merge with any existing active annotations
    if os.path.exists(ACTIVE_ANNOT) and done_texts:
        existing = pd.read_csv(ACTIVE_ANNOT, encoding="utf-8-sig")
        new_df = pd.concat([existing, new_df], ignore_index=True)

    new_df.to_csv(ACTIVE_ANNOT, index=False, encoding="utf-8-sig")

    overall_agreement = sum(agreements) / len(agreements) if agreements else 0.0
    wandb.log({
        "annotate/total":           len(rows),
        "annotate/errors":          errors,
        "annotate/model_agreement": round(overall_agreement, 4),
        **{f"annotate/final_{l}": label_counts[l] for l in LABELS},
    })

    print(f"[annotate] done — {len(rows)} annotations, "
          f"agreement={overall_agreement:.2%}, dist={label_counts}")
    return new_df


# ── Phase 3: Retrain ──────────────────────────────────────────────────────────

def phase_retrain(
    checkpoint:   str,
    output_dir:   str,
    hub_model_id: str | None,
    epochs:       int,
    lr:           float,
    literal_ratio: int,
) -> None:
    import wandb
    from funcs import calibrate

    gold   = pd.read_csv(ANNOT_FILE, encoding="utf-8-sig")
    active = pd.read_csv(ACTIVE_ANNOT, encoding="utf-8-sig")

    # High-confidence figurative predictions → pseudo-labels
    pool   = pd.read_csv(ACTIVE_POOL, encoding="utf-8-sig")
    pseudo = pool[
        (pool["confidence"] >= HIGH_CONF) & (pool["label"] != "literal")
    ][["text_cree", "text_en", "label"]].copy()

    # Remove pseudo-label candidates already in active annotations
    active_texts = set(active["text_cree"].dropna().str.strip().tolist())
    pseudo = pseudo[~pseudo["text_cree"].isin(active_texts)]

    combined = pd.concat([
        gold[["text_cree", "text_en", "label"]],
        active[["text_cree", "text_en", "label"]],
        pseudo[["text_cree", "text_en", "label"]],
    ], ignore_index=True).drop_duplicates(subset=["text_cree"])

    print(f"[retrain] {len(combined):,} sentences "
          f"(gold={len(gold)}, active={len(active)}, pseudo={len(pseudo)})")

    wandb.log({
        "retrain/gold_sentences":   len(gold),
        "retrain/active_sentences": len(active),
        "retrain/pseudo_sentences": len(pseudo),
        "retrain/total_sentences":  len(combined),
        **{f"retrain/label_{l}": int((combined["label"] == l).sum()) for l in LABELS},
    })

    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w",
                                      encoding="utf-8-sig", newline="")
    combined.to_csv(tmp.name, index=False)
    tmp.close()

    # Tell HF Trainer's WandbCallback to join our existing run
    os.environ["WANDB_PROJECT"] = WANDB_PROJECT
    os.environ["WANDB_RUN_ID"]  = wandb.run.id
    os.environ["WANDB_RESUME"]  = "allow"

    print(f"[retrain] training from {checkpoint} ...")
    calibrate(
        checkpoint=checkpoint,
        output_dir=output_dir,
        hub_model_id=hub_model_id,
        annot_file=tmp.name,
        epochs=epochs,
        batch_size=8,
        learning_rate=lr,
        literal_ratio=literal_ratio,
        max_length=128,
        wandb_project=WANDB_PROJECT,
    )
    os.unlink(tmp.name)
    print(f"[retrain] saved → {output_dir}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint",   default=CALIBRATED_MODEL,
                   help="Model to run inference with and retrain from")
    p.add_argument("--skip-infer",   action="store_true",
                   help="Skip inference — reuse existing active_pool.csv")
    p.add_argument("--max-annotate", type=int, default=0,
                   help="Max sentences to annotate (0 = all)")
    p.add_argument("--no-retrain",   action="store_true",
                   help="Stop after annotation, do not retrain")
    p.add_argument("--output-dir",   default="data/figurative/active_calibrated")
    p.add_argument("--push-to-hub",  default=None, metavar="HUB_ID",
                   help="Push retrained model to this Hub ID")
    p.add_argument("--epochs",       type=int,   default=15)
    p.add_argument("--lr",           type=float, default=3.78e-6)
    p.add_argument("--literal-ratio",type=int,   default=5)
    p.add_argument("--run-name",     default=None,
                   help="wandb run name (default: auto)")
    args = p.parse_args()

    import wandb
    wandb.init(
        project=WANDB_PROJECT,
        name=args.run_name,
        job_type="active-loop",
        config={
            "checkpoint":    args.checkpoint,
            "skip_infer":    args.skip_infer,
            "max_annotate":  args.max_annotate,
            "no_retrain":    args.no_retrain,
            "epochs":        args.epochs,
            "lr":            args.lr,
            "literal_ratio": args.literal_ratio,
            "high_conf":     HIGH_CONF,
            "low_conf":      LOW_CONF,
        },
    )

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    if args.skip_infer:
        if not os.path.exists(ACTIVE_POOL):
            sys.exit(f"--skip-infer set but {ACTIVE_POOL} not found. Run without --skip-infer first.")
        print(f"[infer] skipping — loading {ACTIVE_POOL}")
        pool = pd.read_csv(ACTIVE_POOL, encoding="utf-8-sig")
        wandb.log({"infer/pool_size": len(pool), "infer/skipped": True})
    else:
        pool = phase_infer(args.checkpoint)

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    phase_annotate(pool, max_annotate=args.max_annotate)

    # ── Phase 3 ───────────────────────────────────────────────────────────────
    if not args.no_retrain:
        phase_retrain(
            checkpoint=args.checkpoint,
            output_dir=args.output_dir,
            hub_model_id=args.push_to_hub,
            epochs=args.epochs,
            lr=args.lr,
            literal_ratio=args.literal_ratio,
        )

    wandb.finish()
    print("[active-loop] done.")


if __name__ == "__main__":
    main()
