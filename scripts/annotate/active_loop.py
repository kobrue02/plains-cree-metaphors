"""
Active annotation loop — single job (infer → DeepSeek annotate → retrain).

All three phases run in sequence and log to a single wandb run so training
curves, annotation stats, and data provenance are all in one place.

Usage
-----
  # Full loop (infer + annotate + retrain)
  python scripts/annotate/active_loop.py
  python scripts/annotate/active_loop.py loop

  # Skip inference if active_pool.parquet already exists
  python scripts/annotate/active_loop.py --skip-infer
  python scripts/annotate/active_loop.py loop --skip-infer

  # Annotate all queue items (default) or cap at N
  python scripts/annotate/active_loop.py --max-annotate 500

  # Dry run: infer + annotate only, no retraining
  python scripts/annotate/active_loop.py --no-retrain

  # Push retrained model to Hub
  python scripts/annotate/active_loop.py --push-to-hub KonradBRG/xlm-mlm-plains-cree-en-active-v1

  # Individual phase subcommands (no wandb by default)
  python scripts/annotate/active_loop.py infer
  python scripts/annotate/active_loop.py annotate [--mode human|deepseek] [--batch N]
  python scripts/annotate/active_loop.py retrain
  python scripts/annotate/active_loop.py status
"""

from __future__ import annotations
import os, sys, argparse, tempfile
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd

WANDB_PROJECT    = "FNLP"
CALIBRATED_MODEL = "KonradBRG/xlm-mlm-plains-cree-en-calibrated"
ANNOT_FILE       = "data/figurative/annotations.parquet"
POOL_FILE        = "data/bloomfield_texts_sentences.parquet"
ACTIVE_POOL      = "data/figurative/active_pool.parquet"
LABELS           = ["literal", "idiom", "metaphor", "simile"]

HIGH_CONF = 0.90
LOW_CONF  = 0.75


# ── Phase 1: Infer (full-loop version, logs to wandb) ─────────────────────────

def phase_infer(checkpoint: str) -> pd.DataFrame:
    import wandb
    from src.figurative.predict import load_model, predict_sentences

    annotated   = pd.read_parquet(ANNOT_FILE)
    known_texts = set(annotated["text_cree"].dropna().str.strip().tolist())

    pool     = pd.read_parquet(POOL_FILE)
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
    unlabeled.to_parquet(ACTIVE_POOL, index=False)

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


# ── Phase 2: DeepSeek annotation (full-loop version, logs to wandb) ───────────

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

    # only low-confidence sentences — high-conf figurative become pseudo-labels in phase_retrain
    queue = pool[pool["confidence"] < LOW_CONF].copy()

    done_texts: set[str] = set()
    if os.path.exists(ANNOT_FILE):
        done_all = pd.read_parquet(ANNOT_FILE)
        active_done = done_all[done_all["source"] == "active_deepseek"]
        done_texts = set(active_done["text_cree"].dropna().str.strip().tolist())
        queue = queue[~queue["text_cree"].isin(done_texts)]

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

    if os.path.exists(ANNOT_FILE):
        existing = pd.read_parquet(ANNOT_FILE)
        new_df = pd.concat([existing, new_df], ignore_index=True)

    new_df.drop_duplicates(subset=["text_cree"], inplace=True)
    new_df.to_parquet(ANNOT_FILE, index=False)

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


# ── Phase 3: Retrain (full-loop version, logs to wandb) ───────────────────────

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

    # Unified annotations file contains both gold (bloomfield) and active rows
    all_annot = pd.read_parquet(ANNOT_FILE)
    gold   = all_annot[all_annot["source"] != "active_deepseek"]
    active = all_annot[all_annot["source"] == "active_deepseek"]

    # High-confidence figurative predictions → pseudo-labels
    pool   = pd.read_parquet(ACTIVE_POOL)
    pseudo = pool[
        (pool["confidence"] >= HIGH_CONF) & (pool["label"] != "literal")
    ][["text_cree", "text_en", "label"]].copy()

    annotated_texts = set(all_annot["text_cree"].dropna().str.strip().tolist())
    pseudo = pseudo[~pseudo["text_cree"].isin(annotated_texts)]

    combined = pd.concat([
        all_annot[["text_cree", "text_en", "label"]],
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

    tmp = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
    tmp.close()
    combined.to_parquet(tmp.name, index=False)

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


# ── Subcommand implementations (no wandb) ─────────────────────────────────────

def cmd_infer(args) -> None:
    """Run inference only (no wandb)."""
    import torch
    from src.figurative.predict import load_model, predict_sentences

    annotated = pd.read_parquet(ANNOT_FILE)
    known_texts = set(annotated["text_cree"].dropna().str.strip().tolist())

    pool = pd.read_parquet(POOL_FILE)
    pool = pool.dropna(subset=["text_cree", "text_en"])
    pool["text_cree"] = pool["text_cree"].str.strip()

    unlabeled = pool[~pool["text_cree"].isin(known_texts)].copy()
    print(f"Unlabeled pool: {len(unlabeled):,} sentences "
          f"({len(pool)-len(unlabeled):,} already annotated)")

    checkpoint = getattr(args, "checkpoint", CALIBRATED_MODEL)
    print(f"Loading model: {checkpoint}")
    model, tok = load_model(checkpoint)

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

    high_fig = df[(df["confidence"] >= HIGH_CONF) & (df["label"] != "literal")]
    low      = df[df["confidence"] < LOW_CONF]
    high_lit = df[(df["confidence"] >= HIGH_CONF) & (df["label"] == "literal")]

    print(f"\nResults saved → {ACTIVE_POOL}")
    print(f"  High-conf figurative (pseudo-label ready) : {len(high_fig):>5,}")
    print(f"  Low-conf  (annotation queue)              : {len(low):>5,}")
    print(f"  High-conf literal   (skip)                : {len(high_lit):>5,}")


def _deepseek_annotate_simple(prompt: str, model_probs: dict) -> str:
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


def _human_annotate(prompt: str, model_probs: dict) -> str | None:
    """Interactive CLI annotation. Returns label string or None to skip."""
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
    """Annotate queue only (no wandb)."""
    from src.scrapers.scrape_itwewina import lookup_sentence
    from src.scrapers.scrape_itwewina import format_for_prompt

    if not os.path.exists(ACTIVE_POOL):
        sys.exit(f"Run 'infer' first — {ACTIVE_POOL} not found.")

    pool = pd.read_parquet(ACTIVE_POOL)

    done_texts: set[str] = set()
    if os.path.exists(ANNOT_FILE):
        done_all = pd.read_parquet(ANNOT_FILE)
        active_done = done_all[done_all["source"].str.startswith("active_", na=False)]
        done_texts = set(active_done["text_cree"].dropna().str.strip().tolist())

    queue = pool[
        (pool["confidence"] < LOW_CONF) |
        ((pool["confidence"] >= HIGH_CONF) & (pool["label"] != "literal"))
    ].copy()
    queue = queue[~queue["text_cree"].isin(done_texts)]

    queue["_fig"] = queue["label"] != "literal"
    queue = queue.sort_values(["_fig", "confidence"], ascending=[False, True])
    queue = queue.drop(columns=["_fig"])

    mode  = getattr(args, "mode",  "human")
    batch = getattr(args, "batch", 20)
    print(f"Annotation queue: {len(queue):,} sentences  "
          f"(mode={mode}, batch={batch})")

    annotated_rows = []
    batch_df = queue.head(batch)

    for _, row in batch_df.iterrows():
        lookups = lookup_sentence(row["text_cree"], verbose=False)
        prompt  = format_for_prompt(row["text_cree"], row["text_en"], lookups)
        probs   = {l: row[f"prob_{l}"] for l in LABELS}

        if mode == "human":
            label = _human_annotate(prompt, probs)
            if label is None:
                continue
        else:
            print(f"  annotating: {row['text_cree'][:60]}...")
            label = _deepseek_annotate_simple(prompt, probs)
            print(f"    → {label}")

        annotated_rows.append({
            "text_cree":   row["text_cree"],
            "text_en":     row["text_en"],
            "label":       label,
            "source":      f"active_{mode}",
            "model_label": row["label"],
            "confidence":  row["confidence"],
        })

    if not annotated_rows:
        print("Nothing annotated.")
        return

    new_df = pd.DataFrame(annotated_rows)
    if os.path.exists(ANNOT_FILE):
        existing = pd.read_parquet(ANNOT_FILE)
        new_df = pd.concat([existing, new_df], ignore_index=True)
    new_df.drop_duplicates(subset=["text_cree"], inplace=True)
    new_df.to_parquet(ANNOT_FILE, index=False)
    print(f"\nSaved {len(annotated_rows)} annotations → {ANNOT_FILE}")
    label_dist = pd.Series([r["label"] for r in annotated_rows]).value_counts()
    print(f"Distribution: {label_dist.to_dict()}")


def cmd_retrain(args) -> None:
    """Retrain calibration with expanded data (no wandb)."""
    from funcs import calibrate

    if not os.path.exists(ANNOT_FILE):
        sys.exit(f"No annotations found at {ANNOT_FILE}. Run 'annotate' first.")
    all_annot = pd.read_parquet(ANNOT_FILE)
    is_active = all_annot["source"].str.startswith("active_", na=False)
    gold   = all_annot[~is_active]
    active = all_annot[is_active]

    if os.path.exists(ACTIVE_POOL):
        pool = pd.read_parquet(ACTIVE_POOL)
        pseudo = pool[
            (pool["confidence"] >= HIGH_CONF) & (pool["label"] != "literal")
        ][["text_cree", "text_en", "label"]].copy()
        pseudo["source"] = "pseudo_label"
        print(f"Adding {len(pseudo):,} high-confidence pseudo-labels")
    else:
        pseudo = pd.DataFrame(columns=["text_cree", "text_en", "label"])

    annotated_texts = set(all_annot["text_cree"].dropna().str.strip().tolist())
    pseudo = pseudo[~pseudo["text_cree"].isin(annotated_texts)]
    combined = pd.concat([
        all_annot[["text_cree", "text_en", "label"]],
        pseudo[["text_cree", "text_en", "label"]],
    ], ignore_index=True).drop_duplicates(subset=["text_cree"])

    print(f"\nTraining set: {len(combined):,} sentences "
          f"(gold={len(gold):,}, active={len(active):,}, pseudo={len(pseudo):,})")
    print(f"Label dist: {combined['label'].value_counts().to_dict()}")

    tmp = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
    tmp.close()
    combined.to_parquet(tmp.name, index=False)

    checkpoint  = getattr(args, "checkpoint",   CALIBRATED_MODEL)
    output_dir  = getattr(args, "output_dir",   "data/figurative/active_calibrated")
    epochs      = getattr(args, "epochs",        15)
    lr          = getattr(args, "lr",            3.78e-6)
    literal_ratio = getattr(args, "literal_ratio", 5)

    print(f"\nRetraining calibration from {checkpoint} ...")
    calibrate(
        checkpoint=checkpoint,
        output_dir=output_dir,
        annot_file=tmp.name,
        epochs=epochs,
        batch_size=8,
        learning_rate=lr,
        literal_ratio=literal_ratio,
        max_length=128,
        wandb_project="FNLP",
    )
    os.unlink(tmp.name)
    print(f"\nRetrained model saved → {output_dir}")


def cmd_status(_args) -> None:
    """Show annotation progress (no wandb)."""
    gold_n = active_n = 0
    if os.path.exists(ANNOT_FILE):
        all_annot = pd.read_parquet(ANNOT_FILE)
        is_active = all_annot["source"].str.startswith("active_", na=False)
        gold_n   = int((~is_active).sum())
        active_n = int(is_active.sum())

    pool_n = pseudo_n = queue_n = 0
    if os.path.exists(ACTIVE_POOL):
        pool = pd.read_parquet(ACTIVE_POOL)
        pool_n  = len(pool)
        queue_n = len(pool[pool["confidence"] < LOW_CONF])
        pseudo_n = len(pool[
            (pool["confidence"] >= HIGH_CONF) & (pool["label"] != "literal")
        ])

    print(f"Gold annotations        : {gold_n:>5,}  ({ANNOT_FILE})")
    print(f"Active annotations done : {active_n:>5,}  ({ANNOT_FILE})")
    print(f"Unlabeled pool (inferred): {pool_n:>5,}  ({ACTIVE_POOL})")
    print(f"  → annotation queue    : {queue_n:>5,}  (conf < {LOW_CONF})")
    print(f"  → pseudo-label ready  : {pseudo_n:>5,}  (conf ≥ {HIGH_CONF}, figurative)")
    print(f"Total for retrain       : {gold_n + active_n + pseudo_n:>5,}")


# ── Full loop runner ───────────────────────────────────────────────────────────

def run_loop(args) -> None:
    """Run the full infer → annotate → retrain loop with wandb."""
    import wandb
    wandb.init(
        project=WANDB_PROJECT,
        name=getattr(args, "run_name", None),
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
        pool = pd.read_parquet(ACTIVE_POOL)
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd")

    # ── Full loop subcommand (also the default when no subcommand given) ───────
    loop_p = sub.add_parser("loop", help="Run full infer→annotate→retrain loop (default)")
    loop_p.add_argument("--checkpoint",   default=CALIBRATED_MODEL,
                        help="Model to run inference with and retrain from")
    loop_p.add_argument("--skip-infer",   action="store_true",
                        help="Skip inference — reuse existing active_pool.parquet")
    loop_p.add_argument("--max-annotate", type=int, default=0,
                        help="Max sentences to annotate (0 = all)")
    loop_p.add_argument("--no-retrain",   action="store_true",
                        help="Stop after annotation, do not retrain")
    loop_p.add_argument("--output-dir",   default="data/figurative/active_calibrated")
    loop_p.add_argument("--push-to-hub",  default=None, metavar="HUB_ID",
                        help="Push retrained model to this Hub ID")
    loop_p.add_argument("--epochs",       type=int,   default=15)
    loop_p.add_argument("--lr",           type=float, default=3.78e-6)
    loop_p.add_argument("--literal-ratio",type=int,   default=5)
    loop_p.add_argument("--run-name",     default=None,
                        help="wandb run name (default: auto)")

    # ── Individual subcommands ─────────────────────────────────────────────────
    infer_p = sub.add_parser("infer", help="Run inference on unlabeled pool")
    infer_p.add_argument("--checkpoint", default=CALIBRATED_MODEL)

    ann_p = sub.add_parser("annotate", help="Annotate low-confidence sentences")
    ann_p.add_argument("--mode",  choices=["human", "deepseek"], default="human")
    ann_p.add_argument("--batch", type=int, default=20,
                       help="Sentences to annotate per session (default: 20)")

    ret_p = sub.add_parser("retrain", help="Retrain calibration with expanded data")
    ret_p.add_argument("--checkpoint",    default=CALIBRATED_MODEL)
    ret_p.add_argument("--output-dir",    default="data/figurative/active_calibrated")
    ret_p.add_argument("--epochs",        type=int,   default=15)
    ret_p.add_argument("--lr",            type=float, default=3.78e-6)
    ret_p.add_argument("--literal-ratio", type=int,   default=5)

    sub.add_parser("status", help="Show annotation progress")

    # ── for backwards compatibility: top-level loop flags forwarded to run_loop() when no subcommand is given ─────
    p.add_argument("--checkpoint",   default=CALIBRATED_MODEL)
    p.add_argument("--skip-infer",   action="store_true")
    p.add_argument("--max-annotate", type=int, default=0)
    p.add_argument("--no-retrain",   action="store_true")
    p.add_argument("--output-dir",   default="data/figurative/active_calibrated")
    p.add_argument("--push-to-hub",  default=None, metavar="HUB_ID")
    p.add_argument("--epochs",       type=int,   default=15)
    p.add_argument("--lr",           type=float, default=3.78e-6)
    p.add_argument("--literal-ratio",type=int,   default=5)
    p.add_argument("--run-name",     default=None)

    args = p.parse_args()

    if args.cmd is None or args.cmd == "loop":
        run_loop(args)
    elif args.cmd == "infer":
        cmd_infer(args)
    elif args.cmd == "annotate":
        cmd_annotate(args)
    elif args.cmd == "retrain":
        cmd_retrain(args)
    elif args.cmd == "status":
        cmd_status(args)


if __name__ == "__main__":
    main()
