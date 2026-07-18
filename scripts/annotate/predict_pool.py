"""
Label the sentence pool with a trained classifier only (no DeepSeek involved).

Runs the same pool filtering as deepseek_label_pool.py (excluding the 1930
manuscript by default) so the two label sets line up 1:1 for
scripts/annotate/agreement_eval.py. That alignment isn't automatic though:
deepseek_label_pool.py additionally excludes gold-labeled sentences (via its
own --gold-file), which this script doesn't do on its own, and the
underlying sentence pool can drift between when a silver-labeling run
happened and whenever this is run (corrections, outage cleanups, etc.) —
both cause the two label sets to disagree on which sentences even exist,
which showed up once already as ~2,000 sentences dropped from a merge. Use
--restrict-to to guarantee exact 1:1 alignment against a specific silver
label file instead of relying on the current pool matching it by chance.

Usage:
  python scripts/annotate/predict_pool.py --checkpoint KonradBRG/xlm-mlm-plains-cree-en-figurative
  python scripts/annotate/predict_pool.py --checkpoint <ckpt> --limit 500   # smoke test
  python scripts/annotate/predict_pool.py --checkpoint <ckpt> \
      --restrict-to data/figurative/deepseek_labels_qwen_qwen3.5-122b-a10b.parquet
"""

from __future__ import annotations
import argparse, os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd

from scripts.annotate._pool_utils import load_pool, POOL_FILE, EXCLUDE_SOURCE
from src.figurative.predict import load_model, predict_sentences

OUTPUT_FILE = "data/figurative/model_predictions.parquet"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True,
                   help="Hub or local checkpoint to run inference with")
    p.add_argument("--pool",    default=POOL_FILE)
    p.add_argument("--exclude-source", default=EXCLUDE_SOURCE,
                   help="source_file value to exclude (default: bloomfield_1930; pass '' to include everything)")
    p.add_argument("--out",         default=OUTPUT_FILE)
    p.add_argument("--limit",       type=int, default=None,
                   help="Cap the number of sentences (for a smoke test)")
    p.add_argument("--batch-size",  type=int, default=32)
    p.add_argument("--max-length",  type=int, default=256)
    p.add_argument("--restrict-to", default=None, metavar="PARQUET",
                   help="Only predict on sentences whose text_cree appears in this file "
                        "(e.g. a deepseek_labels_qwen_*.parquet) — guarantees exact 1:1 "
                        "alignment for agreement_eval.py instead of hoping the current "
                        "pool still matches whatever that file was built from.")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    pool = load_pool(args.pool, exclude_source=args.exclude_source or None, limit=args.limit)
    print(f"[pool] {len(pool):,} sentences (excluding source={args.exclude_source!r})")

    if args.restrict_to:
        restrict_texts = set(pd.read_parquet(args.restrict_to)["text_cree"].str.strip())
        before = len(pool)
        pool = pool[pool["text_cree"].isin(restrict_texts)].reset_index(drop=True)
        missing = len(restrict_texts) - len(pool)
        print(f"[pool] restricted to {args.restrict_to} — {len(pool):,}/{before:,} sentences kept"
              + (f"  ({missing:,} restrict-to sentences not found in the current pool)" if missing else ""))

    print(f"[model] loading {args.checkpoint} ...")
    model, tokenizer = load_model(args.checkpoint)

    preds = predict_sentences(
        pool["text_cree"].tolist(), model, tokenizer,
        batch_size=args.batch_size, max_length=args.max_length,
    )
    pool["model_label"]      = [pr["label"]      for pr in preds]
    pool["model_confidence"] = [pr["confidence"] for pr in preds]
    for name in ("literal", "idiom", "metaphor", "simile"):
        key = f"prob_{name}"
        if key in preds[0]:
            pool[key] = [pr[key] for pr in preds]

    out_cols = (["paragraph_id", "sentence_id", "text_cree", "text_en",
                 "model_label", "model_confidence"]
                + [f"prob_{n}" for n in ("literal", "idiom", "metaphor", "simile")])
    pool[[c for c in out_cols if c in pool.columns]].to_parquet(args.out, index=False)

    print(f"\nLabel distribution:")
    print(pool["model_label"].value_counts().to_string())
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
