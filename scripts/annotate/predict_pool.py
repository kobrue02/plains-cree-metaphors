"""
Label the sentence pool with a trained classifier only (no DeepSeek involved).

Runs the same pool filtering as deepseek_label_pool.py (excluding the 1930
manuscript by default) so the two label sets line up 1:1 for
scripts/annotate/agreement_eval.py.

Usage:
  python scripts/annotate/predict_pool.py --checkpoint KonradBRG/xlm-mlm-plains-cree-en-calibrated
  python scripts/annotate/predict_pool.py --checkpoint <ckpt> --limit 500   # smoke test
"""

from __future__ import annotations
import argparse, os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

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
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    pool = load_pool(args.pool, exclude_source=args.exclude_source or None, limit=args.limit)
    print(f"[pool] {len(pool):,} sentences (excluding source={args.exclude_source!r})")

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
