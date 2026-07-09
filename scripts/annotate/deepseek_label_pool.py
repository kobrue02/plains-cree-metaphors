"""
Label the sentence pool with DeepSeek only (no model involved).

For every Cree sentence in the pool (excluding the 1930 manuscript by default),
DeepSeek is given the Cree sentence, its English gloss, and itwêwina dictionary
entries for each content word, and asked to classify it as literal / idiom /
metaphor / simile — purely from that evidence, with no model prediction passed
in. The model's reasoning chain (reasoning_content) is kept alongside the label
in a "reasoning" field, for auditing why a given label was chosen.

Resume-safe: labels (and reasoning) are checkpointed to a JSONL cache as they
come in, so an interrupted run picks up where it left off.

Run this on its own, then scripts/annotate/predict_pool.py on its own, then
scripts/annotate/agreement_eval.py to compare the two.

Usage:
  python scripts/annotate/deepseek_label_pool.py
  python scripts/annotate/deepseek_label_pool.py --limit 500   # smoke test
  python scripts/annotate/deepseek_label_pool.py --workers 16
"""

from __future__ import annotations
import argparse, json, os, sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from scripts.annotate._pool_utils import load_pool, POOL_FILE, EXCLUDE_SOURCE
from src.annotate.deepseek import client, MODEL_ID
from src.scrapers.itwewina import lookup_sentence, format_for_prompt

OUTPUT_FILE = "data/figurative/deepseek_labels.parquet"
CACHE_JSONL = "data/figurative/deepseek_labels_cache.jsonl"

LABELS = ["literal", "idiom", "metaphor", "simile"]

_SYSTEM_PROMPT = """\
You are annotating Plains Cree sentences for figurative language, using each \
sentence's English gloss and word-level dictionary glosses as evidence.

Critical context:
- The English gloss is a MEANING-FOR-MEANING translation, not word-for-word. \
It already conveys the intended sense, so a gloss that reads literally in \
English does NOT mean the Cree original is literal.
- Figurative language must be identified in the CREE STRUCTURE: compare the \
word-level Cree meanings (dictionary entries) against the whole-sentence \
English gloss. If the individual Cree words literally mean one thing but the \
sentence as a whole means something else, figurative language is present.
- idiom    — a fixed expression whose overall meaning cannot be composed from \
the meanings of its individual words.
- metaphor — an implicit, non-literal predication or conceptual transfer (e.g. \
a body-part or concrete term used to describe landscape, emotion, or an \
abstract state) where the word-level meanings do not match the sentence's \
real meaning.
- simile   — an explicit comparison, typically marked with tâpiskôc \
("like"/"as if").
- literal  — the word-level meanings and the sentence's real meaning match; \
no figurative device is present.

Respond with EXACTLY one word: literal / metaphor / idiom / simile\
"""


def _deepseek_annotate(prompt: str) -> tuple[str, str]:
    resp = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        # thinking with reasoning_effort=high can burn several hundred tokens
        # before emitting the one-word answer — too small a budget here means
        # the response gets cut off mid-thought with empty content (finish_reason
        # "length"), which silently looked like "literal" downstream.
        max_tokens=4096,
        stream=False,
        extra_body={"thinking": {"type": "enabled", "reasoning_effort": "high"}},
    )
    choice    = resp.choices[0]
    label     = (choice.message.content or "").strip().lower()
    reasoning = (getattr(choice.message, "reasoning_content", None) or "").strip()
    if label not in LABELS:
        print(f"  [deepseek] WARNING — invalid/truncated response "
              f"(finish_reason={choice.finish_reason!r}, content={label!r}); "
              f"defaulting to 'literal'")
        label = "literal"
    return label, reasoning


def _annotate_one(text_cree: str, text_en: str) -> tuple[str, str, str]:
    lookups = lookup_sentence(text_cree, verbose=False)
    prompt  = format_for_prompt(text_cree, text_en, lookups)
    label, reasoning = _deepseek_annotate(prompt)
    return text_cree, label, reasoning


def load_cache(cache_path: str) -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                cache[entry["text_cree"]] = {
                    "label":     entry["label"],
                    "reasoning": entry.get("reasoning", ""),
                }
    return cache


def annotate_pool(pool, cache_path: str, workers: int) -> dict[str, dict]:
    cache = load_cache(cache_path)
    todo  = pool[~pool["text_cree"].isin(cache.keys())]
    print(f"[deepseek] {len(cache):,} cached  |  {len(todo):,} to annotate  "
          f"(workers={workers})")

    if todo.empty:
        return cache

    done = 0
    with open(cache_path, "a", encoding="utf-8") as cache_f, \
         ThreadPoolExecutor(max_workers=workers) as pool_exec:
        futures = {
            pool_exec.submit(_annotate_one, row["text_cree"], row["text_en"]): row["text_cree"]
            for _, row in todo.iterrows()
        }
        for future in as_completed(futures):
            text_cree = futures[future]
            try:
                _, label, reasoning = future.result()
            except Exception as exc:
                # A real API/network failure, not a model answer — do NOT cache
                # a fallback label here, or the resume logic would treat this
                # sentence as done and never retry it. Just skip; it stays
                # "to do" and gets picked up on the next run.
                print(f"  [deepseek] error on {text_cree[:40]!r}: {exc} — will retry next run")
                done += 1
                continue

            cache[text_cree] = {"label": label, "reasoning": reasoning}
            cache_f.write(json.dumps({
                "text_cree": text_cree, "label": label, "reasoning": reasoning,
            }) + "\n")
            cache_f.flush()

            done += 1
            if done % 50 == 0:
                print(f"  [{done:,}/{len(todo):,}] annotated")

    return cache


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool",    default=POOL_FILE)
    p.add_argument("--exclude-source", default=EXCLUDE_SOURCE,
                   help="source_file value to exclude (default: bloomfield_1930; pass '' to include everything)")
    p.add_argument("--out",     default=OUTPUT_FILE)
    p.add_argument("--cache",   default=CACHE_JSONL)
    p.add_argument("--limit",   type=int, default=None,
                   help="Cap the number of sentences (for a smoke test)")
    p.add_argument("--workers", type=int, default=8,
                   help="Concurrent DeepSeek requests (default: 8)")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    pool = load_pool(args.pool, exclude_source=args.exclude_source or None, limit=args.limit)
    print(f"[pool] {len(pool):,} sentences (excluding source={args.exclude_source!r})")

    annotations = annotate_pool(pool, cache_path=args.cache, workers=args.workers)
    pool["deepseek_label"] = pool["text_cree"].map(lambda t: annotations[t]["label"])
    pool["reasoning"]      = pool["text_cree"].map(lambda t: annotations[t]["reasoning"])

    out_cols = ["paragraph_id", "sentence_id", "text_cree", "text_en",
                "deepseek_label", "reasoning"]
    pool[[c for c in out_cols if c in pool.columns]].to_parquet(args.out, index=False)

    print(f"\nLabel distribution:")
    print(pool["deepseek_label"].value_counts().to_string())
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
