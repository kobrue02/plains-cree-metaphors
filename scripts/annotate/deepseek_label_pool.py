"""
Label the sentence pool with DeepSeek only (no model involved).

For every Cree sentence in the pool (excluding the 1930 manuscript by
default), DeepSeek is given the Cree sentence, its English gloss, and
itwêwina dictionary entries for each content word, and classifies it as
literal / idiom / metaphor / simile from that evidence alone. The reasoning
chain is kept alongside the label in a "reasoning" field, for auditing why a
given label was chosen.

Gold and silver are meant to be disjoint (silver = "the un-footnoted majority
of the corpus" per the writeup) — gold sentences are excluded here so silver
never re-annotates something that already has a better-grounded label.

Resume-safe: labels (and reasoning) are checkpointed to a JSONL cache as they
come in, so an interrupted run picks up where it left off.

Run this on its own, then scripts/annotate/predict_pool.py, then
scripts/annotate/agreement_eval.py to compare the two.

Usage:
  python scripts/annotate/deepseek_label_pool.py
  python scripts/annotate/deepseek_label_pool.py --limit 500   # smoke test
  python scripts/annotate/deepseek_label_pool.py --workers 16
"""

from __future__ import annotations
import argparse, json, os, sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from scripts.annotate._pool_utils import load_pool, POOL_FILE, EXCLUDE_SOURCE
from src.annotate.deepseek import client, MODEL_ID
from src.annotate.figurative_prompt import LABELS, SYSTEM_PROMPT, parse_label, prompt_version
from src.scrapers.itwewina import lookup_sentence, format_for_prompt

OUTPUT_FILE = "data/figurative/deepseek_labels.parquet"
CACHE_JSONL = "data/figurative/deepseek_labels_cache.jsonl"
GOLD_FILE   = "data/figurative/bloomfield_annotated.parquet"


def _deepseek_annotate(prompt: str, retries: int = 3) -> tuple[str, str]:
    # thinking with reasoning_effort=high can burn several thousand tokens
    # before emitting the one-word answer, scaling with sentence length (the
    # per-word walkthrough means longer sentences need more) — too small a
    # budget here means the response gets cut off mid-thought with empty
    # content (finish_reason "length"). max_tokens=8192 has been sufficient in
    # practice for the original SYSTEM_PROMPT (the current production silver
    # cache has zero empty-response entries), but is not guaranteed to stay
    # sufficient if the prompt grows — if this starts firing, raise it the
    # same way src/annotate/llm.py's call_llm default was raised to 16384.
    for attempt in range(retries + 1):
        resp = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=8192,
            stream=False,
            extra_body={"thinking": {"type": "enabled", "reasoning_effort": "high"}},
        )
        choice    = resp.choices[0]
        content   = (choice.message.content or "").strip()
        reasoning = (getattr(choice.message, "reasoning_content", None) or "").strip()

        label = parse_label(content)
        if label is not None:
            # keep the structured EXPRESSION/MEANING lines alongside the reasoning
            # chain for auditing — e.g. checking whether a metaphor/idiom call
            # actually named something concrete, per the "must name a specific
            # expression" constraint above.
            full_reasoning = f"{reasoning}\n\n--- final answer ---\n{content}".strip()
            return label, full_reasoning
        print(f"  [deepseek] empty/unparseable response on attempt {attempt + 1}/{retries + 1} "
              f"(finish_reason={choice.finish_reason!r}, content={content[:80]!r})"
              + ("; retrying" if attempt < retries else "; giving up"))
    # Do NOT default to "literal" here — that would get written into the cache
    # as if it were a genuine result, permanently defeating annotate_pool()'s
    # resume-safe retry (a sentence the cache thinks is "done" never gets
    # revisited). Raising lets the caller's existing "don't cache, retry next
    # run" exception handling take over instead.
    raise RuntimeError(f"deepseek: empty/unparseable response after {retries + 1} attempts")


def _annotate_one(text_cree: str, text_en: str, annotate_fn=_deepseek_annotate) -> tuple[str, str, str]:
    """annotate_fn: (prompt: str) -> (label, reasoning). Defaults to DeepSeek;
    pass a different provider's callable (see src/annotate/llm.py) to reuse
    this same pool-annotation machinery for another model."""
    lookups = lookup_sentence(text_cree, verbose=False)
    prompt  = format_for_prompt(text_cree, text_en, lookups)
    label, reasoning = annotate_fn(prompt)
    return text_cree, label, reasoning


def load_cache(cache_path: str, version: str | None = None) -> dict[str, dict]:
    """version: current prompt_version(SYSTEM_PROMPT). Entries stamped with a
    different (or missing/pre-versioning) prompt_version are dropped — they
    were annotated under a prompt that no longer exists, so treating them as
    "already done" would silently freeze in stale labels after a prompt patch."""
    cache: dict[str, dict] = {}
    stale = 0
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                if version is not None and entry.get("prompt_version") != version:
                    stale += 1
                    continue
                cache[entry["text_cree"]] = {
                    "label":     entry["label"],
                    "reasoning": entry.get("reasoning", ""),
                }
    if stale:
        print(f"  [cache] {stale:,} entries in {cache_path} used a different prompt "
              f"version — treating as not-yet-annotated and re-querying")
    return cache


def annotate_pool(pool, cache_path: str, workers: int, annotate_fn=_deepseek_annotate) -> dict[str, dict]:
    """annotate_fn: (prompt: str) -> (label, reasoning), forwarded to _annotate_one
    for every sentence — swap it to reuse this pool-annotation loop for another
    provider/model (see src/annotate/llm.py)."""
    version = prompt_version(SYSTEM_PROMPT)
    cache = load_cache(cache_path, version=version)
    todo  = pool[~pool["text_cree"].isin(cache.keys())]
    print(f"[annotate] {len(cache):,} cached  |  {len(todo):,} to annotate  "
          f"(workers={workers})")

    if todo.empty:
        return cache

    with open(cache_path, "a", encoding="utf-8") as cache_f, \
         ThreadPoolExecutor(max_workers=workers) as pool_exec:
        futures = {
            pool_exec.submit(_annotate_one, row["text_cree"], row["text_en"], annotate_fn): row["text_cree"]
            for _, row in todo.iterrows()
        }
        pbar = tqdm(as_completed(futures), total=len(futures), desc="annotating", unit="sentence")
        for future in pbar:
            text_cree = futures[future]
            try:
                _, label, reasoning = future.result()
            except Exception as exc:
                # A real API/network failure, not a model answer — do NOT cache
                # a fallback label here, or the resume logic would treat this
                # sentence as done and never retry it. Just skip; it stays
                # "to do" and gets picked up on the next run.
                pbar.write(f"  [annotate] error on {text_cree[:40]!r}: {exc} — will retry next run")
                continue

            cache[text_cree] = {"label": label, "reasoning": reasoning}
            cache_f.write(json.dumps({
                "text_cree": text_cree, "label": label, "reasoning": reasoning,
                "prompt_version": version,
            }) + "\n")
            cache_f.flush()

    return cache


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool",    default=POOL_FILE)
    p.add_argument("--exclude-source", default=EXCLUDE_SOURCE,
                   help="source_file value to exclude (default: bloomfield_1930; pass '' to include everything)")
    p.add_argument("--gold-file", default=GOLD_FILE,
                   help="Gold annotations to exclude from silver (default: %(default)s; "
                        "pass '' to disable this exclusion)")
    p.add_argument("--out",     default=OUTPUT_FILE)
    p.add_argument("--cache",   default=CACHE_JSONL)
    p.add_argument("--limit",   type=int, default=None,
                   help="Cap the number of sentences (for a smoke test)")
    p.add_argument("--workers", type=int, default=8,
                   help="Concurrent DeepSeek requests (default: 8)")
    p.add_argument("--nvidia-model", default=None, metavar="MODEL_ID",
                   help="Run this NVIDIA-hosted model instead of DeepSeek "
                        "(e.g. qwen/qwen3.5-122b-a10b) — see src/annotate/llm.py")
    p.add_argument("--no-reasoning", action="store_true",
                   help="Don't send reasoning_effort — needed for plain instruct "
                        "models that don't support it (e.g. meta/llama-3.3-70b-instruct)")
    args = p.parse_args()

    annotate_kwargs = {}
    if args.nvidia_model:
        from src.annotate.llm import make_annotate_fn
        slug = args.nvidia_model.replace("/", "_").replace(":", "_")
        args.out   = args.out   if args.out   != OUTPUT_FILE  else OUTPUT_FILE.replace(".parquet", f"_{slug}.parquet")
        args.cache = args.cache if args.cache != CACHE_JSONL else CACHE_JSONL.replace(".jsonl", f"_{slug}.jsonl")
        annotate_kwargs["annotate_fn"] = make_annotate_fn(args.nvidia_model, reasoning=not args.no_reasoning)
        print(f"[model] {args.nvidia_model}  (out={args.out}, cache={args.cache})")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    pool = load_pool(args.pool, exclude_source=args.exclude_source or None)
    print(f"[pool] {len(pool):,} sentences (excluding source={args.exclude_source!r})")

    if args.gold_file and os.path.exists(args.gold_file):
        import pandas as pd
        gold_texts = set(pd.read_parquet(args.gold_file)["text_cree"].dropna().str.strip())
        before = len(pool)
        pool = pool[~pool["text_cree"].isin(gold_texts)]
        print(f"[pool] excluded {before - len(pool):,} sentences already gold-annotated "
              f"({args.gold_file}) — {len(pool):,} remain")

    if args.limit:
        pool = pool.head(args.limit)

    annotations = annotate_pool(pool, cache_path=args.cache, workers=args.workers, **annotate_kwargs)
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
