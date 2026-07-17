"""
Ablation: how much does dictionary grounding — and translation itself —
contribute to the LLM figurative-language annotation procedure (Section 3.3)?

Three conditions, all scored against the same 228 footnote-verified gold
sentences used everywhere else in the paper:

  full              Cree + English gloss + itwewina dictionary entries.
                     This is the actual silver-annotation procedure
                     (scripts/annotate/deepseek_label_pool.py). No new API
                     calls: its scores are pulled directly from the already-
                     cached deepseek_on_gold.parquet run.
  no_dict           Cree + English gloss, no dictionary entries.
  no_dict_no_gloss  Cree sentence ONLY — no gloss, no dictionary. Tests
                     whether the model can do this from pretraining exposure
                     to Cree alone.

The two new conditions get their own adapted system prompt (so the model
isn't told it has evidence it wasn't actually given) but the same three-line
LABEL/EXPRESSION/MEANING output format as the main procedure, so parsing and
scoring are directly comparable. Each has its own resume-safe JSONL cache.

Usage:
  python scripts/annotate/ablation_llm_grounding.py
  python scripts/annotate/ablation_llm_grounding.py --limit 20   # smoke test
  python scripts/annotate/ablation_llm_grounding.py --workers 8
"""

from __future__ import annotations
import argparse, json, os, sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd

from src.annotate.deepseek import client, MODEL_ID
from src.annotate.figurative_prompt import parse_label, LABELS, SYSTEM_PROMPT, prompt_version
from src.scrapers.itwewina import lookup_sentence, format_for_prompt
from scripts.evals.eval_all import metrics_for, bootstrap_ci

GOLD_FILE       = "data/figurative/bloomfield_annotated.parquet"
FULL_PRED_FILE  = "data/figurative/deepseek_on_gold.parquet"
OUT_DIR         = "data/figurative"

NO_DICT_SYSTEM_PROMPT = """\
You are annotating Plains Cree sentences for figurative language. You are \
given the sentence's English gloss (its CONTEXTUAL MEANING — a \
meaning-for-meaning translation, not word-for-word). You do NOT have \
dictionary entries for the individual words in this condition — judge \
purely from the Cree sentence and its English gloss.

Decide whether the sentence contains figurative language:
  simile   — an explicit comparison, typically marked with tâpiskôc \
("like"/"as if").
  metaphor — an implicit, cross-domain comparison (describing one kind of \
thing in terms of a clearly different conceptual domain) without an \
explicit comparison marker.
  idiom    — a fixed, conventionalized multi-word expression whose overall \
meaning is not built compositionally from its parts.
  literal  — none of the above; the sentence's meaning follows directly and \
compositionally from its parts.

You may only conclude metaphor or idiom if you can point to one specific \
word or expression and state its figurative meaning concretely in a single \
clause. If you cannot do this, the label must be literal instead.

Respond in EXACTLY this three-line format and nothing else:
LABEL: literal / metaphor / idiom / simile
EXPRESSION: the specific word or expression that is figurative (or "none" if literal)
MEANING: a one-clause paraphrase of its figurative meaning (or "none" if literal)\
"""

BLIND_SYSTEM_PROMPT = """\
You are annotating Plains Cree sentences for figurative language. You are \
given ONLY the Cree sentence itself — no English translation and no \
dictionary entries. Decide, using only your own knowledge of Plains Cree \
(nêhiyawêwin), whether the sentence contains figurative language:
  simile   — an explicit comparison, typically marked with tâpiskôc \
("like"/"as if").
  metaphor — an implicit, cross-domain comparison without an explicit marker.
  idiom    — a fixed, conventionalized multi-word expression whose overall \
meaning is not built compositionally from its parts.
  literal  — none of the above.

You may only conclude metaphor or idiom if you can point to one specific \
word or expression and state its figurative meaning concretely in a single \
clause. If you cannot do this, the label must be literal instead.

Respond in EXACTLY this three-line format and nothing else:
LABEL: literal / metaphor / idiom / simile
EXPRESSION: the specific word or expression that is figurative (or "none" if literal)
MEANING: a one-clause paraphrase of its figurative meaning (or "none" if literal)\
"""


# The two targeted patches (idiom-glossed dictionary senses, constructional
# personification check) validated by this ablation are now baked directly
# into src/annotate/figurative_prompt.py's SYSTEM_PROMPT — it IS the patched
# prompt already, so this condition just uses it as-is. (Historical note: this
# used to re-derive the patch here via the same marker-replace as
# figurative_prompt.py; now that the patch lives upstream, doing that again
# here would silently double-apply it.)
PATCHED_SYSTEM_PROMPT = SYSTEM_PROMPT


def build_no_dict_prompt(cree: str, english: str) -> str:
    return (
        f"Cree sentence  : {cree}\n"
        f"English gloss  : {english}\n\n"
        "Given only the Cree sentence and its English gloss above (no dictionary evidence),\n"
        "is figurative language (metaphor, idiom, or simile) present in this sentence?"
    )


def build_blind_prompt(cree: str) -> str:
    return (
        f"Cree sentence  : {cree}\n\n"
        "Using only your own knowledge of Plains Cree, is figurative language\n"
        "(metaphor, idiom, or simile) present in this sentence?"
    )


def get_conditions(cache_suffix: str) -> dict:
    """cache_suffix namespaces cache files per model (e.g. '' for DeepSeek,
    '_meta_llama-3.3-70b-instruct' for an NVIDIA model), matching
    deepseek_eval_gold.py's slug convention so runs never collide."""
    return {
        "patched": {
            "system_prompt": PATCHED_SYSTEM_PROMPT,
            "build_prompt": lambda row: format_for_prompt(
                row["text_cree"], row["text_en"], lookup_sentence(row["text_cree"])),
            "cache_file": os.path.join(OUT_DIR, f"ablation_patched_cache{cache_suffix}.jsonl"),
        },
        "no_dict": {
            "system_prompt": NO_DICT_SYSTEM_PROMPT,
            "build_prompt": lambda row: build_no_dict_prompt(row["text_cree"], row["text_en"]),
            "cache_file": os.path.join(OUT_DIR, f"ablation_no_dict_cache{cache_suffix}.jsonl"),
        },
        "no_dict_no_gloss": {
            "system_prompt": BLIND_SYSTEM_PROMPT,
            "build_prompt": lambda row: build_blind_prompt(row["text_cree"]),
            "cache_file": os.path.join(OUT_DIR, f"ablation_no_dict_no_gloss_cache{cache_suffix}.jsonl"),
        },
    }


def make_deepseek_call_fn():
    def _call(system_prompt: str, user_prompt: str, retries: int = 2) -> str:
        last_exc = None
        for attempt in range(retries + 1):
            try:
                resp = client.chat.completions.create(
                    model=MODEL_ID,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                    max_tokens=4096,
                    stream=False,
                    extra_body={"thinking": {"type": "enabled", "reasoning_effort": "high"}},
                )
                content = (resp.choices[0].message.content or "").strip()
                label = parse_label(content)
                if label is None:
                    if attempt < retries:
                        continue
                    label = "literal"
                return label
            except Exception as exc:
                last_exc = exc
        raise last_exc
    return _call


def make_nvidia_call_fn(model_id: str, reasoning: bool):
    from src.annotate.llm import call_llm
    import time

    def _call(system_prompt: str, user_prompt: str, retries: int = 4) -> str:
        last_exc = None
        for attempt in range(retries + 1):
            try:
                _, content = call_llm(model_id, system_prompt, user_prompt,
                                       max_tokens=4096, reasoning=reasoning)
                label = parse_label(content)
                if label is None:
                    if attempt < retries:
                        continue
                    label = "literal"
                return label
            except Exception as exc:
                last_exc = exc
                if "429" in str(exc) and attempt < retries:
                    time.sleep(2 ** attempt * 2)  # 2s, 4s, 8s, 16s
        raise last_exc
    return _call


def load_cache(path: str, version: str | None = None) -> dict[str, str]:
    """version: prompt_version() of this condition's own system_prompt. Entries
    stamped with a different (or missing/pre-versioning) version are dropped —
    see the matching fix in deepseek_label_pool.py's load_cache()."""
    cache: dict[str, str] = {}
    stale = 0
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                if version is not None and rec.get("prompt_version") != version:
                    stale += 1
                    continue
                cache[rec["text_cree"]] = rec["label"]
    if stale:
        print(f"  [cache] {stale:,} entries in {path} used a different prompt "
              f"version — treating as not-yet-annotated and re-querying")
    return cache


def _build_and_call(cfg: dict, call_fn, row) -> str:
    """Runs build_prompt() (which, for 'patched'/'full'-style conditions, makes
    live itwewina network calls) INSIDE the worker thread, not the caller —
    submitting `call_fn(cfg["build_prompt"](row))` directly to executor.submit()
    would evaluate build_prompt(row) as an argument in the main thread before
    submission even happens, serializing every sentence's dictionary lookups
    (0.3s/word x ~228 sentences) with zero parallelism and zero progress output
    before a single DeepSeek call even starts."""
    prompt = cfg["build_prompt"](row)
    return call_fn(cfg["system_prompt"], prompt)


def run_condition(name: str, gold: pd.DataFrame, workers: int, conditions: dict, call_fn) -> dict:
    cfg = conditions[name]
    cache_path = cfg["cache_file"]
    version = prompt_version(cfg["system_prompt"])
    cache = load_cache(cache_path, version=version)
    todo = gold[~gold["text_cree"].isin(cache.keys())]
    print(f"[{name}] {len(cache):,} cached | {len(todo):,} to annotate (workers={workers})")

    if not todo.empty:
        print(f"  [{name}] submitting {len(todo):,} tasks to {workers} workers...")
        with open(cache_path, "a") as cache_f, ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(_build_and_call, cfg, call_fn, row): row["text_cree"]
                for _, row in todo.iterrows()
            }
            done = 0
            for fut in as_completed(futures):
                text_cree = futures[fut]
                try:
                    label = fut.result()
                except Exception as exc:
                    print(f"  [{name}] error on {text_cree[:40]!r}: {exc} — will retry next run")
                    continue
                cache[text_cree] = label
                cache_f.write(json.dumps({
                    "text_cree": text_cree, "label": label, "prompt_version": version,
                }) + "\n")
                cache_f.flush()
                done += 1
                if done % 5 == 0:
                    print(f"  [{name}] {done}/{len(todo)} annotated")

    y_true = gold["label"].tolist()
    y_pred = [cache[t] for t in gold["text_cree"]]
    row = {"condition": name, **metrics_for(y_true, y_pred), **bootstrap_ci(y_true, y_pred)}
    return row, y_pred


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gold-file", default=GOLD_FILE)
    p.add_argument("--out",       default=None)
    p.add_argument("--limit",     type=int, default=None, help="Cap sentences (smoke test)")
    p.add_argument("--workers",   type=int, default=8)
    p.add_argument("--nvidia-model", default=None, metavar="MODEL_ID",
                   help="Run this NVIDIA-hosted model instead of DeepSeek (e.g. "
                        "meta/llama-3.3-70b-instruct) — see src/annotate/llm.py")
    p.add_argument("--no-reasoning", action="store_true",
                   help="Don't send reasoning_effort — needed for plain instruct "
                        "models that don't support it (e.g. meta/llama-3.3-70b-instruct)")
    p.add_argument("--full-pred-file", default=None,
                   help="Override which cached predictions back the 'full' baseline row "
                        "(e.g. deepseek_on_gold_dictfix.parquet after the itwewina lookup fix)")
    p.add_argument("--skip-patched", action="store_true",
                   help="Don't run the patched-prompt condition (only no_dict/no_dict_no_gloss)")
    args = p.parse_args()

    model_label = args.nvidia_model or "deepseek"
    if args.nvidia_model:
        slug = model_label.replace("/", "_").replace(":", "_")
        cache_suffix = f"_{slug}"
        full_pred_file = args.full_pred_file or FULL_PRED_FILE.replace(".parquet", f"_{slug}.parquet")
        call_fn = make_nvidia_call_fn(args.nvidia_model, reasoning=not args.no_reasoning)
    else:
        cache_suffix = ""
        full_pred_file = args.full_pred_file or FULL_PRED_FILE
        call_fn = make_deepseek_call_fn()
    conditions = get_conditions(cache_suffix)
    args.out = args.out or os.path.join(OUT_DIR, f"ablation_llm_grounding{cache_suffix}.parquet")

    gold = pd.read_parquet(args.gold_file)
    gold = gold[gold["footnote_applies"] == True].dropna(subset=["text_cree", "text_en", "label"]).copy()
    gold["text_cree"] = gold["text_cree"].str.strip()
    gold["text_en"]   = gold["text_en"].str.strip()
    if args.limit:
        gold = gold.head(args.limit)
    print(f"[gold] model={model_label}  {len(gold):,} sentences — {gold['label'].value_counts().to_dict()}")

    rows = []

    if os.path.exists(full_pred_file):
        full_df = pd.read_parquet(full_pred_file)
        full_df["text_cree"] = full_df["text_cree"].str.strip()
        full_df = full_df[full_df["text_cree"].isin(gold["text_cree"])]
        y_true = full_df["label"].tolist()
        y_pred = full_df["deepseek_label"].tolist()
        rows.append({"condition": "full (dict + gloss)", **metrics_for(y_true, y_pred), **bootstrap_ci(y_true, y_pred)})
        print(f"[full] reused {len(full_df):,} cached predictions from {full_pred_file} — no new API calls")
    else:
        print(f"[full] {full_pred_file} not found — run scripts/annotate/deepseek_eval_gold.py "
              f"{'--nvidia-model ' + args.nvidia_model if args.nvidia_model else ''} first. Skipping this row.")

    to_run = [("no_dict", "no_dict (gloss only)"), ("no_dict_no_gloss", "no_dict_no_gloss (blind)")]
    if not args.skip_patched:
        to_run.insert(0, ("patched", "patched (dict + gloss, revised prompt)"))
    for name, label in to_run:
        row, _ = run_condition(name, gold, args.workers, conditions, call_fn)
        row["condition"] = label
        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_parquet(args.out, index=False)

    print(f"\n{'='*90}")
    print("| Condition | Accuracy | Macro F1 (95% CI) | Literal | Idiom | Metaphor | Simile |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        print(f"| {r['condition']} | {r['accuracy']:.3f} | "
              f"{r['macro_f1']:.3f} [{r['macro_f1_ci_lo']:.3f}, {r['macro_f1_ci_hi']:.3f}] | "
              f"{r['f1_literal']:.3f} | {r['f1_idiom']:.3f} | {r['f1_metaphor']:.3f} | {r['f1_simile']:.3f} |")
    print(f"{'='*90}")
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
