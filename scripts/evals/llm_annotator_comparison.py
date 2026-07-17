"""
Canonical LLM-annotator-vs-gold comparison table — the numbers for "how well
does each LLM do the dictionary-grounded annotation procedure, judged against
Bloomfield's 228 footnote-verified gold sentences, under the CURRENT
(patched) production prompt."

This is the single source of truth for that comparison. Do NOT use the older
deepseek_on_gold*.parquet / deepseek_on_gold*_summary.parquet files for this —
those were produced before the idiom-gloss/constructional prompt patch
existed (src/annotate/figurative_prompt.py) and are frozen at pre-patch
numbers; they've been moved to data/figurative/_stale_pre_patch/. Every
number here is instead read straight from each model's
"patched (dict + gloss, revised prompt)" row in
scripts/annotate/ablation_llm_grounding.py's own output — the same script
used for the grounding ablation (paper Section 6.1) — so there is exactly one
code path that ever computes an LLM's gold performance under the current
prompt, and both the ablation table and this comparison table read from it.

Requires each model's ablation_llm_grounding*.parquet to already exist (run
scripts/annotate/ablation_llm_grounding.py --nvidia-model <id> for each model
first). Models missing that file are skipped with a note, not silently
dropped from consideration.

Usage:
  python scripts/evals/llm_annotator_comparison.py
"""

from __future__ import annotations
import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd

GOLD_FILE   = "data/figurative/bloomfield_annotated.parquet"
OUTPUT_FILE = "data/figurative/llm_annotator_comparison.parquet"

# label -> ablation_llm_grounding output file. DeepSeek's has no model suffix
# (see ablation_llm_grounding.py's cache_suffix logic).
MODEL_FILES = {
    "DeepSeek":            "data/figurative/ablation_llm_grounding.parquet",
    "Llama-3.3-70B":       "data/figurative/ablation_llm_grounding_meta_llama-3.3-70b-instruct.parquet",
    "Mistral-Medium-3.5":  "data/figurative/ablation_llm_grounding_mistralai_mistral-medium-3.5-128b.parquet",
    "GPT-OSS-120B":        "data/figurative/ablation_llm_grounding_openai_gpt-oss-120b.parquet",
    "Qwen3.5-122B-A10B":   "data/figurative/ablation_llm_grounding_qwen_qwen3.5-122b-a10b.parquet",
}

REPORT_COLS = [
    "macro_f1", "macro_f1_ci_lo", "macro_f1_ci_hi",
    "f1_literal", "f1_idiom", "f1_metaphor", "f1_simile",
]


def main() -> None:
    n_gold = len(pd.read_parquet(GOLD_FILE).pipe(lambda d: d[d["footnote_applies"] == True]))

    rows = []
    for model, path in MODEL_FILES.items():
        if not os.path.exists(path):
            print(f"[skip] {model}: {path} not found — run ablation_llm_grounding.py for it first")
            continue
        df = pd.read_parquet(path)
        patched = df[df["condition"].str.startswith("patched")]
        if patched.empty:
            print(f"[skip] {model}: {path} has no 'patched' condition row")
            continue
        row = patched.iloc[0]
        rows.append({"model": model, "n": n_gold, **{c: row[c] for c in REPORT_COLS}})

    out = pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    out.to_parquet(OUTPUT_FILE, index=False)

    print(f"\n{'model':20s} {'macro F1':>10s}  {'95% CI':>15s}  {'lit':>5s} {'idiom':>6s} {'meta':>5s} {'simile':>6s}")
    for _, r in out.iterrows():
        ci = f"[{r['macro_f1_ci_lo']:.3f},{r['macro_f1_ci_hi']:.3f}]"
        print(f"{r['model']:20s} {r['macro_f1']:>10.3f}  {ci:>15s}  "
              f"{r['f1_literal']:>5.2f} {r['f1_idiom']:>6.2f} {r['f1_metaphor']:>5.2f} {r['f1_simile']:>6.2f}")
    print(f"\nSaved → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
