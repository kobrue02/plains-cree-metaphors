"""
Uses DeepSeek to read Bloomfield's footnotes on the ~420 footnoted paragraphs
and translate them into the 4-class figurative-language scheme, assigning a
label to each sentence in those paragraphs.
"""

from __future__ import annotations
import os, sys, json, time, re, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
from src.annotate.deepseek import client, MODEL_ID

TEXTS_CSV   = "data/bloomfield_texts.parquet"
SENTS_CSV   = "data/bloomfield_texts_sentences.parquet"
OUTPUT_CSV  = "data/figurative/bloomfield_annotated.parquet"
CACHE_JSONL = "data/figurative/annotation_cache.jsonl"

LABEL_MAP = {
    "literal": "literal", "none": "literal",
    "idiom": "idiom", "proverb": "idiom",
    "metaphor": "metaphor",
    "simile": "simile",
}

SYSTEM_PROMPT = """\
You are an expert linguist specialising in figurative language in Indigenous oral \
literature, with deep knowledge of Plains Cree (nêhiyawêwin) and Leonard Bloomfield's \
1934 "Plains Cree Texts". You annotate with precision and are conservative: \
when in doubt, prefer "literal"."""


def build_prompt(text_cree: str, text_en: str, footnote: str,
                 sentences: list[tuple[int, str, str]]) -> str:
    numbered = "\n".join(
        f"  [{i}] (Cree)    {cr}\n      (English) {en}"
        for i, cr, en in sentences
    )
    return f"""\
You are given a passage from Bloomfield's Plains Cree Texts and his own footnote \
commentary. Your PRIMARY task is to read the footnote and determine:
  1. Whether it describes figurative language (idiom, metaphor, simile, proverb, \
conventional non-literal expression, etc.).
  2. If so, which sentence number(s) it applies to.
  3. What class best fits.

Only fall back to reading the sentences directly if the footnote says nothing \
about figurative language.

Classes:
  "literal"  — plain, non-figurative language
  "metaphor" — implicit comparison or non-literal predication (X is Y)
  "simile"   — explicit comparison using a marker (tâpiskôc / "like" / "as if")
  "idiom"    — fixed expression, proverb, or conventional phrase whose meaning \
departs from its literal reading

Bloomfield's footnote (primary evidence):
  {footnote}

Full paragraph — Cree:
  {text_cree}

Full paragraph — English:
  {text_en}

Sentences to classify:
{numbered}

Return ONLY a JSON array, one object per sentence, in order:
[
  {{
    "sentence_num": <int>,
    "label": "<literal|metaphor|simile|idiom>",
    "footnote_applies": <true|false>,
    "rationale": "<one sentence, citing the footnote where relevant>"
  }},
  ...
]"""


def call_deepseek(prompt: str, retries: int = 2) -> list[dict]:
    for attempt in range(retries + 1):
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=4096,
            stream=False,
            response_format={"type": "json_object"},
            extra_body={"thinking": {"type": "enabled", "reasoning_effort": "high"}},
        )
        raw = (response.choices[0].message.content or "").strip()
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed
            for v in parsed.values():
                if isinstance(v, list):
                    return v
            raise ValueError("no array found")
        except (json.JSONDecodeError, ValueError):
            # Truncated output — salvage complete objects
            recovered = []
            for obj in re.findall(r'\{[^{}]+\}', raw):
                try:
                    recovered.append(json.loads(obj))
                except json.JSONDecodeError:
                    pass
            if recovered:
                return recovered
            if attempt < retries:
                time.sleep(1)
            else:
                raise


def load_cache() -> dict[int, list[dict]]:
    cache: dict[int, list[dict]] = {}
    if os.path.exists(CACHE_JSONL):
        with open(CACHE_JSONL) as f:
            for line in f:
                entry = json.loads(line)
                cache[entry["paragraph_id"]] = entry["annotations"]
    return cache


def save_to_cache(paragraph_id: int, annotations: list[dict]) -> None:
    with open(CACHE_JSONL, "a") as f:
        f.write(json.dumps({"paragraph_id": paragraph_id,
                            "annotations": annotations}) + "\n")


def main(limit: int | None = None) -> None:
    os.makedirs("data/figurative", exist_ok=True)

    texts = pd.read_parquet(TEXTS_CSV).reset_index(drop=True)
    sents = pd.read_parquet(SENTS_CSV)

    has_fn = texts["footnote_en"].notna() & (texts["footnote_en"].str.strip() != "")
    footnoted = texts[has_fn]
    if limit:
        footnoted = footnoted.head(limit)

    cache = load_cache()
    todo  = [pid for pid in footnoted.index if pid not in cache]
    print(f"Footnoted paragraphs: {len(footnoted)}  |  "
          f"Already cached: {len(footnoted) - len(todo)}  |  "
          f"To annotate: {len(todo)}")

    rows_out = []
    n_api = 0

    for pid, para in footnoted.iterrows():
        para_sents = sents[sents["paragraph_id"] == pid].sort_values("sentence_id")
        if para_sents.empty:
            continue

        sent_triples = [
            (i + 1, str(row["text_cree"]), str(row["text_en"]))
            for i, (_, row) in enumerate(para_sents.iterrows())
        ]

        if pid in cache:
            annotations = cache[pid]
        else:
            prompt = build_prompt(
                str(para.get("text_cree", "")),
                str(para.get("text_en", "")),
                str(para.get("footnote_en", "")),
                sent_triples,
            )
            try:
                annotations = call_deepseek(prompt)
                save_to_cache(pid, annotations)
                n_api += 1
                if n_api % 10 == 0:
                    print(f"  {n_api}/{len(todo)} API calls done...")
                time.sleep(0.3)
            except Exception as exc:
                print(f"  [para {pid}] FAILED — {exc}")
                continue

        ann_by_num = {a["sentence_num"]: a for a in annotations}
        for i, (_, sent_row) in enumerate(para_sents.iterrows()):
            ann      = ann_by_num.get(i + 1, {})
            raw_lbl  = ann.get("label", "literal").strip().lower()
            rows_out.append({
                "paragraph_id":    pid,
                "sentence_id":     sent_row["sentence_id"],
                "source":          para.get("source_file", ""),
                "text_cree":       sent_row["text_cree"],
                "text_en":         sent_row["text_en"],
                "label":           LABEL_MAP.get(raw_lbl, "literal"),
                "label_raw":       raw_lbl,
                "footnote_applies":ann.get("footnote_applies", False),
                "rationale":       ann.get("rationale", ""),
                "footnote_en":     str(para.get("footnote_en", "")),
            })

    out = pd.DataFrame(rows_out)
    out.to_parquet(OUTPUT_CSV, index=False)

    print(f"\nDone.  {len(out)} sentences  |  {n_api} new API calls")
    print(f"Label distribution:\n{out['label'].value_counts().to_string()}")
    print(f"Figurative (footnote_applies=True): "
          f"{out[out['footnote_applies']==True]['label'].value_counts().to_string()}")
    print(f"\nSaved → {OUTPUT_CSV}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most N paragraphs (for testing)")
    args = p.parse_args()
    main(limit=args.limit)
