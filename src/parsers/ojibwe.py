"""
Parser for "Ojibwa Texts" (Jones/Michelson 1917, Vol. VII Part I), a facing-page
dual-language edition where the Internet Archive djvu OCR dump alternates Ojibwe (even)
and English (odd) pages in sequence. Outputs a CSV with columns story, para_idx,
text_ojibwe, text_en.
"""

from __future__ import annotations
import re
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field

# strong ojibwe markers: vowel-apostrophe clusters, affricates tc/dc, subscript digits, /u /i encodings
_OJI_RE = re.compile(
    r"[aeiou]['''][a-z]"          # vowel-apostrophe (a'pi, o'o')
    r"|[a-z][0-9]{1,2}[a-z]"     # subscript numbers: a11, Ii8i
    r"|\b(tc|dc)[a-z]"            # Ojibwe affricates
    r"|[/'\\][uia][/\\]"          # /u /i encodings
    r"|ugri|midac|misa|kaga|cigwa|anlc|mldac|ugrl|uglwab",
)

_EN_WORDS = re.compile(
    r"\b(the|was|and|that|had|her|him|with|this|then|they|were|have|from|"
    r"upon|when|said|told|thus|what|where|some|did|now|him|one|who|"
    r"verily|therefore|hither|thereof)\b",
    re.IGNORECASE,
)

_PAGE_NUM_RE = re.compile(r"^[IVXLC]+$|^\d+$")

# story-section headers accept both arabic and lower-case roman numerals (i, ii, iii...)
_HEADER_RE = re.compile(
    r"^[ivxlc\d]+\."          # "i." or "2."
    r"\s+[A-Z][A-Z\s,'\-]{4,}"  # title in caps
    r"[.\d]*\s*$",             # optional trailing footnote ref or period
    re.IGNORECASE,
)

# series/chapter dividers (not story headers)
_SERIES_RE = re.compile(r"^(SERIES|PART|I\s*[.—-]|II\s*[.—-])", re.IGNORECASE)

# footnote: short line starting with a digit (e.g. "1 Saga'a'man, 'when you go out'")
_FOOTNOTE_RE = re.compile(r"^\d+\s+\S")

def _ojibwe_score(text: str) -> int:
    return len(_OJI_RE.findall(text))

def _english_score(text: str) -> int:
    return len(_EN_WORDS.findall(text))

def classify_block(raw: str) -> str:
    """Return 'ojibwe' | 'english' | 'header' | 'meta' | 'footnote'."""
    text = " ".join(raw.split())
    if not text:
        return "meta"

    if _PAGE_NUM_RE.match(text):
        return "meta"

    if _SERIES_RE.match(text):
        return "meta"

    if re.match(r"^[A-Z\s.,'\-]{3,60}$", text) and len(text.split()) <= 8:
        return "meta"

    if _HEADER_RE.match(text):
        return "header"

    if _FOOTNOTE_RE.match(text) and len(text) < 300:
        return "footnote"

    oji = _ojibwe_score(text)
    eng = _english_score(text)
    if oji == 0 and eng == 0:
        return "meta"
    return "ojibwe" if oji >= eng else "english"

_MARGIN_NUM_RE = re.compile(r"(?m)^\s*\d+\s{2,}")  # e.g. "5   " at line start

def clean_block(raw: str) -> str:
    """Normalize OCR artifacts in a paragraph."""
    text = _MARGIN_NUM_RE.sub("", raw)
    text = re.sub(r"  +", " ", text)
    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    return text.strip()

def strip_header(text: str) -> str:
    """Normalize a story header to a stable key."""
    t = re.sub(r"\s+", " ", text).strip()
    t = re.sub(r"\d+\s*$", "", t).strip().rstrip(".")
    return t.upper()

@dataclass
class Story:
    title: str
    ojibwe: list[str] = field(default_factory=list)
    english: list[str] = field(default_factory=list)

def parse(path: str | Path) -> list[Story]:
    raw = Path(path).read_text(encoding="utf-8", errors="replace")

    # skip preface/front matter; texts begin at the first story header
    match = re.search(
        r"(?m)^i\.?\s+THE\s+BIRTH\s+OF\s+NANABUSHU",
        raw, re.IGNORECASE,
    )
    if match:
        raw = raw[match.start():]

    blocks = re.split(r"\n{2,}", raw)

    stories: dict[str, Story] = {}
    current_title = "__preamble__"
    stories[current_title] = Story(title=current_title)

    for block in blocks:
        label = classify_block(block)
        if label == "header":
            key = strip_header(block)
            if key not in stories:
                stories[key] = Story(title=key)
            current_title = key
            continue
        if label in ("meta", "footnote"):
            continue
        text = clean_block(block)
        if not text or len(text) < 10:
            continue
        if label == "ojibwe":
            stories[current_title].ojibwe.append(text)
        elif label == "english":
            stories[current_title].english.append(text)

    stories.pop("__preamble__", None)
    return list(stories.values())

def to_parquet(stories: list[Story], out_path: str | Path) -> None:
    import pandas as pd
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for story in stories:
        pairs = list(zip(story.ojibwe, story.english))
        for i, (oji, eng) in enumerate(pairs):
            rows.append({"story": story.title, "para_idx": i,
                         "text_ojibwe": oji, "text_en": eng})
    pd.DataFrame(rows).to_parquet(out_path, index=False)

def to_parallel(stories: list[Story], out_path: str | Path) -> None:
    """Write src ||| tgt format for TLM fine-tuning."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for story in stories:
            for oji, eng in zip(story.ojibwe, story.english):
                f.write(f"{oji} ||| {eng}\n")

def to_parallel_df(stories: list[Story]) -> "pd.DataFrame":
    """Return a DataFrame with text_cree and text_en columns (no file I/O)."""
    import pandas as pd
    rows = []
    for story in stories:
        for oji, eng in zip(story.ojibwe, story.english):
            rows.append({"text_cree": oji, "text_en": eng})
    return pd.DataFrame(rows)

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Parse Jones/Michelson Ojibwa Texts into aligned paragraph pairs.")
    p.add_argument("--input",   default="data/ojibwatextscoll07jonerich_djvu.txt")
    p.add_argument("--csv",     default="data/ojibwe_texts_aligned.parquet")
    p.add_argument("--parallel", default=None,
                   help="Also write src ||| tgt file for TLM (e.g. data/ojibwe_sentences.txt)")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    print(f"Parsing {args.input} ...")
    stories = parse(args.input)

    total_pairs = sum(min(len(s.ojibwe), len(s.english)) for s in stories)
    print(f"Found {len(stories)} stories  →  {total_pairs} aligned paragraph pairs")

    if args.verbose:
        for s in stories:
            n = min(len(s.ojibwe), len(s.english))
            print(f"  {s.title[:60]:<60}  oji={len(s.ojibwe)}  en={len(s.english)}  pairs={n}")

    to_parquet(stories, args.csv)
    print(f"Saved parquet  → {args.csv}")

    if args.parallel:
        to_parallel(stories, args.parallel)
        print(f"Saved TLM  → {args.parallel}")
