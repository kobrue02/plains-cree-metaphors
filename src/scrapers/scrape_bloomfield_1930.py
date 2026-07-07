"""
Extract paragraph-aligned Cree-English pairs from Bloomfield (1930)
"Sacred Stories of the Sweet Grass Cree".

Reads from the Internet Archive DJVU plain-text export (P005409_djvu.txt),
which has much cleaner English translations than the PDF OCR.

Each of the 36 stories has a Cree text block followed immediately by its
English translation. Paragraphs within each story are aligned 1:1 by index.

Note on orthography: Bloomfield (1930) uses his own phonetic notation — not
modern SRO. The DJVU OCR renders macron-a (â) as 'd' throughout; the text
is preserved as-is since the model sees character patterns, not orthography.

Usage:
  python src/scrapers/scrape_bloomfield_1930.py --txt data/P005409_djvu.txt
  python src/scrapers/scrape_bloomfield_1930.py --txt data/P005409_djvu.txt --stats
"""

from __future__ import annotations
import argparse, re, sys, os

# common english function words used to classify paragraphs as english
_EN_WORDS = frozenset({
    "the", "he", "she", "was", "had", "his", "her", "and", "then", "to", "of",
    "that", "this", "it", "in", "for", "on", "at", "by", "with", "as", "said",
    "not", "but", "they", "them", "were", "from", "when", "all", "one", "have",
    "so", "what", "would", "who", "him", "their", "upon", "time", "once",
    "there", "which", "into", "out", "up", "an", "no", "its", "did", "will",
    "be", "are", "is", "a", "my", "me", "we", "you", "us", "now", "very",
    "then", "went", "came", "could", "told", "went", "come", "like", "do",
    "got", "went", "asked", "thought", "saw", "took", "made", "just", "back",
    "away", "man", "woman", "old", "new", "good", "great", "too", "if", "how",
})

# story header: "(N)  Title-in-Title-Case" — allow leading OCR noise characters before the (N)
_STORY_HEADER = re.compile(r"^[\s._>\'\-]*\((\d{1,2})\)\s+([A-Z][^()]+?)\s*$")

_PAGE_NUMBER = re.compile(r"^\s*\d{1,4}\s*$")

_FOOTNOTE_LINE = re.compile(
    r"^\s*\*\s*|^\s*\d+\s+|^\s*[Ww]ord\s|^\s*[Oo]r\s+read|^\s*[Pp]robably\s"
    r"|^\s*[Tt]ranslation\s|^\s*[Cc][Ff]\.|^\s*[Ii]\.e\.",
)

_RUNNING_HEADER = re.compile(
    r"(?i)(sacred\s+stories|sweet\s+grass\s+cree|national\s+museum"
    r"|department\s+of\s+mines|anthropological\s+ser)",
)


def _is_english_para(text: str) -> bool:
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    if len(words) < 5:
        return False
    en_count = sum(1 for w in words if w in _EN_WORDS)
    return en_count / len(words) >= 0.30


def _clean_para(text: str) -> str:
    lines = []
    for ln in text.splitlines():
        if _PAGE_NUMBER.match(ln):
            continue
        if _FOOTNOTE_LINE.match(ln):
            continue
        if _RUNNING_HEADER.search(ln):
            continue
        lines.append(ln)
    text = " ".join(lines)
    # remove footnote superscript markers embedded mid-word (e.g. ^2, ^a)
    text = re.sub(r"\^[\w]?", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _is_real_para(text: str, min_alpha_ratio: float = 0.40, min_words: int = 5) -> bool:
    alpha = sum(1 for c in text if c.isalpha())
    if len(text) < 20 or alpha / max(len(text), 1) < min_alpha_ratio:
        return False
    return len(text.split()) >= min_words


def _split_paragraphs(text: str) -> list[str]:
    raw = re.split(r"\n\s*\n", text)
    paras = []
    for p in raw:
        p = p.strip()
        if not p:
            continue
        lines = p.splitlines()
        content_lines = [
            ln for ln in lines
            if not _PAGE_NUMBER.match(ln)
            and not _FOOTNOTE_LINE.match(ln)
            and not _RUNNING_HEADER.search(ln)
        ]
        if content_lines:
            paras.append(p)
    return paras


def _merge_fragments(paras: list[str]) -> list[str]:
    """Re-join line-level fragments caused by DJVU blank lines after every physical line."""
    merged: list[str] = []
    buf = ""
    for p in paras:
        if buf:
            # leading '. ' means OCR split the period onto the next line
            if re.match(r"^\.\s+[A-Z]", p):
                merged.append(buf.strip())
                buf = p.lstrip(". ")
            else:
                buf = buf.rstrip() + " " + p.lstrip()
        else:
            buf = p
        tail = buf.rstrip().rstrip("\"'”’")
        if tail and tail[-1] in ".!?" and len(buf.split()) >= 8:
            merged.append(buf.strip())
            buf = ""
    if buf.strip():
        merged.append(buf.strip())
    return merged


def _find_en_split(paragraphs: list[str]) -> int:
    """Return the index where the English block begins, requiring two consecutive English paragraphs to avoid false triggers on Cree-embedded footnotes."""
    for i in range(len(paragraphs) - 1):
        if _is_english_para(paragraphs[i]) and _is_english_para(paragraphs[i + 1]):
            return i
    # fallback: single english paragraph near the end
    for i in range(len(paragraphs) - 1, -1, -1):
        if _is_english_para(paragraphs[i]):
            return i
    return len(paragraphs)


def _extract_stories(full_text: str) -> list[dict]:
    lines = full_text.splitlines()

    # skip past table of contents / introduction to the actual texts
    texts_start = 0
    for i, ln in enumerate(lines):
        if re.match(r"\s*TEXTS AND TRANSLATIONS\s*$", ln):
            texts_start = i
            break

    # require monotonically increasing story numbers to avoid false positives from garbled ToC entries
    story_spans: list[tuple[int, int, str]] = []
    last_num = 0
    for i, ln in enumerate(lines[texts_start:], start=texts_start):
        m = _STORY_HEADER.match(ln)
        if m:
            num = int(m.group(1))
            if num > last_num and num <= 36:
                story_spans.append((i, num, ln.strip()))
                last_num = num

    stories = []
    for idx, (start, num, title) in enumerate(story_spans):
        end = story_spans[idx + 1][0] if idx + 1 < len(story_spans) else len(lines)
        story_text = "\n".join(lines[start + 1 : end])

        paragraphs = _split_paragraphs(story_text)
        if not paragraphs:
            continue

        split_idx = _find_en_split(paragraphs)

        cree_paras = [
            _clean_para(p) for p in paragraphs[:split_idx]
            if not _is_english_para(p)
        ]
        en_paras_raw = [_clean_para(p) for p in paragraphs[split_idx:]]
        en_paras = _merge_fragments([p for p in en_paras_raw if _is_real_para(p)])

        stories.append({
            "number":     num,
            "title":      title,
            "cree_paras": [p for p in cree_paras if _is_real_para(p)],
            "en_paras":   [p for p in en_paras   if _is_real_para(p)],
        })

    return stories


def _align_pairs(stories: list[dict], max_para_diff: int = 6) -> list[tuple[str, str]]:
    """Align Cree and English paragraphs 1:1 by position, skipping stories whose paragraph counts diverge too much."""
    pairs = []
    for story in stories:
        cree = story["cree_paras"]
        en   = story["en_paras"]
        if not cree or not en:
            continue
        if abs(len(cree) - len(en)) > max_para_diff:
            continue
        for c, e in zip(cree, en):
            c = re.sub(r"\s+", " ", c).strip()
            e = re.sub(r"\s+", " ", e).strip()
            if _is_real_para(c) and _is_real_para(e):
                pairs.append((c, e))
    return pairs


def _dedup(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: set[str] = set()
    out = []
    for c, e in pairs:
        key = c.lower()
        if key not in seen:
            seen.add(key)
            out.append((c, e))
    return out


def extract(txt_path: str, max_para_diff: int = 10) -> list[tuple[str, str]]:
    """Extract deduplicated (cree, english) paragraph-aligned pairs from the Bloomfield 1930 DJVU text."""
    with open(txt_path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    stories = _extract_stories(text)
    pairs = _align_pairs(stories, max_para_diff=max_para_diff)
    return _dedup(pairs)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--txt", required=True,
                   help="Path to P005409_djvu.txt (Internet Archive DJVU export)")
    p.add_argument("--out", default="data/sentences_bloomfield_1930.txt",
                   help="Output file (Cree ||| English format)")
    p.add_argument("--stats", action="store_true",
                   help="Print per-story paragraph counts for alignment diagnostics")
    p.add_argument("--max-para-diff", type=int, default=10,
                   help="Only include stories where |cree_paras - en_paras| <= this (default: 10)")
    args = p.parse_args()

    if not os.path.exists(args.txt):
        sys.exit(f"File not found: {args.txt}")

    print(f"Extracting from {args.txt} ...")

    with open(args.txt, encoding="utf-8", errors="replace") as f:
        text = f.read()
    stories = _extract_stories(text)

    if args.stats:
        print(f"  Found {len(stories)} stories\n")
        print(f"{'#':<4} {'Title':<50} {'Cree':>5} {'EN':>5} {'diff':>5} {'use?':>5}")
        print("─" * 78)
        for s in stories:
            nc   = len(s["cree_paras"])
            ne   = len(s["en_paras"])
            diff = abs(nc - ne)
            use  = "yes" if diff <= args.max_para_diff else "skip"
            print(f"({s['number']:<2}) {s['title'][:48]:<48}  {nc:>4}  {ne:>4}  {diff:>4}  {use}")
        print()

    pairs = _align_pairs(stories, max_para_diff=args.max_para_diff)
    pairs = _dedup(pairs)

    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for cree, en in pairs:
            f.write(f"{cree} ||| {en}\n")

    print(f"Extracted {len(pairs):,} paragraph-aligned pairs → {args.out}")
    print("\nSample (first 5):")
    for cree, en in pairs[:5]:
        print(f"  Cree: {cree[:80]}")
        print(f"  EN:   {en[:80]}")
        print()


if __name__ == "__main__":
    main()
