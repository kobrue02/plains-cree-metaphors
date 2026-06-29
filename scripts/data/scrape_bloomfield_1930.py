"""
Extract paragraph-aligned Cree-English pairs from Bloomfield (1930)
"Sacred Stories of the Sweet Grass Cree".

Each of the 36 stories has a Cree text block followed immediately by its
English translation. Paragraphs within each story are aligned 1:1 by index.

Bloomfield (1930) uses his own phonetic notation — not SRO. The text is
preserved as-is (OCR artefacts included); the model sees the character
patterns, not the orthography.

Usage:
  python scripts/scrape_bloomfield_1930.py --pdf <path>
  python scripts/scrape_bloomfield_1930.py --pdf <path> --out data/sentences_bloomfield_1930.txt
"""

from __future__ import annotations
import argparse, re, sys, os

# Common English function words — used to detect English paragraphs
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

# Story header: "(N)  Title-in-Title-Case" — must be the whole line (no inline prose)
_STORY_HEADER = re.compile(r"^\s*\((\d{1,2})\)\s+([A-Z][^()]+?)\s*$")

# Footnote marker in Cree text (^ followed by optional word/number)
_FOOTNOTE_MARKER = re.compile(r"\^[\w]?")

# Standalone page number (line is only digits and whitespace)
_PAGE_NUMBER = re.compile(r"^\s*\d{1,4}\s*$")

# Translator/author note lines (appear at footnote boundary)
_FOOTNOTE_LINE = re.compile(
    r"^\s*\*\s*|^\s*\d+\s+|^\s*[Ww]ord\s|^\s*[Oo]r\s+read|^\s*[Pp]robably\s"
    r"|^\s*[Tt]ranslation\s|^\s*[Cc][Ff]\.|^\s*[Ii]\.e\.",
)


def _is_english_para(text: str) -> bool:
    """Return True if the paragraph is predominantly English prose."""
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    if len(words) < 5:
        return False
    en_count = sum(1 for w in words if w in _EN_WORDS)
    return en_count / len(words) >= 0.30


def _clean_para(text: str) -> str:
    """Strip footnote markers, page numbers, and excess whitespace."""
    text = _FOOTNOTE_MARKER.sub("", text)
    lines = [ln for ln in text.splitlines() if not _PAGE_NUMBER.match(ln)]
    lines = [ln for ln in lines if not _FOOTNOTE_LINE.match(ln)]
    text = " ".join(lines)
    return re.sub(r"\s+", " ", text).strip()


def _is_real_para(text: str, min_alpha_ratio: float = 0.40, min_words: int = 5) -> bool:
    """Return False for OCR-garbage or fragment paragraphs."""
    alpha = sum(1 for c in text if c.isalpha())
    if len(text) < 20 or alpha / max(len(text), 1) < min_alpha_ratio:
        return False
    return len(text.split()) >= min_words


def _split_paragraphs(text: str) -> list[str]:
    """Split on blank lines; filter empties and footnote-only paragraphs."""
    raw = re.split(r"\n\s*\n", text)
    paras = []
    for p in raw:
        p = p.strip()
        if not p:
            continue
        # Skip lines that are only page numbers or footnote lines
        lines = p.splitlines()
        content_lines = [
            ln for ln in lines
            if not _PAGE_NUMBER.match(ln) and not _FOOTNOTE_LINE.match(ln)
        ]
        if content_lines:
            paras.append(p)
    return paras


def extract_stories(full_text: str) -> list[dict]:
    """
    Find each story boundary, split into Cree + English paragraph blocks,
    and return a list of story dicts.
    """
    lines = full_text.splitlines()

    # Locate the "TEXTS AND TRANSLATIONS" section to ignore the ToC
    texts_start = 0
    for i, ln in enumerate(lines):
        if re.match(r"\s*TEXTS AND TRANSLATIONS\s*$", ln):
            texts_start = i
            break

    # Find story headers after the section start — require monotonically increasing
    # numbers to avoid false positives from inline numbered lists in the notes.
    story_spans: list[tuple[int, int, str]] = []  # (line_start, story_num, header_line)
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
        story_lines = lines[start + 1 : end]  # skip title line itself
        story_text = "\n".join(story_lines)

        paragraphs = _split_paragraphs(story_text)
        if not paragraphs:
            continue

        # Find where English translation begins
        split_idx = _find_en_split(paragraphs)

        # Cree section: also drop paragraphs that are actually English
        # (editorial notes, translator comments embedded in the Cree block)
        cree_paras = [
            _clean_para(p) for p in paragraphs[:split_idx]
            if not _is_english_para(p)
        ]
        en_paras = [_clean_para(p) for p in paragraphs[split_idx:]]

        stories.append({
            "number":     num,
            "title":      title,
            "cree_paras": [p for p in cree_paras if _is_real_para(p)],
            "en_paras":   [p for p in en_paras   if _is_real_para(p)],
        })

    return stories


def _find_en_split(paragraphs: list[str]) -> int:
    """
    Return the index of the first paragraph that is English.

    Look for two consecutive English paragraphs to avoid false positives
    from isolated English footnotes embedded in the Cree section.
    """
    for i in range(len(paragraphs) - 1):
        if _is_english_para(paragraphs[i]) and _is_english_para(paragraphs[i + 1]):
            return i
    # Fallback: single English paragraph at the end
    for i in range(len(paragraphs) - 1, -1, -1):
        if _is_english_para(paragraphs[i]):
            return i
    return len(paragraphs)  # all Cree (shouldn't happen)


def align_pairs(
    stories: list[dict],
    max_para_diff: int = 6,
) -> list[tuple[str, str]]:
    """
    Align Cree and English paragraphs 1:1 by position within each story.

    Only stories where abs(len(cree) - len(en)) <= max_para_diff are included;
    beyond that threshold, proportional merging produces wrong-content pairs
    because the Cree splits each dialogue line into its own paragraph while the
    English merges them.
    """
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


def dedup(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: set[str] = set()
    out = []
    for c, e in pairs:
        key = c.lower()
        if key not in seen:
            seen.add(key)
            out.append((c, e))
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--pdf", required=True, help="Path to Bloomfield 1930 PDF")
    p.add_argument(
        "--out",
        default="data/sentences_bloomfield_1930.txt",
        help="Output file (Cree ||| English format)",
    )
    p.add_argument(
        "--stats",
        action="store_true",
        help="Print per-story paragraph counts for alignment diagnostics",
    )
    p.add_argument(
        "--max-para-diff",
        type=int,
        default=6,
        help="Only include stories where |cree_paras - en_paras| <= this (default: 6)",
    )
    args = p.parse_args()

    if not os.path.exists(args.pdf):
        sys.exit(f"PDF not found: {args.pdf}")

    print(f"Extracting text from {args.pdf} ...")
    try:
        import fitz
        doc  = fitz.open(args.pdf)
        text = "\n".join(page.get_text("text", sort=True) for page in doc)
    except ImportError:
        sys.exit("pymupdf not installed — run: uv sync")

    print("Identifying story boundaries ...")
    stories = extract_stories(text)
    print(f"  Found {len(stories)} stories")

    if args.stats:
        print(f"\n{'#':<4} {'Title':<50} {'Cree':>5} {'EN':>5} {'use?':>5}")
        print("─" * 76)
        for s in stories:
            nc   = len(s["cree_paras"])
            ne   = len(s["en_paras"])
            diff = abs(nc - ne)
            use  = "yes" if diff <= args.max_para_diff else "skip"
            print(f"({s['number']:<2}) {s['title'][:48]:<48}  {nc:>4}  {ne:>4}  {use}")
        print()

    pairs = align_pairs(stories, max_para_diff=args.max_para_diff)
    pairs = dedup(pairs)

    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for cree, en in pairs:
            f.write(f"{cree} ||| {en}\n")

    print(f"\nExtracted {len(pairs):,} paragraph-aligned pairs → {args.out}")
    print("\nSample (first 5):")
    for cree, en in pairs[:5]:
        print(f"  Cree: {cree[:80]}")
        print(f"  EN:   {en[:80]}")
        print()


if __name__ == "__main__":
    main()
