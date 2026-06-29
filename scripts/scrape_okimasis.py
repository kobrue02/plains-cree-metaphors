"""
Extract Plains Cree–English parallel sentence pairs from:
  Okimāsis, Jean L. (2018). Cree: Language of the Plains.
  University of Regina Press. (Open Access PDF)

Two dominant patterns in the grammar chapters:

  Pattern A — inline (numbered examples):
    1. nikī-atoskān.                    I worked.
    2. kinōhtē-mīcison cī?             Do you want to eat?

  Pattern B — stacked:
    nikī-atoskānān otākosīhk niyanān.
    We (not you) worked yesterday.

Cree text is identified by SRO macron vowels: ā ē ī ō
(this book uses macrons; Bloomfield uses circumflexes — both are valid SRO)

Usage:
  python scripts/scrape_okimasis.py --pdf <path>
  python scripts/scrape_okimasis.py --pdf <path> --out data/sentences_okimasis.txt
"""

from __future__ import annotations
import argparse, re, subprocess, sys, os

# SRO macron vowels used in this textbook
_CREE_RE  = re.compile(r"[āēīōĀĒĪŌ]")
# Numbered example line: "  N. <text>   <translation>"
_NUM_LINE = re.compile(r"^\s{1,6}\d{1,2}\.\s+(.+)")
# Detect lines that are only English prose (long sentences, chapter headers, etc.)
_PROSE    = re.compile(
    r"^(Chapter|Note|Figure|The |This |In |For |When |One |Unlike|Although|Because|"
    r"While |Consider|There |Taking|Putting|Examples?:|These |As |So |That |It |If )", re.I
)
# Vocabulary-only lines (single Cree word/preverb followed by English gloss)
_VOCAB    = re.compile(r"^([^\s]{2,25})\s{3,}([^\s].{0,60})$")


def _has_cree(text: str) -> bool:
    return bool(_CREE_RE.search(text))


def _is_cree_dominant(text: str) -> bool:
    """True if the text is predominantly Cree (not just an English line with one accented name).

    Heuristic: at least 25% of whitespace-delimited tokens must contain a macron vowel,
    OR the text is very short (≤3 tokens) and at least one token has a macron.
    """
    tokens = text.split()
    if not tokens:
        return False
    cree_tokens = [t for t in tokens if _CREE_RE.search(t)]
    if len(tokens) <= 3:
        return len(cree_tokens) >= 1
    return len(cree_tokens) / len(tokens) >= 0.25


def _is_english(text: str) -> bool:
    return not _has_cree(text) and bool(text.strip())


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().strip('"').strip("'")


def _split_inline(line: str) -> tuple[str, str] | None:
    """Try to split 'Cree sentence     English translation' on a single line.

    The split heuristic: find the boundary where a run of ≥3 spaces separates
    Cree text (contains macrons) from English text (no macrons).
    """
    # Must contain a Cree indicator
    if not _has_cree(line):
        return None

    # Find the longest whitespace gap
    gaps = [(m.start(), m.end()) for m in re.finditer(r" {3,}", line)]
    if not gaps:
        return None

    # Try each gap from right to left; keep the first split where left=Cree, right=English
    for start, end in reversed(gaps):
        left  = line[:start].strip()
        right = line[end:].strip()
        if _has_cree(left) and right and not _has_cree(right) and len(right) > 4:
            return _clean(left), _clean(right)
    return None


def extract_pairs(text: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    lines  = text.splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()

        # Skip empty lines, prose headers, page numbers
        if not stripped or stripped.isdigit() or _PROSE.match(stripped):
            i += 1
            continue

        # ── Pattern A: numbered example on a single line ───────────────────
        m = _NUM_LINE.match(raw)
        if m:
            content = m.group(1).strip()
            pair = _split_inline(content)
            if pair:
                pairs.append(pair)
                i += 1
                continue
            # May be a numbered stacked pair — Cree on this line, English next
            if _is_cree_dominant(content) and not _PROSE.match(content):
                # Look ahead for the English line
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                if j < len(lines):
                    nxt = lines[j].strip()
                    if _is_english(nxt) and len(nxt) > 3 and not _NUM_LINE.match(lines[j]):
                        pairs.append((_clean(content), _clean(nxt)))
                        i = j + 1
                        continue
            i += 1
            continue

        # ── Pattern B: unnumbered Cree line followed by English line ────────
        if _is_cree_dominant(stripped) and not _PROSE.match(stripped):
            pair = _split_inline(stripped)
            if pair:
                pairs.append(pair)
                i += 1
                continue
            # Look ahead
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                nxt = lines[j].strip()
                if _is_english(nxt) and len(nxt) > 3 and not _has_cree(nxt):
                    # Sanity: Cree line should look like a sentence or phrase (not a header)
                    if len(stripped) > 5:
                        pairs.append((_clean(stripped), _clean(nxt)))
                        i = j + 1
                        continue

        i += 1

    return pairs


def dedup(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: set[str] = set()
    out  = []
    for cree, en in pairs:
        key = cree.lower()
        if key not in seen:
            seen.add(key)
            out.append((cree, en))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pdf", required=True, help="Path to Okimasis PDF")
    p.add_argument("--out", default="data/sentences_okimasis.txt",
                   help="Output file (Cree ||| English format)")
    p.add_argument("--min-cree-len", type=int, default=5,
                   help="Min characters in Cree sentence (default: 5)")
    args = p.parse_args()

    if not os.path.exists(args.pdf):
        sys.exit(f"PDF not found: {args.pdf}")

    print(f"Extracting text from {args.pdf} ...")
    result = subprocess.run(
        ["pdftotext", "-layout", args.pdf, "-"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if result.returncode != 0:
        sys.exit(f"pdftotext failed: {result.stderr}")

    print("Parsing parallel pairs ...")
    pairs = extract_pairs(result.stdout)
    pairs = dedup(pairs)

    # Filter junk
    def _keep(cree: str, en: str) -> bool:
        if len(cree) < args.min_cree_len:
            return False
        if not _has_cree(cree):
            return False
        # Syllable breakdown: Cree side has isolated single syllables with spaces
        # e.g. "tā ni si" or "pi mā ci ho win"
        if re.search(r"\b[a-zāēīō]{1,3} [a-zāēīō]{1,3} [a-zāēīō]{1,3}\b", cree):
            return False
        # Derivation formula lines
        if "+" in cree or "=" in cree:
            return False
        # Parenthetical gloss in Cree side indicates derivation example
        if re.search(r"\([a-z]", cree):
            return False
        # Grammar metalanguage on English side
        if re.search(r"\b(subject|verb|object|noun|stem|suffix|prefix)\b", en, re.I):
            return False
        # English side starts with a number (numbered list item from derivation chapter)
        if re.match(r"\d+\.", en):
            return False
        # Bibliography / metadata
        if re.search(r"(isbn|pm\d{3}|\d{4}-\d{4}|edition\.|new edition)", en, re.I):
            return False
        # English side looks like a prose sentence fragment (from front matter)
        if re.match(r"(Includes|As a result|Some changes|This edition)", en):
            return False
        # Author name line
        if re.match(r"Jean L\.", cree):
            return False
        return True

    pairs = [(c, e) for c, e in pairs if _keep(c, e)]

    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for cree, en in pairs:
            f.write(f"{cree} ||| {en}\n")

    print(f"\nExtracted {len(pairs):,} unique Cree–English pairs → {args.out}")
    print("\nSample:")
    for cree, en in pairs[:10]:
        print(f"  {cree[:55]:<55}  ||| {en[:50]}")


if __name__ == "__main__":
    main()
