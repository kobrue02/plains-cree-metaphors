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
  python src/parsers/okimasis.py --pdf <path>
  python src/parsers/okimasis.py --pdf <path> --out data/sentences_okimasis.txt
"""

from __future__ import annotations
import argparse, re, sys, os

_CREE_RE  = re.compile(r"[āēīōĀĒĪŌ]")
_NUM_LINE = re.compile(r"^\s{1,6}\d{1,2}\.\s+(.+)")
_PROSE    = re.compile(
    r"^(Chapter|Note|Figure|The |This |In |For |When |One |Unlike|Although|Because|"
    r"While |Consider|There |Taking|Putting|Examples?:|These |As |So |That |It |If )", re.I
)
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
    if not _has_cree(line):
        return None

    gaps = [(m.start(), m.end()) for m in re.finditer(r" {3,}", line)]
    if not gaps:
        return None

    for start, end in reversed(gaps):
        left  = line[:start].strip()
        right = line[end:].strip()
        if _has_cree(left) and right and not _has_cree(right) and len(right) > 4:
            return _clean(left), _clean(right)
    return None


def _extract_pairs(text: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    lines  = text.splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()

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
            # may be a numbered stacked pair — Cree on this line, English next
            if _is_cree_dominant(content) and not _PROSE.match(content):
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
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                nxt = lines[j].strip()
                if _is_english(nxt) and len(nxt) > 3 and not _has_cree(nxt):
                    # sanity: cree line should look like a sentence or phrase (not a header)
                    if len(stripped) > 5:
                        pairs.append((_clean(stripped), _clean(nxt)))
                        i = j + 1
                        continue

        i += 1

    return pairs


def _dedup(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: set[str] = set()
    out  = []
    for cree, en in pairs:
        key = cree.lower()
        if key not in seen:
            seen.add(key)
            out.append((cree, en))
    return out


def _keep(cree: str, en: str, min_cree_len: int = 5) -> bool:
    if len(cree) < min_cree_len:
        return False
    if not _has_cree(cree):
        return False
    # syllable-breakdown lines are not sentences (e.g. "tā ni si")
    if re.search(r"\b[a-zāēīō]{1,3} [a-zāēīō]{1,3} [a-zāēīō]{1,3}\b", cree):
        return False
    # derivation formula lines
    if "+" in cree or "=" in cree:
        return False
    # parenthetical gloss in cree side indicates derivation example
    if re.search(r"\([a-z]", cree):
        return False
    # grammar metalanguage on english side
    if re.search(r"\b(subject|verb|object|noun|stem|suffix|prefix)\b", en, re.I):
        return False
    # numbered list item from derivation chapter
    if re.match(r"\d+\.", en):
        return False
    # bibliography / metadata
    if re.search(r"(isbn|pm\d{3}|\d{4}-\d{4}|edition\.|new edition)", en, re.I):
        return False
    # prose fragment from front matter
    if re.match(r"(Includes|As a result|Some changes|This edition)", en):
        return False
    # author name line
    if re.match(r"Jean L\.", cree):
        return False
    return True


def extract(pdf_path: str, min_cree_len: int = 5) -> list[tuple[str, str]]:
    """Extract deduplicated (cree, english) parallel pairs from the Okimasis PDF."""
    try:
        import fitz  # pymupdf
        doc  = fitz.open(pdf_path)
        text = "\n".join(
            page.get_text("text", sort=True) for page in doc
        )
    except ImportError:
        raise ImportError("pymupdf not installed — run: uv sync")

    pairs = _extract_pairs(text)
    pairs = _dedup(pairs)
    pairs = [(c, e) for c, e in pairs if _keep(c, e, min_cree_len=min_cree_len)]
    return pairs


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
    pairs = extract(args.pdf, min_cree_len=args.min_cree_len)

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
