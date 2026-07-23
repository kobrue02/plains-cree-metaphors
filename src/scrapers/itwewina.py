"""
itwêwina Plains Cree Dictionary scraper.

Queries https://itwewina.altlab.app/search?q=<word> and parses the HTML
to extract lemma, part-of-speech, grammatical class, and English definitions.

Usage:
    from src.scrapers.itwewina import lookup, lookup_sentence

    # Single word
    entries = lookup("nipiy")
    # → [{"lemma": "nipiy", "pos": "NI-2", "definitions": ["water", "body of water"]}, ...]

    # All content words in a Cree sentence
    results = lookup_sentence("nipiy kâ-pahkihtik")
    # → {"nipiy": [...], "pahkihtik": [...]}
"""

from __future__ import annotations

import re
import time
from functools import lru_cache

import requests
from bs4 import BeautifulSoup

_BASE = "https://itwewina.altlab.app/search"
_SESSION = requests.Session()
_SESSION.headers["User-Agent"] = "Mozilla/5.0 (academic research scraper)"

# cree preverbs / particles to skip (not worth looking up)
_SKIP = re.compile(
    r"^(kâ|ka|ê|e|wî|wi|nî|ni|ki|kî|pê|pî|isko|isi|mâh|âh|nôh|"
    r"mêk|êkwa|êkosi|mîna|piko|tâpwê|êsa|ana|awa|ôma|ôhi|ôho|"
    r"anima|aniki|ôki|nêhiyaw|peyak|nisto|nîso)$",
    re.IGNORECASE,
)

def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def _parse_results(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    entries = []

    for result in soup.select("li.search-results__result"):
        lemma_el = result.select_one("h2.definition-title__title")
        if not lemma_el:
            continue
        lemma = _clean(lemma_el.get_text())

        # grammatical class: first token before emoji/whitespace
        elab = result.select_one("div.definition__elaboration")
        pos = ""
        if elab:
            raw = _clean(elab.get_text())
            pos = raw.split()[0] if raw else ""

        # part-of-speech label (Naming word, Action word, etc.)
        wc = result.select_one("span.wordclass")
        pos_label = ""
        if wc:
            # strip emoji
            pos_label = re.sub(r"[^\w\s-]", "", wc.get_text()).strip()

        definitions = []
        for meaning_li in result.select("ol.meanings li"):
            # the definition text is the first text node; source citation follows
            texts = [t.strip() for t in meaning_li.stripped_strings]
            if texts:
                definitions.append(texts[0])

        if lemma and definitions:
            entries.append({
                "lemma":       lemma,
                "pos":         pos,
                "pos_label":   pos_label,
                "definitions": definitions,
            })

    return entries

@lru_cache(maxsize=2048)
def lookup(word: str, delay: float = 0.3) -> list[dict]:
    """Return dictionary entries for *word*, cached in-process; *delay* is a polite inter-request pause."""
    time.sleep(delay)
    try:
        r = _SESSION.get(_BASE, params={"q": word}, timeout=10)
        r.raise_for_status()
        return _parse_results(r.text)
    except Exception as exc:
        print(f"  [itwewina] lookup failed for {word!r}: {exc}")
        return []

def _tokenise(sentence: str) -> list[str]:
    """Split a Cree sentence into content-word tokens, stripping punctuation."""
    tokens = re.findall(r"[a-zA-ZâêîôâÂÊÎÔāēīōĀĒĪŌ'ʼ\-]+", sentence)
    return [t.lower() for t in tokens if len(t) >= 3 and not _SKIP.match(t)]

def _normalize(s: str) -> str:
    """Strip vowel-length diacritics (â/ê/î/ô) for lenient matching. Bloomfield-era/
    OCR'd Cree text is inconsistent about marking vowel length; itwêwina's own
    search normalizes this, but our post-filter previously compared raw strings,
    silently dropping real dictionary entries whenever the source token's
    diacritics didn't exactly match the site's citation form (e.g. querying
    "asê-takosinomakaniyiw" filtered out the site's actual lemma
    "asê-takosinômakaniyiw" over a single macron)."""
    import unicodedata
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    ).lower()

def lookup_sentence(
    sentence: str,
    delay: float = 0.3,
    verbose: bool = False,
) -> dict[str, list[dict]]:
    """Look up every content word in *sentence*; returns a dict of token→entries, omitting tokens with no results."""
    tokens = list(dict.fromkeys(_tokenise(sentence)))  # deduplicate, preserve order
    results: dict[str, list[dict]] = {}
    for tok in tokens:
        if verbose:
            print(f"  looking up: {tok}")
        tok_norm = _normalize(tok)
        entries = [
            e for e in lookup(tok, delay=delay)
            if _normalize(e["lemma"]) == tok_norm or _normalize(e["lemma"]).startswith(tok_norm)
        ]
        if entries:
            results[tok] = entries
    return results

def format_for_prompt(
    cree: str,
    english: str,
    lookups: dict[str, list[dict]],
) -> str:
    """Format dictionary lookups into an annotation prompt context block."""
    lines = [
        f"Cree sentence  : {cree}",
        f"English gloss  : {english}",
        "",
        "Word-level dictionary entries:",
    ]
    for word, entries in lookups.items():
        defs = "; ".join(
            " / ".join(e["definitions"][:2]) for e in entries[:2]
        )
        pos = entries[0]["pos"] if entries else ""
        lines.append(f"  {word} [{pos}]: {defs}")

    lines += [
        "",
        "Given the English gloss and the literal dictionary meanings above,",
        "is figurative language (metaphor, idiom, or simile) present in this sentence?",
        "Answer: literal / metaphor / idiom / simile",
    ]
    return "\n".join(lines)

if __name__ == "__main__":
    import argparse, json

    p = argparse.ArgumentParser(description="Look up Cree words in itwêwina.")
    p.add_argument("query", nargs="+", help="Word(s) or a quoted sentence to look up")
    p.add_argument("--sentence", action="store_true",
                   help="Treat input as a sentence and look up all content words")
    p.add_argument("--json", action="store_true", help="Output raw JSON")
    args = p.parse_args()

    query = " ".join(args.query)

    if args.sentence:
        results = lookup_sentence(query, verbose=True)
        if args.json:
            print(json.dumps(results, ensure_ascii=False, indent=2))
        else:
            for word, entries in results.items():
                print(f"\n{word}:")
                for e in entries:
                    print(f"  [{e['pos']}] {'; '.join(e['definitions'][:3])}")
    else:
        entries = lookup(query)
        if args.json:
            print(json.dumps(entries, ensure_ascii=False, indent=2))
        else:
            for e in entries:
                print(f"[{e['pos']}] {e['lemma']}: {'; '.join(e['definitions'][:3])}")
