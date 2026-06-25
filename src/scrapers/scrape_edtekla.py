"""
Fetch parallel Plains Cree / English texts from the EdTeKLA IndigenousLanguages_Corpora
repository and write them in the ``src ||| tgt`` format used by the TLM pipeline.

All files are fetched via raw GitHub content URLs — no API token required.
Files with equal line counts are zipped directly.  Mismatched counts fall
back to the Gale–Church DP aligner already used by ParallelSentenceSplitter.

Source: https://github.com/EdTeKLA/IndigenousLanguages_Corpora
Citation: Teodorescu et al., ACL 2022
"""

from __future__ import annotations

import os
import re

import requests


_RAW = "https://raw.githubusercontent.com/EdTeKLA/IndigenousLanguages_Corpora/main"

# All (cree_path, english_path, label) pairs confirmed present in the repo.
# BoW-only files (e.g. SolomonRatt, Bible) are intentionally excluded.
_PAIRS: list[tuple[str, str, str]] = [
    # Speaker stories
    ("PlainsCree/SpeakerStories/Neil/Neil_cr.txt",
     "PlainsCree/SpeakerStories/Neil/Neil_en.txt",
     "speaker_neil"),
    ("PlainsCree/SpeakerStories/Beatrice/Beatrice_cr.txt",
     "PlainsCree/SpeakerStories/Beatrice/Beatrice_en.txt",
     "speaker_beatrice"),
    # Facebook
    ("PlainsCree/Facebook/Fish/Fish_cr.txt",
     "PlainsCree/Facebook/Fish/Fish_en.txt",
     "facebook_fish"),
    ("PlainsCree/Facebook/Foods/Foods_cr.txt",
     "PlainsCree/Facebook/Foods/Foods_en.txt",
     "facebook_foods"),
    # Twitter
    ("PlainsCree/Twitter/Twitter_cr.txt",
     "PlainsCree/Twitter/Twitter_en.txt",
     "twitter"),
    # Elections
    ("PlainsCree/Elections/Canada/VG_cr.txt",
     "PlainsCree/Elections/Canada/VG_en.txt",
     "elections_canada"),
    ("PlainsCree/Elections/Alberta/304/elections304_cr.txt",
     "PlainsCree/Elections/Alberta/304/elections304_en.txt",
     "elections_alberta_304"),
    ("PlainsCree/Elections/Alberta/305/elections305_cr.txt",
     "PlainsCree/Elections/Alberta/305/elections305_en.txt",
     "elections_alberta_305"),
    ("PlainsCree/Elections/Alberta/528/elections528_cr.txt",
     "PlainsCree/Elections/Alberta/528/elections528_en.txt",
     "elections_alberta_528"),
    # Children's books
    ("PlainsCree/ChildrenBooks/LittleBear/LittleBear_cr.txt",
     "PlainsCree/ChildrenBooks/LittleBear/LittleBear_en.txt",
     "children_littlebear"),
    ("PlainsCree/ChildrenBooks/SeasonKitty/SeasonKitten_cr.txt",
     "PlainsCree/ChildrenBooks/SeasonKitty/SeasonKitten_en.txt",
     "children_seasonkitten"),
    # Educational
    ("PlainsCree/Educational/CreeLanguageTextbook/CreeLanguageTextbook_cr.txt",
     "PlainsCree/Educational/CreeLanguageTextbook/CreeLanguageTextbook_en.txt",
     "textbook"),
    ("PlainsCree/Educational/TeachingWhy/TeachingWhy_cr.txt",
     "PlainsCree/Educational/TeachingWhy/TeachingWhy_en.txt",
     "teaching_why"),
    # CreeLiteracyOrg
    ("PlainsCree/CreeLiteracyOrg/CongratsGrads/congrats_grads_cr.txt",
     "PlainsCree/CreeLiteracyOrg/CongratsGrads/congrats_grads_en.txt",
     "congrats_grads"),
    ("PlainsCree/CreeLiteracyOrg/ForMoment/for_a_moment_cr.txt",
     "PlainsCree/CreeLiteracyOrg/ForMoment/for_a_moment_en.txt",
     "for_a_moment"),
    ("PlainsCree/CreeLiteracyOrg/GraduateSong/graduate_song_cr.txt",
     "PlainsCree/CreeLiteracyOrg/GraduateSong/graduate_song_en.txt",
     "graduate_song"),
    ("PlainsCree/CreeLiteracyOrg/HappyMothersDay/happy_mothers_day_cr.txt",
     "PlainsCree/CreeLiteracyOrg/HappyMothersDay/happy_mothers_day_en.txt",
     "mothers_day"),
    ("PlainsCree/CreeLiteracyOrg/OvercomeErasure/overcoming_erasure_cr.txt",
     "PlainsCree/CreeLiteracyOrg/OvercomeErasure/overcoming_erasure_en.txt",
     "overcome_erasure"),
    ("PlainsCree/CreeLiteracyOrg/ThinkingYou/thinking_of_you_cr.txt",
     "PlainsCree/CreeLiteracyOrg/ThinkingYou/thinking_of_you_en.txt",
     "thinking_of_you"),
    ("PlainsCree/CreeLiteracyOrg/Verbs/verbs_cr.txt",
     "PlainsCree/CreeLiteracyOrg/Verbs/verbs_en.txt",
     "verbs"),
    # Articles
    ("PlainsCree/Articles/CreeLangIsBeaut/CreeLanguageBeautiful_cr.txt",
     "PlainsCree/Articles/CreeLangIsBeaut/CreeLanguageBeautiful_en.txt",
     "article_cree_beautiful"),
    ("PlainsCree/Articles/NotesOnPlainsCree/NotesOnPlainsCree_cr.txt",
     "PlainsCree/Articles/NotesOnPlainsCree/NotesOnPlainsCree_en.txt",
     "article_notes"),
    ("PlainsCree/Articles/ThreeKindsNominal/ThreeKindsNominal_cr.txt",
     "PlainsCree/Articles/ThreeKindsNominal/ThreeKindsNominal_en.txt",
     "article_nominal"),
]


def _fetch(path: str, session: requests.Session) -> list[str]:
    url = f"{_RAW}/{path}"
    r = session.get(url, timeout=20)
    r.raise_for_status()
    r.encoding = "utf-8"
    return [_clean(line) for line in r.text.splitlines()]


def _clean(line: str) -> str:
    line = re.sub(r"\s+", " ", line).strip()
    # Strip lone punctuation-only lines
    return line if re.search(r"\w", line) else ""


def _dp_align(src: list[str], tgt: list[str]) -> list[tuple[str, str]]:
    """Minimal Gale-Church 1-1/1-2/2-1 aligner used as fallback."""
    src_lens = [max(len(s), 1) for s in src]
    tgt_lens = [max(len(t), 1) for t in tgt]
    m, n = len(src), len(tgt)
    INF = float("inf")

    dp   = [[INF] * (n + 1) for _ in range(m + 1)]
    back = [[None] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = 0.0
    src_total = sum(src_lens) or 1
    tgt_total = sum(tgt_lens) or 1

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 and j == 0:
                continue
            best_cost, best_prev = INF, None
            for di, dj in [(1, 1), (2, 1), (1, 2), (3, 1), (1, 3)]:
                pi, pj = i - di, j - dj
                if pi < 0 or pj < 0 or dp[pi][pj] == INF:
                    continue
                s = sum(src_lens[pi:i]) / src_total
                t = sum(tgt_lens[pj:j]) / tgt_total
                cost = dp[pi][pj] + (s - t) ** 2
                if cost < best_cost:
                    best_cost, best_prev = cost, (pi, pj)
            if best_prev is not None:
                dp[i][j]   = best_cost
                back[i][j] = best_prev

    if back[m][n] is None:
        return [(" ".join(src), " ".join(tgt))]

    path_steps = []
    i, j = m, n
    while i > 0 or j > 0:
        pi, pj = back[i][j]
        path_steps.append(((pi, i), (pj, j)))
        i, j = pi, pj
    path_steps.reverse()

    return [
        (" ".join(src[si:ei]), " ".join(tgt[sj:ej]))
        for (si, ei), (sj, ej) in path_steps
    ]


class EdTeKLAScraper:
    """Download all parallel Cree–English pairs from EdTeKLA and write src ||| tgt."""

    def scrape(
        self,
        output: str = "data/sentences_edtekla.txt",
        min_chars: int = 5,
        append: bool = False,
    ) -> list[tuple[str, str]]:
        """Fetch all parallel pairs and write to *output*.

        Parameters
        ----------
        output:
            Destination file path (``src ||| tgt`` format, one pair per line).
        min_chars:
            Drop pairs where either side is shorter than this many characters.
        append:
            If True, open *output* in append mode instead of overwriting.

        Returns
        -------
        list of (cree, english) string pairs collected across all sources.
        """
        session = requests.Session()
        session.headers["User-Agent"] = "Mozilla/5.0 (academic scraper)"

        all_pairs: list[tuple[str, str]] = []

        for cr_path, en_path, label in _PAIRS:
            try:
                cr_lines = [l for l in _fetch(cr_path, session) if l]
                en_lines = [l for l in _fetch(en_path, session) if l]
            except Exception as exc:
                print(f"  [skip] {label}: {exc}")
                continue

            if len(cr_lines) == len(en_lines):
                pairs = list(zip(cr_lines, en_lines))
            else:
                pairs = _dp_align(cr_lines, en_lines)

            before = len(pairs)
            pairs = [(c, e) for c, e in pairs if len(c) >= min_chars and len(e) >= min_chars]
            all_pairs.extend(pairs)
            print(f"  {label:30s}  {before:3d} → {len(pairs):3d} pairs")

        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        mode = "a" if append else "w"
        with open(output, mode, encoding="utf-8") as f:
            for cree, english in all_pairs:
                f.write(f"{cree} ||| {english}\n")

        print(f"\nEdTeKLA: wrote {len(all_pairs):,} pairs → {output}"
              f"  (mode={'append' if append else 'overwrite'})")
        return all_pairs
