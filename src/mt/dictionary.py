"""Thin wrapper around the itwêwina Plains Cree dictionary (itwewina.altlab.app)."""

import time
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup


@dataclass
class DictionaryEntry:
    headword:    str
    syllabics:   str
    word_class:  str 
    pos_tags:    list[str]
    definitions: list[str]
    sources:     list[str]


class ItwewinaClient:
    """HTTP client for the itwêwina Plains Cree online dictionary."""

    _BASE   = "https://itwewina.altlab.app"
    _SEARCH = f"{_BASE}/search"

    def __init__(self, min_interval: float = 0.5, timeout: int = 10):
        self.min_interval = min_interval
        self.timeout      = timeout
        self._last_request = 0.0
        self._session = requests.Session()
        self._session.headers["User-Agent"] = "fnlp-research/0.1 (academic use)"

    def search(self, query: str) -> list[DictionaryEntry]:
        """Search itwêwina for a Cree or English word; return all result entries."""
        self._rate_limit()
        resp = self._session.get(self._SEARCH, params={"q": query}, timeout=self.timeout)
        resp.raise_for_status()
        self._last_request = time.monotonic()
        return self._parse(resp.text)

    # ── private ───────────────────────────────────────────────────────────────

    def _rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

    @staticmethod
    def _parse(html: str) -> list[DictionaryEntry]:
        soup = BeautifulSoup(html, "lxml")
        return [ItwewinaClient._parse_article(a)
                for a in soup.find_all("article", class_="definition")]

    @staticmethod
    def _parse_article(article) -> DictionaryEntry:
        span     = article.find("span", attrs={"data-orth-latn": True})
        headword = span["data-orth-latn"] if span else ""
        syllabics = span.get("data-orth-cans", "") if span else ""

        elab       = article.find("div", class_="definition__elaboration")
        word_class = elab.get_text(separator=" ").split()[0] if elab else ""

        lb       = article.find("div", attrs={"data-cy": "linguistic-breakdown"})
        pos_tags = [li.get_text(strip=True) for li in lb.find_all("li")] if lb else []

        definitions, sources = [], []
        meanings = article.find("ol", class_="meanings")
        if meanings:
            for li in meanings.find_all("li", class_="meanings__meaning"):
                cite   = li.find("cite")
                source = cite.get_text(strip=True) if cite else ""
                if cite:
                    cite.decompose()
                for div in li.find_all("div"):
                    div.decompose()
                definitions.append(li.get_text(separator=" ", strip=True))
                sources.append(source)

        return DictionaryEntry(
            headword=headword, syllabics=syllabics, word_class=word_class,
            pos_tags=pos_tags, definitions=definitions, sources=sources,
        )


if __name__ == "__main__":
    client = ItwewinaClient()
    for query in ("iskwew", "as", "bear"):
        print(f"\n=== {query} ===")
        for entry in client.search(query)[:3]:
            print(f"{entry.headword} ({entry.syllabics})  [{entry.word_class}]")
            for defn, src in zip(entry.definitions, entry.sources):
                print(f"  [{src}] {defn}")
