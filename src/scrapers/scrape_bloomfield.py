import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup, NavigableString, Tag


class BloomfieldScraper:
    BASE_URL = "https://bloomfield.kiyanaw.net"
    INDEX_URL = BASE_URL + "/"

    _session = requests.Session()
    _session.headers["User-Agent"] = "Mozilla/5.0 (academic scraper)"

    def scrape(self, output: str = "bloomfield_texts.csv") -> pd.DataFrame:
        """Main driver: fetch all texts and write to CSV."""
        print("Fetching index page...")
        index_soup = self._fetch(self.INDEX_URL)
        links = self._get_text_links(index_soup)
        print(f"Found {len(links)} text pages.\n")

        all_rows = []
        for i, url in enumerate(links, 1):
            slug = url.split("/")[-1]
            print(f"[{i:3d}/{len(links)}] {slug}")
            try:
                rows = self._parse_page(url)
                all_rows.extend(rows)
                print(f"         -> {len(rows)} paragraphs")
            except Exception as e:
                print(f"         -> ERROR: {e}")
            time.sleep(0.3)

        cols = ["source_file", "paragraph_num", "text_cree", "text_en", "footnote_en"]
        df = pd.DataFrame(all_rows, columns=cols)
        df.to_csv(output, index=False, encoding="utf-8-sig")
        print(f"\nDone. {len(df)} total paragraphs written to {output}.")
        return df

    def _fetch(self, url: str) -> BeautifulSoup:
        r = self._session.get(url, timeout=20)
        r.raise_for_status()
        r.encoding = "utf-8"
        return BeautifulSoup(r.text, "lxml")

    def _get_text_links(self, index_soup: BeautifulSoup) -> list[str]:
        links = []
        for a in index_soup.find_all("a", href=True):
            href = a["href"]
            if href.endswith(".html") and href.startswith("/"):
                links.append(self.BASE_URL + href)
        return links

    def _parse_page(self, url: str) -> list[dict]:
        soup = self._fetch(url)
        slug = url.split("/")[-1].replace(".html", "")
        footnotes = self._collect_footnotes(soup)

        records: dict[int, dict] = {}
        current_crk_num: int | None = None

        for p in soup.find_all("p"):
            classes = p.get("class", [])

            if "paragraph" in classes and "crk" in classes:
                para_id = p.get("id")
                if para_id and para_id.isdigit():
                    current_crk_num = int(para_id)
                    records[current_crk_num] = {
                        "cree_parts": [self._para_text(p)],
                        "eng_parts": [],
                        "fn_nums": self._footnote_nums(p),
                    }

            elif "paragraph" in classes and "eng" in classes:
                strong = p.find("strong")
                if strong:
                    m = re.match(r"\(?(\d+)\)?", strong.get_text().strip())
                    if m:
                        eng_num = int(m.group(1))
                        if eng_num not in records:
                            records[eng_num] = {"cree_parts": [], "eng_parts": [], "fn_nums": []}
                        records[eng_num]["eng_parts"].append(self._para_text(p))
                        current_crk_num = None

            elif "footnote" not in classes and current_crk_num is not None:
                text = self._para_text(p)
                if text:
                    records[current_crk_num]["cree_parts"].append(text)
                    records[current_crk_num]["fn_nums"].extend(self._footnote_nums(p))

        return self._build_rows(slug, records, footnotes)

    def _collect_footnotes(self, soup: BeautifulSoup) -> dict[int, str]:
        footnotes: dict[int, str] = {}
        for p in soup.find_all("p", class_="footnote"):
            sup = p.find("sup")
            if sup and sup.get_text().strip().isdigit():
                num = int(sup.get_text().strip())
                p_copy = BeautifulSoup(str(p), "lxml").find("p")
                p_copy.find("sup").decompose()
                footnotes[num] = p_copy.get_text(" ", strip=True)
        return footnotes

    def _build_rows(
        self, slug: str, records: dict[int, dict], footnotes: dict[int, str]
    ) -> list[dict]:
        rows = []
        for num in sorted(records):
            rec = records[num]
            text_cree = "\n".join(p for p in rec["cree_parts"] if p).strip()
            text_en = "\n".join(p for p in rec["eng_parts"] if p).strip()
            fn_texts = [
                f"[{n}] {footnotes[n]}"
                for n in sorted(set(rec["fn_nums"]))
                if n in footnotes
            ]
            if text_cree or text_en:
                rows.append({
                    "source_file": slug,
                    "paragraph_num": num,
                    "text_cree": text_cree,
                    "text_en": text_en,
                    "footnote_en": "\n".join(fn_texts),
                })
        return rows

    @staticmethod
    def _para_text(p: Tag) -> str:
        """Visible text of a <p>, dropping the leading paragraph-number <strong> and <sup> markers."""
        parts = []
        for child in p.children:
            if isinstance(child, NavigableString):
                parts.append(str(child))
            elif isinstance(child, Tag):
                if child.name == "strong" and not parts:
                    continue
                if child.name == "sup":
                    continue
                parts.append(child.get_text())
        return " ".join(parts).strip()

    @staticmethod
    def _footnote_nums(p: Tag) -> list[int]:
        """Footnote reference numbers from <sup> tags in a paragraph."""
        return [
            int(sup.get_text().strip())
            for sup in p.find_all("sup")
            if sup.get_text().strip().isdigit()
        ]


if __name__ == "__main__":
    BloomfieldScraper().scrape()
