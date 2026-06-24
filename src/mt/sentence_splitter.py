"""Split a paragraph-aligned parallel DataFrame into sentence-level pairs."""

import os
import re

import pandas as pd


_SENT_BOUNDARY = re.compile(r'(?<=[.!?])\s+|\n')


class ParallelSentenceSplitter:
    """Split paragraph-aligned Cree/English data into sentence pairs.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns ``text_cree`` and ``text_en``.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def split(self, min_chars: int = 20) -> pd.DataFrame:
        """Return a sentence-level DataFrame.

        Paragraphs with equal sentence counts on both sides are zipped directly.
        Mismatched counts are handled by a DP aligner (Gale-Church style, 1-1 /
        1-2 / 2-1 / 3-1 / 1-3) that minimises cumulative character-length error.

        A ``confidence`` score (0–1) reflects character-length proportionality;
        use it to filter noisy pairs (e.g. ``df[df.confidence > 0.6]``).

        Parameters
        ----------
        min_chars : int
            Drop pairs where either side is shorter than this.

        Returns
        -------
        pd.DataFrame
            Columns: paragraph_id, sentence_id, text_cree, text_en, confidence
        """
        rows = []
        for para_id, row in enumerate(self.df.itertuples(index=False)):
            cree_sents = self._split_sentences(self._normalise(str(row.text_cree)))
            en_sents   = self._split_sentences(self._normalise(str(row.text_en)))

            if not cree_sents or not en_sents:
                continue

            if len(cree_sents) == len(en_sents):
                pairs, confs = self._score(cree_sents, en_sents)
            else:
                pairs, confs = self._dp_align(cree_sents, en_sents)

            for sent_id, ((cree, en), conf) in enumerate(zip(pairs, confs)):
                if len(cree) < min_chars or len(en) < min_chars:
                    continue
                rows.append({
                    "paragraph_id": para_id,
                    "sentence_id":  sent_id,
                    "text_cree":    cree,
                    "text_en":      en,
                    "confidence":   conf,
                })

        return pd.DataFrame(rows)

    def write(self, path: str, min_confidence: float = 0.0) -> str:
        """Write sentence pairs in ``src ||| tgt`` format.

        Parameters
        ----------
        path : str
            Output file path.
        min_confidence : float
            Only write pairs at or above this confidence threshold.

        Returns
        -------
        str
            The output path.
        """
        sent_df = self.split()
        if min_confidence > 0:
            sent_df = sent_df[sent_df.confidence >= min_confidence]

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for _, row in sent_df.iterrows():
                f.write(f"{row.text_cree} ||| {row.text_en}\n")

        print(f"Wrote {len(sent_df):,} sentence pairs → {path}")
        return path

    # ── alignment helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _score(
        src: list[str], tgt: list[str]
    ) -> tuple[list[tuple[str, str]], list[float]]:
        src_total = sum(len(s) for s in src) or 1
        tgt_total = sum(len(t) for t in tgt) or 1
        pairs, confs = [], []
        for s, t in zip(src, tgt):
            s_frac = len(s) / src_total
            t_frac = len(t) / tgt_total
            denom  = max(s_frac, t_frac, 1e-9)
            pairs.append((s, t))
            confs.append(round(1.0 - abs(s_frac - t_frac) / denom, 3))
        return pairs, confs

    @staticmethod
    def _dp_align(
        src: list[str], tgt: list[str]
    ) -> tuple[list[tuple[str, str]], list[float]]:
        src_lens  = [max(len(s), 1) for s in src]
        tgt_lens  = [max(len(t), 1) for t in tgt]
        src_total = sum(src_lens)
        tgt_total = sum(tgt_lens)
        m, n = len(src), len(tgt)
        INF  = float("inf")

        dp   = [[INF] * (n + 1) for _ in range(m + 1)]
        back = [[None] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = 0.0

        MOVES = [(1, 1), (2, 1), (1, 2), (3, 1), (1, 3)]

        for i in range(m + 1):
            for j in range(n + 1):
                if i == 0 and j == 0:
                    continue
                cands = []
                for di, dj in MOVES:
                    pi, pj = i - di, j - dj
                    if pi < 0 or pj < 0 or dp[pi][pj] == INF:
                        continue
                    s = sum(src_lens[pi:i]) / src_total
                    t = sum(tgt_lens[pj:j]) / tgt_total
                    cands.append((dp[pi][pj] + (s - t) ** 2, (pi, pj)))
                if cands:
                    best       = min(cands, key=lambda x: x[0])
                    dp[i][j]   = best[0]
                    back[i][j] = best[1]

        if back[m][n] is None:
            return [(" ".join(src), " ".join(tgt))], [0.0]

        path = []
        i, j = m, n
        while i > 0 or j > 0:
            pi, pj = back[i][j]
            path.append(((pi, i), (pj, j)))
            i, j = pi, pj
        path.reverse()

        pairs, confs = [], []
        for (si, ei), (sj, ej) in path:
            c_frac = sum(src_lens[si:ei]) / src_total
            e_frac = sum(tgt_lens[sj:ej]) / tgt_total
            denom  = max(c_frac, e_frac, 1e-9)
            pairs.append((" ".join(src[si:ei]), " ".join(tgt[sj:ej])))
            confs.append(round(max(0.0, 1.0 - abs(c_frac - e_frac) / denom), 3))
        return pairs, confs

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        return [s.strip() for s in _SENT_BOUNDARY.split(text.strip()) if s.strip()]

    @staticmethod
    def _normalise(text: str) -> str:
        return re.sub(r'\s+', ' ', text).strip()
