import re
import textwrap
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class EDA:
    def __init__(self, df: pd.DataFrame) -> None:
        required = {"source_file", "paragraph_num", "text_cree", "text_en", "footnote_en"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing columns: {missing}")
        self._df = df.copy()

    def run(self) -> tuple[pd.DataFrame, list[plt.Figure]]:
        """Compute all statistics and produce plots."""
        per_para = self._per_paragraph_frame()
        summary = self._summary_frame(per_para)
        figures = [
            self._plot_distributions(per_para),
            self._plot_sentence_lengths(per_para),
            self._plot_per_text(per_para),
            self._plot_vocabulary(),
        ]
        return summary, figures
    
    def _per_paragraph_frame(self) -> pd.DataFrame:
        df = self._df.copy()
        for lang, col in (("crk", "text_cree"), ("en", "text_en")):
            words = df[col].map(self._words)
            sents = df[col].map(self._sentences)
            df[f"word_count_{lang}"] = words.map(len)
            df[f"char_count_{lang}"] = df[col].map(len)
            df[f"sent_count_{lang}"] = sents.map(len)
            df[f"avg_word_len_{lang}"] = words.map(
                lambda ws: sum(len(w) for w in ws) / len(ws) if ws else 0
            )
            df[f"avg_sent_len_{lang}"] = df.apply(
                lambda r, c=col: self._avg_words_per_sent(r[c]), axis=1
            )
        df["word_ratio_crk_en"] = df["word_count_crk"] / df["word_count_en"].replace(0, float("nan"))
        df["has_footnote"] = df["footnote_en"] != ""
        return df

    def _summary_frame(self, per_para: pd.DataFrame) -> pd.DataFrame:
        def _vocab(col):
            all_words = [w for t in self._df[col] for w in self._words(t)]
            types = len(set(all_words))
            return types, round(types / len(all_words), 4) if all_words else 0

        crk_vocab, crk_ttr = _vocab("text_cree")
        en_vocab, en_ttr = _vocab("text_en")

        rows = {
            "total_words":                (per_para["word_count_crk"].sum(), per_para["word_count_en"].sum()),
            "total_sentences":            (per_para["sent_count_crk"].sum(), per_para["sent_count_en"].sum()),
            "vocabulary_size":            (crk_vocab, en_vocab),
            "type_token_ratio":           (crk_ttr, en_ttr),
            "avg_words_per_paragraph":    (per_para["word_count_crk"].mean().round(2), per_para["word_count_en"].mean().round(2)),
            "avg_sentences_per_paragraph":(per_para["sent_count_crk"].mean().round(2), per_para["sent_count_en"].mean().round(2)),
            "avg_words_per_sentence":     (per_para["avg_sent_len_crk"].mean().round(2), per_para["avg_sent_len_en"].mean().round(2)),
            "avg_word_length_chars":      (per_para["avg_word_len_crk"].mean().round(2), per_para["avg_word_len_en"].mean().round(2)),
            "avg_chars_per_paragraph":    (per_para["char_count_crk"].mean().round(2), per_para["char_count_en"].mean().round(2)),
            "mean_cree_en_word_ratio":    (per_para["word_ratio_crk_en"].mean().round(3), "—"),
            "footnote_coverage_pct":      (round(per_para["has_footnote"].mean() * 100, 2), "—"),
        }
        return pd.DataFrame.from_dict(rows, orient="index", columns=["cree", "english"])

    def _plot_distributions(self, per_para: pd.DataFrame) -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle("Word counts & word lengths", fontsize=13)

        ax = axes[0]
        clip = (0, self._p99(per_para, "word_count_crk", "word_count_en"))
        sns.kdeplot(per_para["word_count_crk"], ax=ax, fill=True, alpha=0.5, label="Cree",    color="#2196F3", clip=clip)
        sns.kdeplot(per_para["word_count_en"],  ax=ax, fill=True, alpha=0.5, label="English", color="#FF9800", clip=clip)
        ax.set_xlim(*clip)
        ax.set_xlabel("Words per paragraph")
        ax.set_ylabel("Density")
        ax.set_title("Words per paragraph")
        ax.legend()

        ax = axes[1]
        clip = (0, self._p99(per_para, "avg_word_len_crk", "avg_word_len_en"))
        sns.kdeplot(per_para["avg_word_len_crk"], ax=ax, fill=True, alpha=0.5, label="Cree",    color="#2196F3", clip=clip)
        sns.kdeplot(per_para["avg_word_len_en"],  ax=ax, fill=True, alpha=0.5, label="English", color="#FF9800", clip=clip)
        ax.set_xlim(*clip)
        ax.set_xlabel("Average word length (chars)")
        ax.set_ylabel("Density")
        ax.set_title("Average word length per paragraph")
        ax.legend()

        fig.tight_layout()
        return fig

    def _plot_sentence_lengths(self, per_para: pd.DataFrame) -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle("Sentence-level statistics", fontsize=13)

        ax = axes[0]
        clip = (0, self._p99(per_para, "sent_count_crk", "sent_count_en"))
        sns.kdeplot(per_para["sent_count_crk"], ax=ax, fill=True, alpha=0.5, label="Cree",    color="#2196F3", clip=clip)
        sns.kdeplot(per_para["sent_count_en"],  ax=ax, fill=True, alpha=0.5, label="English", color="#FF9800", clip=clip)
        ax.set_xlim(*clip)
        ax.set_xlabel("Sentences per paragraph")
        ax.set_ylabel("Density")
        ax.set_title("Sentences per paragraph")
        ax.legend()

        ax = axes[1]
        clip = (0, self._p99(per_para, "avg_sent_len_crk", "avg_sent_len_en"))
        sns.kdeplot(per_para["avg_sent_len_crk"], ax=ax, fill=True, alpha=0.5, label="Cree",    color="#2196F3", clip=clip)
        sns.kdeplot(per_para["avg_sent_len_en"],  ax=ax, fill=True, alpha=0.5, label="English", color="#FF9800", clip=clip)
        ax.set_xlim(*clip)
        ax.set_xlabel("Words per sentence (avg)")
        ax.set_ylabel("Density")
        ax.set_title("Average words per sentence")
        ax.legend()

        fig.tight_layout()
        return fig

    def plot_figurative_language(self, annotated_df: pd.DataFrame) -> plt.Figure:
        """Bar plot of figurative language type counts from the annotated dataset.
        Also prints one sample per non-none type to stdout."""
        labelled = annotated_df[
            annotated_df["annotation"].notna()
            & (annotated_df["annotation"] != "")
        ]
        counts = labelled["annotation"].value_counts().reset_index()
        counts.columns = ["type", "count"]

        figurative = labelled[labelled["annotation"] != "none"]
        for label, group in figurative.groupby("annotation"):
            row = group.iloc[0]
            print(f"--- {label.upper()} ---")
            print(f"Cree:\n{row['text_cree']}")
            print(f"English:\n{row['text_en']}")
            if row.get("reasoning"):
                print(f"Reasoning:\n{str(row['reasoning'])}")
            print()

        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(data=counts, x="type", y="count", hue="type", ax=ax, palette="Blues_d", legend=False)
        ax.set_title("Figurative language types")
        ax.set_xlabel("Type")
        ax.set_ylabel("Count")
        for bar in ax.patches:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2,
                int(bar.get_height()),
                ha="center", va="bottom", fontsize=10,
            )
        fig.tight_layout()
        return fig

    def _plot_per_text(self, per_para: pd.DataFrame) -> plt.Figure:
        grp = per_para.groupby("source_file")[["word_count_crk", "word_count_en"]].mean().round(1)
        grp = grp.sort_values("word_count_crk", ascending=True).tail(20)
        short_names = grp.index.map(lambda s: s[:45])

        fig, ax = plt.subplots(figsize=(10, 8))
        y = range(len(grp))
        ax.barh([i + 0.2 for i in y], grp["word_count_crk"], height=0.4, label="Cree", color="#2196F3")
        ax.barh([i - 0.2 for i in y], grp["word_count_en"],  height=0.4, label="English", color="#FF9800")
        ax.set_yticks(list(y))
        ax.set_yticklabels(short_names, fontsize=8)
        ax.set_xlabel("Avg words per paragraph")
        ax.set_title("Avg words per paragraph by text (top 20)")
        ax.legend()
        fig.tight_layout()
        return fig

    def _plot_vocabulary(self) -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle("Vocabulary", fontsize=13)

        for lang, col, color, label in (
            ("crk", "text_cree",  "#2196F3", "Cree"),
            ("en",  "text_en",    "#FF9800", "English"),
        ):
            all_words = [w for t in self._df[col] for w in self._words(t)]
            top = Counter(all_words).most_common(15)
            words, counts = zip(*top)

            ax = axes[0 if lang == "crk" else 1]
            ax.barh(range(len(words)), counts, color=color, alpha=0.8)
            ax.set_yticks(range(len(words)))
            ax.set_yticklabels(words, fontsize=9)
            ax.invert_yaxis()
            ax.set_xlabel("Frequency")
            ax.set_title(f"Top 15 {label} words")

        fig.tight_layout()
        return fig

    _FIGURE_NAMES = [
        "word_distributions",
        "sentence_distributions",
        "per_text",
        "vocabulary",
    ]

    def to_tikz(self, figures: list[plt.Figure], output_dir: str | Path = ".") -> list[Path]:
        """Serialise each figure to a standalone pgfplots .tex file.

        The files can be included in a LaTeX document with::

            \\input{word_distributions.tex}

        Requires ``\\usepackage{pgfplots}`` and ``\\usetikzlibrary{pgfplots.fillbetween}``
        in the document preamble.

        Returns the list of written paths.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for fig, name in zip(figures, self._FIGURE_NAMES):
            code = self._figure_to_tikz(fig)
            path = output_dir / f"{name}.tex"
            path.write_text(code, encoding="utf-8")
            paths.append(path)
        return paths

    @staticmethod
    def _figure_to_tikz(fig: plt.Figure) -> str:
        axes_blocks = "\n\n".join(
            EDA._axes_to_tikz(ax) for ax in fig.get_axes()
        )
        suptitle = fig.texts[0].get_text() if fig.texts else ""
        comment = f"% {suptitle}\n" if suptitle else ""
        return (
            comment
            + "% Requires: \\usepackage{pgfplots}\n"
            + "\\begin{tikzpicture}\n"
            + axes_blocks
            + "\n\\end{tikzpicture}\n"
        )

    @staticmethod
    def _axes_to_tikz(ax: plt.Axes) -> str:
        # seaborn kdeplot fill=True → PolyCollection; plain bar → patches
        if ax.collections:
            return EDA._kde_axes_to_tikz(ax)
        elif ax.patches:
            return EDA._bar_axes_to_tikz(ax)
        return ""

    @staticmethod
    def _kde_axes_to_tikz(ax: plt.Axes) -> str:
        xlabel = ax.get_xlabel().replace("_", "\\_")
        ylabel = ax.get_ylabel().replace("_", "\\_")
        title  = ax.get_title().replace("_", "\\_")
        xmin, xmax = ax.get_xlim()

        options = textwrap.dedent(f"""\
            title={{{title}}},
            xlabel={{{xlabel}}},
            ylabel={{{ylabel}}},
            xmin={xmin:.4f}, xmax={xmax:.4f},
            width=0.48\\linewidth,
            legend pos=north east,
            legend style={{font=\\small}},
            axis lines=left,
            tick align=outside,
        """)

        plots = []
        for col in ax.collections:
            paths = col.get_paths()
            if not paths:
                continue
            label = col.get_label()
            color = EDA._mpl_color_to_tikz(col.get_facecolor()[0])
            # Extract the upper KDE curve: keep only y > 0, sort by x
            verts = paths[0].vertices
            curve = verts[verts[:, 1] > 1e-10]
            curve = curve[np.argsort(curve[:, 0])]
            # Downsample to ~100 points so the .tex file stays manageable
            step = max(1, len(curve) // 100)
            curve = curve[::step]
            coords = " ".join(f"({x:.5f},{y:.6f})" for x, y in curve)
            plots.append(
                f"\\addplot[{color}, thick, fill={color}, fill opacity=0.4, mark=none]\n"
                f"  coordinates {{{coords}}}\\closedcycle;\n"
                f"\\addlegendentry{{{label}}}"
            )

        inner = "\n".join(plots)
        return f"\\begin{{axis}}[\n{textwrap.indent(options, '  ')}]\n{inner}\n\\end{{axis}}"

    @staticmethod
    def _bar_axes_to_tikz(ax: plt.Axes) -> str:
        xlabel = ax.get_xlabel().replace("_", "\\_")
        title  = ax.get_title().replace("_", "\\_")

        # Group patches by colour — each colour = one data series
        from collections import defaultdict
        series: dict[str, list[tuple]] = defaultdict(list)
        for patch in ax.patches:
            color = EDA._mpl_color_to_tikz(patch.get_facecolor())
            # Horizontal bar: x=width, y=centre of bar
            x = patch.get_width()
            y = patch.get_y() + patch.get_height() / 2
            series[color].append((x, y))

        # Recover legend labels in order of first occurrence
        legend_labels: dict[str, str] = {}
        for handle, text in zip(*ax.get_legend_handles_labels()):
            # BarContainer wraps a list of patches; take the first one's colour
            sample = handle.patches[0] if hasattr(handle, "patches") else handle
            color = EDA._mpl_color_to_tikz(sample.get_facecolor())
            if color not in legend_labels:
                legend_labels[color] = text.replace("_", "\\_")

        ytick_vals = [t for t in ax.get_yticks() if ax.get_ylim()[0] <= t <= ax.get_ylim()[1]]
        ytick_labels = [lbl.get_text().replace("_", "\\_") for lbl in ax.get_yticklabels()
                        if lbl.get_text()]

        yticks_opt = ""
        if ytick_vals and ytick_labels:
            vals_str   = ",".join(f"{v:.2f}" for v in ytick_vals[:len(ytick_labels)])
            labels_str = ",".join(ytick_labels)
            yticks_opt = f"ytick={{{vals_str}}},\nyticklabels={{{labels_str}}},\n"

        options = textwrap.dedent(f"""\
            title={{{title}}},
            xlabel={{{xlabel}}},
            xbar,
            bar width=4pt,
            width=0.9\\linewidth,
            height=14cm,
            {yticks_opt}legend pos=south east,
            axis lines=left,
        """)

        plots = []
        for color, data in series.items():
            data_sorted = sorted(data, key=lambda p: p[1])
            coords = " ".join(f"({x:.4f},{y:.4f})" for x, y in data_sorted)
            label = legend_labels.get(color, color)
            plots.append(
                f"\\addplot[{color}, fill={color}, fill opacity=0.8, mark=none]\n"
                f"  coordinates {{{coords}}};\n"
                f"\\addlegendentry{{{label}}}"
            )

        inner = "\n".join(plots)
        return f"\\begin{{axis}}[\n{textwrap.indent(options, '  ')}]\n{inner}\n\\end{{axis}}"

    @staticmethod
    def _mpl_color_to_tikz(color) -> str:
        """Convert a matplotlib colour (any format) to a pgfplots-compatible RGB string."""
        rgba = plt.matplotlib.colors.to_rgba(color)
        r, g, b = (int(c * 255) for c in rgba[:3])
        return f"{{rgb,255:red,{r};green,{g};blue,{b}}}"

    @staticmethod
    def _p99(per_para: pd.DataFrame, *cols: str) -> float:
        """99th-percentile upper bound across one or more columns, for xlim clipping."""
        return max(per_para[col].quantile(0.99) for col in cols)

    @staticmethod
    def _words(text: str) -> list[str]:
        return re.findall(r"[^\s\.,;:!?\"'()\[\]]+", text.lower())

    @staticmethod
    def _sentences(text: str) -> list[str]:
        return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]

    @staticmethod
    def _avg_words_per_sent(text: str) -> float:
        sents = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        if not sents:
            return 0.0
        counts = [len(re.findall(r"[^\s\.,;:!?\"'()\[\]]+", s.lower())) for s in sents]
        return sum(counts) / len(counts)



if __name__ == "__main__":
    base_df = pd.read_csv("data/bloomfield_texts.csv", encoding="utf-8-sig")
    annotated_df = pd.read_csv("data/bloomfield_texts_annotated.csv", encoding="utf-8-sig")
    fig = EDA(base_df).plot_figurative_language(annotated_df)
    fig.savefig("figures/figurative_language_types.png", dpi=300)
    fig.show()