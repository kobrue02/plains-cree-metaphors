"""
Generate TikZ/pgfplots figures for the paper from evaluation CSVs.

Run on the HPC after eval_zero_shot.sh completes:
  python scripts/generate_figures.py

Output:
  figures/results_heatmap.tex         — per-class consistency heatmap
  figures/results_figurative_dist.tex — label distribution on Bloomfield (stacked bar)
  figures/compile_figures.tex         — standalone LaTeX wrapper to compile both
"""

from __future__ import annotations
import os
import pandas as pd

os.makedirs("figures", exist_ok=True)

CONS_CSV = "data/figurative/eval_consistency.csv"
RATE_CSV = "data/figurative/eval_figurative_rate.csv"

MODELS_KEEP = [
    "XLM-R base (figurative)",
    "XLM-MLM CLKD frozen-12",
    "XLM-MLM CLKD full",
    "Glot500 CLKD direct",
    "Glot500 CLKD + TLM",
    "XLM-V CLKD direct",
]

SHORT_NAMES = {
    "XLM-R base (figurative)": r"\textsc{xlm-r}",
    "XLM-MLM CLKD frozen-12":  r"\textsc{xlm-mlm} (f12)",
    "XLM-MLM CLKD full":       r"\textsc{xlm-mlm} (full)",
    "Glot500 CLKD direct":     r"\textsc{glot500} (dir.)",
    "Glot500 CLKD + TLM":      r"\textsc{glot500} (+tlm)",
    "XLM-V CLKD direct":       r"\textsc{xlm-v} (dir.)",
}

cons = (
    pd.read_csv(CONS_CSV)
    .query("model in @MODELS_KEEP", engine="python")
    .query("error.isna()", engine="python")
    .set_index("model")
    .reindex(MODELS_KEEP)
)

rate = (
    pd.read_csv(RATE_CSV)
    .query("model in @MODELS_KEEP", engine="python")
    .query("error.isna()", engine="python")
    .set_index("model")
    .reindex(MODELS_KEEP)
)

# ── figure 1: per-class consistency heatmap ────────────────────────────────────

CLASSES      = ["literal", "idiom", "metaphor", "simile"]
CLASS_LABELS = ["Literal", "Idiom", "Metaphor", "Simile"]
n_models     = len(MODELS_KEEP)
n_classes    = len(CLASSES)

table_lines = ["x y value"]
cell_nodes  = []

for mi, model in enumerate(MODELS_KEEP):
    y = n_models - 1 - mi  # top-to-bottom
    for ci, cls in enumerate(CLASSES):
        val = float(cons.loc[model, f"agree_{cls}"])
        table_lines.append(f"{ci} {y} {val:.4f}")
        text_color = "white" if val > 0.75 else "black!80"
        cell_nodes.append(
            rf"\node[font=\scriptsize,text={text_color}] "
            rf"at (axis cs:{ci},{y}) {{{val:.2f}}};"
        )

table_data       = "\n    ".join(table_lines)
cell_annotations = "\n  ".join(cell_nodes)

ytick_positions = " ".join(str(n_models - 1 - i) for i in range(n_models))
ytick_labels    = ",".join(SHORT_NAMES[m] for m in MODELS_KEEP)
xtick_positions = " ".join(str(i) for i in range(n_classes))
xtick_labels    = ",".join(CLASS_LABELS)

heatmap_tex = rf"""\begin{{tikzpicture}}
\begin{{axis}}[
  width=8.6cm, height=6.2cm,
  colormap/Blues,
  colorbar,
  colorbar style={{
    ylabel={{\small Agreement}},
    ytick={{0,0.25,0.5,0.75,1.0}},
    yticklabel style={{font=\scriptsize}},
    width=0.25cm,
  }},
  point meta min=0, point meta max=1,
  xtick={{{xtick_positions}}},
  xticklabels={{{xtick_labels}}},
  ytick={{{ytick_positions}}},
  yticklabels={{{ytick_labels}}},
  yticklabel style={{font=\small}},
  xticklabel style={{font=\small}},
  tick align=outside,
  axis on top,
  enlarge x limits={{abs=0.5}},
  enlarge y limits={{abs=0.5}},
]
\addplot[matrix plot*, point meta=explicit]
  table [meta=value] {{
    {table_data}
  }};
  {cell_annotations}
\end{{axis}}
\end{{tikzpicture}}
"""

with open("figures/results_heatmap.tex", "w") as f:
    f.write(heatmap_tex)
print("Wrote figures/results_heatmap.tex")


# ── figure 2: label distribution on bloomfield (stacked bar) ──────────────────

RATE_CLASSES = ["literal", "idiom", "metaphor", "simile"]
RATE_LABELS  = ["Literal", "Idiom", "Metaphor", "Simile"]
COLORS       = ["gray!25", "NavyBlue!70", "BrickRed!70", "OliveGreen!70"]

n = len(MODELS_KEEP)

def coords_for_class(cls):
    return " ".join(
        f"({i},{float(rate.iloc[i][f'rate_{cls}']):.4f})"
        for i in range(n)
    )

bar_plots = "\n".join(
    rf"  \addplot+[ybar, fill={COLORS[ci]}, draw={COLORS[ci]}!80!black, opacity=0.9]"
    rf" coordinates {{{coords_for_class(cls)}}};"
    for ci, cls in enumerate(RATE_CLASSES)
)

xticklabels    = ",".join(SHORT_NAMES[m] for m in MODELS_KEEP)
legend_entries = ",".join(RATE_LABELS)

dist_tex = rf"""\begin{{tikzpicture}}
\begin{{axis}}[
  ybar stacked,
  width=8.6cm, height=5.8cm,
  bar width=0.6cm,
  ymin=0, ymax=1,
  ymajorgrids=true,
  grid style={{dashed, gray!35}},
  ylabel={{\small Proportion of sentences}},
  xtick={{0,...,{n - 1}}},
  xticklabels={{{xticklabels}}},
  xticklabel style={{font=\small, rotate=30, anchor=east}},
  yticklabel style={{font=\small}},
  legend style={{
    font=\small,
    at={{(0.5,1.06)}},
    anchor=south,
    legend columns=4,
    /tikz/every even column/.append style={{column sep=0.3em}},
  }},
  legend entries={{{legend_entries}}},
  enlarge x limits=0.12,
]
{bar_plots}
\end{{axis}}
\end{{tikzpicture}}
"""

with open("figures/results_figurative_dist.tex", "w") as f:
    f.write(dist_tex)
print("Wrote figures/results_figurative_dist.tex")


# ── standalone compile wrapper ─────────────────────────────────────────────────

wrapper_tex = r"""\documentclass[tikz, border=6pt]{standalone}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usepackage[dvipsnames]{xcolor}

\begin{document}
\input{results_heatmap}
\hspace{1.5cm}
\input{results_figurative_dist}
\end{document}
"""

with open("figures/compile_figures.tex", "w") as f:
    f.write(wrapper_tex)
print("Wrote figures/compile_figures.tex")
print("  → compile with: cd figures && pdflatex compile_figures.tex")
