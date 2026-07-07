"""
Generate TikZ/pgfplots figures for the paper from evaluation CSVs.

Run on the HPC once the relevant eval job has produced its CSV:
  python scripts/viz/generate_figures.py

Output:
  figures/results_heatmap.tex          — per-class consistency heatmap (eval_zero_shot.sh)
  figures/results_figurative_dist.tex  — label distribution on Bloomfield, stacked bar (eval_zero_shot.sh)
  figures/ablation_heatmap.tex         — per-class F1 heatmap across ablation conditions (eval_validation_set.sh)
  figures/ablation_macro_f1.tex        — macro-F1 bar chart, ablation conditions grouped by role (eval_validation_set.sh)
  figures/contrastive_impact.tex       — per-class F1, full pipeline vs +contrastive alignment (eval_validation_set.sh)
  figures/compile_figures.tex          — standalone LaTeX wrapper for whichever figures were written

A figure whose source CSV or model rows aren't ready yet is skipped with a
message instead of crashing the whole run — safe to re-run as ablation/eval
jobs land one at a time.
"""

from __future__ import annotations
import os
import pandas as pd

os.makedirs("figures", exist_ok=True)

CONS_CSV       = "data/figurative/eval_consistency.csv"
RATE_CSV       = "data/figurative/eval_figurative_rate.csv"
VALID_GOLD_CSV = "data/figurative/eval_validation_gold.csv"

CLASSES      = ["literal", "idiom", "metaphor", "simile"]
CLASS_LABELS = ["Literal", "Idiom", "Metaphor", "Simile"]

# Validated categorical slots 1 (blue) & 2 (aqua) from the dataviz palette, plus a
# neutral de-emphasis gray. Defined once per figure file so each .tex stays
# self-contained wherever it's \input.
COLOR_DEFS = r"""\definecolor{VizBlue}{HTML}{2A78D6}
\definecolor{VizAqua}{HTML}{1BAF7A}
\definecolor{VizGray}{HTML}{9E9D97}
\definecolor{VizInk}{HTML}{0B0B0B}
"""

# Ablation conditions in narrative order — jobs/ablation.sh — with the role each
# plays (baseline / component removed / component added) driving its color.
ABLATION_CONDITIONS = [
    ("Ablation: neither",         "Neither",       "VizGray"),
    ("Ablation: no TLM",          "No TLM",        "VizGray"),
    ("Ablation: no CLKD",         "No CLKD",       "VizGray"),
    ("Ablation: full",            "Full pipeline", "VizBlue"),
    ("Ablation: mono-MLM warmup", "+ Mono-MLM",    "VizAqua"),
    ("Ablation: TLM+contrastive", "+ Contrastive", "VizAqua"),
]

written_figures: list[str] = []


def skip(name: str, reason: str) -> None:
    print(f"SKIPPED {name} — {reason}")


def emit(path: str, content: str) -> None:
    with open(path, "w") as f:
        f.write(content)
    written_figures.append(os.path.splitext(os.path.basename(path))[0])
    print(f"Wrote {path}")


def load_rows(csv_path: str, models: list[str]) -> pd.DataFrame | None:
    """Read `csv_path` and return rows indexed by model, or None (with a skip
    message) if the file or any required, non-errored row is missing."""
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path).set_index("model")
    has_error = "error" in df.columns
    missing = [
        m for m in models
        if m not in df.index or (has_error and pd.notna(df.loc[m, "error"]))
    ]
    if missing:
        print(f"  missing/failed rows: {', '.join(missing)}")
        return None
    return df


# ── figure 1: per-class consistency heatmap ────────────────────────────────────

def make_consistency_heatmap() -> None:
    name = "results_heatmap"
    models_keep = [
        "XLM-R base (figurative)",
        "XLM-MLM CLKD frozen-12",
        "XLM-MLM CLKD full",
        "Glot500 CLKD direct",
        "Glot500 CLKD + TLM",
        "XLM-V CLKD direct",
    ]
    short_names = {
        "XLM-R base (figurative)": r"\textsc{xlm-r}",
        "XLM-MLM CLKD frozen-12":  r"\textsc{xlm-mlm} (f12)",
        "XLM-MLM CLKD full":       r"\textsc{xlm-mlm} (full)",
        "Glot500 CLKD direct":     r"\textsc{glot500} (dir.)",
        "Glot500 CLKD + TLM":      r"\textsc{glot500} (+tlm)",
        "XLM-V CLKD direct":       r"\textsc{xlm-v} (dir.)",
    }

    cons = load_rows(CONS_CSV, models_keep)
    if cons is None:
        skip(name, f"{CONS_CSV} not ready — run jobs/evaluate/eval_zero_shot.sh")
        return

    n_models, n_classes = len(models_keep), len(CLASSES)
    table_lines = ["x y value"]
    cell_nodes  = []
    for mi, model in enumerate(models_keep):
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
    ytick_labels    = ",".join(short_names[m] for m in models_keep)
    xtick_positions = " ".join(str(i) for i in range(n_classes))
    xtick_labels    = ",".join(CLASS_LABELS)

    content = rf"""\begin{{tikzpicture}}
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
    emit(f"figures/{name}.tex", content)


# ── figure 2: label distribution on bloomfield (stacked bar) ──────────────────

def make_figurative_dist() -> None:
    name = "results_figurative_dist"
    models_keep = [
        "XLM-R base (figurative)",
        "XLM-MLM CLKD frozen-12",
        "XLM-MLM CLKD full",
        "Glot500 CLKD direct",
        "Glot500 CLKD + TLM",
        "XLM-V CLKD direct",
    ]
    short_names = {
        "XLM-R base (figurative)": r"\textsc{xlm-r}",
        "XLM-MLM CLKD frozen-12":  r"\textsc{xlm-mlm} (f12)",
        "XLM-MLM CLKD full":       r"\textsc{xlm-mlm} (full)",
        "Glot500 CLKD direct":     r"\textsc{glot500} (dir.)",
        "Glot500 CLKD + TLM":      r"\textsc{glot500} (+tlm)",
        "XLM-V CLKD direct":       r"\textsc{xlm-v} (dir.)",
    }
    colors = ["gray!25", "NavyBlue!70", "BrickRed!70", "OliveGreen!70"]

    rate = load_rows(RATE_CSV, models_keep)
    if rate is None:
        skip(name, f"{RATE_CSV} not ready — run jobs/evaluate/eval_zero_shot.sh")
        return

    n = len(models_keep)

    def coords_for_class(cls):
        return " ".join(
            f"({i},{float(rate.loc[models_keep[i], f'rate_{cls}']):.4f})"
            for i in range(n)
        )

    bar_plots = "\n".join(
        rf"  \addplot+[ybar, fill={colors[ci]}, draw={colors[ci]}!80!black, opacity=0.9]"
        rf" coordinates {{{coords_for_class(cls)}}};"
        for ci, cls in enumerate(CLASSES)
    )
    xticklabels    = ",".join(short_names[m] for m in models_keep)
    legend_entries = ",".join(CLASS_LABELS)

    content = rf"""\begin{{tikzpicture}}
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
    emit(f"figures/{name}.tex", content)


# ── figure 3: per-class F1 heatmap across ablation conditions ─────────────────

def make_ablation_heatmap() -> None:
    name = "ablation_heatmap"
    models = [m for m, _, _ in ABLATION_CONDITIONS]
    df = load_rows(VALID_GOLD_CSV, models)
    if df is None:
        skip(name, f"{VALID_GOLD_CSV} not ready — run jobs/evaluate/eval_validation_set.sh")
        return

    cols       = CLASSES + ["macro"]
    col_labels = CLASS_LABELS + ["Macro"]
    n_models, n_cols = len(models), len(cols)

    table_lines = ["x y value"]
    cell_nodes  = []
    for mi, model in enumerate(models):
        y = n_models - 1 - mi
        for ci, col in enumerate(cols):
            key = "macro_f1" if col == "macro" else f"f1_{col}"
            val = float(df.loc[model, key])
            table_lines.append(f"{ci} {y} {val:.4f}")
            text_color = "white" if val > 0.6 else "VizInk"
            cell_nodes.append(
                rf"\node[font=\scriptsize,text={text_color}] "
                rf"at (axis cs:{ci},{y}) {{{val:.2f}}};"
            )

    table_data       = "\n    ".join(table_lines)
    cell_annotations = "\n  ".join(cell_nodes)
    ytick_positions = " ".join(str(n_models - 1 - i) for i in range(n_models))
    ytick_labels    = ",".join(label for _, label, _ in ABLATION_CONDITIONS)
    xtick_positions = " ".join(str(i) for i in range(n_cols))
    xtick_labels    = ",".join(col_labels)

    content = COLOR_DEFS + rf"""
\begin{{tikzpicture}}
\begin{{axis}}[
  width=9.5cm, height=6.4cm,
  colormap/Blues,
  colorbar,
  colorbar style={{
    ylabel={{\small F1}},
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
    emit(f"figures/{name}.tex", content)


# ── figure 4: ablation macro-F1, grouped by role (baseline / removed / added) ──

def make_ablation_macro_f1() -> None:
    name = "ablation_macro_f1"
    models = [m for m, _, _ in ABLATION_CONDITIONS]
    df = load_rows(VALID_GOLD_CSV, models)
    if df is None:
        skip(name, f"{VALID_GOLD_CSV} not ready — run jobs/evaluate/eval_validation_set.sh")
        return

    bars = [
        (i, float(df.loc[model, "macro_f1"]), label, color)
        for i, (model, label, color) in enumerate(ABLATION_CONDITIONS)
    ]

    coords_by_color: dict[str, list[str]] = {}
    for i, val, _, color in bars:
        coords_by_color.setdefault(color, []).append(f"({i},{val:.4f})")

    plot_lines = "\n".join(
        rf"\addplot[ybar, fill={color}, draw={color}!80!black, bar width=0.55cm] "
        rf"coordinates {{{' '.join(coords)}}};"
        for color, coords in coords_by_color.items()
    )
    label_nodes = "\n  ".join(
        rf"\node[font=\scriptsize, above] at (axis cs:{i},{val:.4f}) {{{val:.2f}}};"
        for i, val, _, _ in bars
    )
    xtick_positions = " ".join(str(i) for i, _, _, _ in bars)
    xticklabels     = ",".join(label for _, _, label, _ in bars)
    # legend entries follow the order colors first appear in coords_by_color
    role_by_color = {"VizGray": "Component removed", "VizBlue": "Full pipeline", "VizAqua": "Component added"}
    legend_entries = ",".join(role_by_color[c] for c in coords_by_color)

    content = COLOR_DEFS + rf"""
\begin{{tikzpicture}}
\begin{{axis}}[
  width=9.5cm, height=6cm,
  ymin=0, ymax=1,
  ymajorgrids=true,
  grid style={{dashed, gray!35}},
  ylabel={{\small Macro F1 (gold subset)}},
  xtick={{{xtick_positions}}},
  xticklabels={{{xticklabels}}},
  xticklabel style={{font=\small, rotate=30, anchor=east}},
  yticklabel style={{font=\small}},
  legend style={{font=\small, at={{(0.5,1.08)}}, anchor=south, legend columns=3}},
  legend entries={{{legend_entries}}},
  enlarge x limits=0.15,
]
{plot_lines}
{label_nodes}
\end{{axis}}
\end{{tikzpicture}}
"""
    emit(f"figures/{name}.tex", content)


# ── figure 5: impact of the InfoNCE contrastive loss, per class ───────────────

def make_contrastive_impact() -> None:
    name = "contrastive_impact"
    series = [
        ("Full pipeline (no contrastive)",           "Ablation: full",            "VizBlue"),
        (r"+ Contrastive alignment ($\alpha$=0.1)",  "Ablation: TLM+contrastive", "VizAqua"),
    ]
    models = [model for _, model, _ in series]
    df = load_rows(VALID_GOLD_CSV, models)
    if df is None:
        skip(name, f"{VALID_GOLD_CSV} not ready — run jobs/evaluate/eval_validation_set.sh")
        return

    cols       = CLASSES + ["macro"]
    col_labels = CLASS_LABELS + ["Macro"]

    def coords_for(model: str) -> str:
        return " ".join(
            f"({i},{float(df.loc[model, 'macro_f1' if c == 'macro' else f'f1_{c}']):.4f})"
            for i, c in enumerate(cols)
        )

    bar_plots = "\n".join(
        rf"\addplot[ybar, fill={color}, draw={color}!80!black] coordinates {{{coords_for(model)}}};"
        for _, model, color in series
    )
    legend_entries = ",".join(label for label, _, _ in series)
    xticklabels    = ",".join(col_labels)

    content = COLOR_DEFS + rf"""
\begin{{tikzpicture}}
\begin{{axis}}[
  ybar,
  width=9.5cm, height=6cm,
  bar width=0.35cm,
  ymin=0, ymax=1,
  ymajorgrids=true,
  grid style={{dashed, gray!35}},
  ylabel={{\small F1 (gold subset)}},
  xtick={{0,...,{len(cols) - 1}}},
  xticklabels={{{xticklabels}}},
  xticklabel style={{font=\small}},
  yticklabel style={{font=\small}},
  legend style={{font=\small, at={{(0.5,1.1)}}, anchor=south, legend columns=1}},
  legend entries={{{legend_entries}}},
  enlarge x limits=0.15,
]
{bar_plots}
\end{{axis}}
\end{{tikzpicture}}
"""
    emit(f"figures/{name}.tex", content)


# ── run ─────────────────────────────────────────────────────────────────────────

make_consistency_heatmap()
make_figurative_dist()
make_ablation_heatmap()
make_ablation_macro_f1()
make_contrastive_impact()


# ── standalone compile wrapper (only figures that were actually written) ──────

LAYOUT = [
    ("results_heatmap",         "results_figurative_dist"),
    ("ablation_heatmap",        None),
    ("ablation_macro_f1",       "contrastive_impact"),
]

rows_tex = []
for left, right in LAYOUT:
    if left not in written_figures and (right is None or right not in written_figures):
        continue
    parts = []
    if left in written_figures:
        parts.append(f"\\input{{{left}}}")
    if right and right in written_figures:
        parts.append(f"\\hspace{{1.5cm}}\n\\input{{{right}}}")
    rows_tex.append("\n".join(parts))

body = "\n\n\\vspace{1cm}\n\n".join(rows_tex) if rows_tex else "% no figures were ready to compile"

wrapper_tex = rf"""\documentclass[tikz, border=6pt]{{standalone}}
\usepackage{{pgfplots}}
\pgfplotsset{{compat=1.18}}
\usepackage[dvipsnames]{{xcolor}}

\begin{{document}}
{body}
\end{{document}}
"""

with open("figures/compile_figures.tex", "w") as f:
    f.write(wrapper_tex)
print("Wrote figures/compile_figures.tex")
print("  → compile with: cd figures && pdflatex compile_figures.tex")

if len(written_figures) < 5:
    print(f"\n{5 - len(written_figures)} figure(s) skipped — re-run once the corresponding eval job finishes.")
