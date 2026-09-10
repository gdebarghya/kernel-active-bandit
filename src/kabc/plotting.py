"""Shared paper-ready figure style, used by all three notebooks
(docs/experiments.md's repository layout section).

Notebooks only load results from `results/expN/*.parquet` and plot -- they
never recompute (CLAUDE.md). This module owns the one thing that must stay
IDENTICAL across notebooks 01/02/03: which color/marker/linestyle means which
algorithm, and what "paper-ready" typography looks like.

Usage:
    from kabc.plotting import set_style, ALGO_COLORS, ALGO_LABELS, ALGO_ORDER, savefig, legend_for_algorithms

    set_style()
    fig, ax = plt.subplots()
    ...
    legend_for_algorithms(ax, ["kabc", "kabc-sg", "acb"])
    savefig(fig, "exp1_fig1_tau_vs_delta")
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# Canonical algorithm identity, fixed across notebooks 01/02/03 and every
# figure in the paper. Colors are Okabe-Ito-derived (colorblind-safe,
# distinguishable in greyscale print): kabc/kabc-sg/kabc-oracle share a
# warm (vermillion/orange/black) family since they are "our method and its
# variants"; acb/tas-fw/unif-kkm get cool, visually distinct competitor
# colors. Order controls legend and box-plot ordering everywhere.
ALGO_ORDER = ["kabc", "kabc-sg", "kabc-oracle", "acb", "tas-fw", "unif-kkm"]

ALGO_COLORS = {
    "kabc": "#D55E00",  # vermillion -- our main method, bold
    "kabc-sg": "#E69F00",  # orange -- our method, sub-Gaussian ablation
    "kabc-oracle": "#000000",  # black -- oracle reference line, not a "real" method
    "acb": "#0072B2",  # blue -- main competitor (mean-based)
    "tas-fw": "#009E73",  # green -- finite-alphabet competitor
    "unif-kkm": "#999999",  # grey -- naive non-adaptive baseline
}

ALGO_LABELS = {
    "kabc": "KABC",
    "kabc-sg": "KABC-SG",
    "kabc-oracle": "KABC-oracle",
    "acb": "ACB",
    "tas-fw": "TaS-FW",
    "unif-kkm": "Unif-KKM",
}

# Linestyle carries a second channel of identity so greyscale figures still
# separate the three KABC variants (which share the warm color family) and
# marks the two reference-only curves (oracle, unif-kkm) as dashed/dotted.
ALGO_LINESTYLES = {
    "kabc": "-",
    "kabc-sg": "--",
    "kabc-oracle": ":",
    "acb": "-",
    "tas-fw": "-",
    "unif-kkm": "--",
}

ALGO_MARKERS = {
    "kabc": "o",
    "kabc-sg": "s",
    "kabc-oracle": "^",
    "acb": "D",
    "tas-fw": "v",
    "unif-kkm": "x",
}


def set_style() -> None:
    """Apply the shared paper-ready matplotlib rcParams. Call once at the
    top of every notebook, before any figure is created.

    Serif/LaTeX-like look (no LaTeX install required -- uses matplotlib's
    built-in STIX fonts via mathtext/serif family), small base font sizes
    suited to a two-column camera-ready figure, thin spines, and no minor
    gridlines.
    """
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.titlesize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.5,
            "grid.alpha": 0.3,
            "axes.grid": True,
            "axes.axisbelow": True,
            "lines.linewidth": 1.5,
            "lines.markersize": 5,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.transparent": False,
        }
    )


def style_for(algorithm: str) -> dict:
    """Return a kwargs dict (color, linestyle, marker, label) for a single
    algorithm's plot call, e.g. `ax.plot(x, y, **style_for("kabc"))`."""
    return {
        "color": ALGO_COLORS[algorithm],
        "linestyle": ALGO_LINESTYLES[algorithm],
        "marker": ALGO_MARKERS[algorithm],
        "label": ALGO_LABELS[algorithm],
    }


def legend_for_algorithms(ax, algorithms: list[str], **legend_kwargs) -> None:
    """Build a legend on `ax` with a fixed algorithm ordering (per
    ALGO_ORDER) regardless of the order lines were plotted in, so every
    figure's legend reads the same way."""
    ordered = [a for a in ALGO_ORDER if a in algorithms]
    handles = [
        plt.Line2D(
            [],
            [],
            color=ALGO_COLORS[a],
            linestyle=ALGO_LINESTYLES[a],
            marker=ALGO_MARKERS[a],
            label=ALGO_LABELS[a],
        )
        for a in ordered
    ]
    ax.legend(handles=handles, **legend_kwargs)


def savefig(fig, name: str, fig_dir: str | Path = "results/figures") -> Path:
    """Save `fig` as both PDF (vector, for the paper) and PNG (quick
    preview) under `fig_dir/name.{pdf,png}`. Creates `fig_dir` if needed.
    Returns the PDF path.
    """
    out_dir = Path(fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{name}.pdf"
    fig.savefig(pdf_path)
    fig.savefig(out_dir / f"{name}.png")
    return pdf_path
