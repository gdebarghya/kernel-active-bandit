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
# distinguishable in greyscale print): kabc/kabc-sg share a warm
# (vermillion/orange) family since they are "our method and its variants";
# acb/tas-fw/unif-kkm get cool, visually distinct competitor colors. Order
# controls legend and box-plot ordering everywhere.
ALGO_ORDER = [
    "kabc", "kabc-0.5x", "kabc-sg", "kabc-sg-0.5x",
    "acb", "acb-0.5x", "acb-gaussian", "tas-fw", "tas-fw-0.1x", "tas-fw-0.2x", "tas-fw-calibrated",
    "unif-kkm-1x", "unif-kkm-0.5x",
    "unif-kkm-highfixed", "unif-kkm-midfixed", "unif-kkm-lowfixed", "unif-kkm-naive",
]

ALGO_COLORS = {
    "kabc": "#D55E00",  # vermillion -- our main method, bold
    # Stopping-rule-conservativeness check (docs/experiments.md, Experiment 1's
    # REDESIGN note / Experiment 3's REVISION 4): each of kabc/kabc-sg/acb
    # re-run at 0.5x ITS OWN realized tau -- MULTIPLIER CHANGED from an
    # earlier 0.9x (2026-09-14): KABC's doubling trick quantizes its own
    # realized tau to a small set of round-boundary checkpoints, and the
    # FINAL round alone typically costs roughly half of the total realized
    # tau -- so a 0.9x cut almost always lands in the same "one checkpoint
    # short" bucket as a much larger cut would (there is no way to construct
    # a real "kabc re-run with 10% less budget" that isn't actually testing
    # "one fewer doubling round" against an independent draw). 0.5x makes
    # what is actually being tested explicit: a meaningfully smaller,
    # still-real budget cut, not a false-precision "10% margin" the doubling
    # schedule cannot honor. See docs/experiments.md's Experiment 1 REDESIGN
    # note for the full reasoning and the concrete evidence that motivated
    # this change. Same color family as its base algorithm, a
    # lighter/softer shade, so each reads as "base algorithm, less budget"
    # at a glance rather than a wholly distinct method.
    "kabc-0.5x": "#F4A582",
    "kabc-sg": "#E69F00",  # orange -- our method, sub-Gaussian ablation
    "kabc-sg-0.5x": "#F0C674",
    # kabc-oracle REMOVED (was "#000000"): no current experiment uses it --
    # Experiment 2's redesign dropped it along with the "price of
    # adaptivity" figure it existed for, and it was never used by
    # Experiment 1/3. Removed entirely rather than kept as a legacy
    # placeholder, same principle as unif-kkm-full/unif-kkm-2x above.
    "acb": "#0072B2",  # blue -- main competitor (mean-based)
    "acb-0.5x": "#89BEDC",
    # acb-gaussian (Experiment 1 ONLY, docs/experiments.md's REVISION 6):
    # the ACB paper's own Gaussian-specialized, non-adaptive variant
    # (Appendix A) -- a DIFFERENT algorithm from acb/acb-0.5x above (no
    # `-0.5x` counterpart, no calibration), not one more budget tier of it,
    # so it deliberately does NOT share acb's blue family -- reddish
    # purple (Okabe-Ito), visually and semantically distinct from both the
    # kabc warm family and the acb/unif-kkm cool families.
    "acb-gaussian": "#CC79A7",
    "tas-fw": "#009E73",  # green -- finite-alphabet competitor
    # tas-fw-0.1x/-0.2x (ad hoc probe, scripts/run_experiment3_tasfw_lowbudget.py,
    # results/exp3_tasfw_lowbudget/results.parquet -- separate from exp3's own
    # production results/exp3_1|2|3): tas-fw re-run at a FIXED budget_cap of
    # 0.1x/0.2x of exp3's production budget_cap=15000 (1500/3000), not a
    # multiple of any run's own realized tau. Same green family as tas-fw,
    # lighter as budget shrinks (0.2x > 0.1x), matching the unif-kkm/kabc-0.5x
    # convention of "lighter = less budget, same hue = same base algorithm".
    "tas-fw-0.2x": "#4DB38F",
    "tas-fw-0.1x": "#A3D9C4",
    # tas-fw-calibrated (scripts/calibrate_tas_fw.py + scripts/run_experiment3_tasfw_calibrated.py):
    # tas-fw at exp3's REAL budget_cap=15000 with a properly calibrated (not
    # just budget-capped) beta(t,delta) threshold -- a darker, more saturated
    # shade than the base tas-fw color (inverting the "-0.1x/-0.2x lighter"
    # convention above) since this is not a cruder variant, it's the
    # intended replacement for the uncalibrated tas-fw row.
    "tas-fw-calibrated": "#00694A",
    # unif-kkm variants (tied to a base algorithm's own realized tau, not an
    # independent sweep -- see docs/experiments.md's Experiment 1/2/3
    # sections): grey family, shaded darker to lighter as budget shrinks (1x
    # > 0.5x), so the visual ordering matches the budget ordering at a
    # glance. Bare "unif-kkm" (no tau-multiplier suffix) and
    # `unif-kkm-full`/`unif-kkm-2x` (earlier designs, since superseded) were
    # removed entirely once no current config or surviving results
    # referenced them -- not kept as legacy placeholders.
    "unif-kkm-1x": "#CCCCCC",
    "unif-kkm-0.5x": "#E5E5E5",
    # Experiment 4's tau_kabc-BLIND fixed-budget variants (docs/experiments.md,
    # Experiment 4 "Algorithms" -- NOT a multiple of tau_kabc like 1x/0.5x
    # above, each condition's own doubling-trick-derived constant instead):
    # kept in the same grey family (still unif-kkm/k-means), shaded by
    # budget tier (highfixed > midfixed > lowfixed), so the ordering reads
    # the same way at a glance. unif-kkm-naive (n=50/arm, no reference to
    # tau_kabc or any budget_cap at all) gets a distinct pale blue-grey --
    # it is categorically different (oracle-free), not just another budget
    # tier of the same family.
    "unif-kkm-highfixed": "#B0B0B0",
    "unif-kkm-midfixed": "#D5D5D5",
    "unif-kkm-lowfixed": "#EDEDED",
    "unif-kkm-naive": "#B8C4CC",
}

ALGO_LABELS = {
    "kabc": "KABC",
    "kabc-0.5x": "KABC (0.5x τ)",
    "kabc-sg": "KABC-SG",
    "kabc-sg-0.5x": "KABC-SG (0.5x τ)",
    "acb": "ACB",
    "acb-0.5x": "ACB (0.5x τ)",
    "acb-gaussian": "ACB-Gaussian",
    "tas-fw": "TaS-FW",
    "tas-fw-0.1x": "TaS-FW (0.1x cap)",
    "tas-fw-0.2x": "TaS-FW (0.2x cap)",
    "tas-fw-calibrated": "TaS-FW (calibrated)",
    "unif-kkm-1x": "Unif-KKM (1x)",
    "unif-kkm-0.5x": "Unif-KKM (0.5x)",
    "unif-kkm-highfixed": "Unif-KKM (high fixed)",
    "unif-kkm-midfixed": "Unif-KKM (mid fixed)",
    "unif-kkm-lowfixed": "Unif-KKM (low fixed)",
    "unif-kkm-naive": "Unif-KKM (naive, n=50/arm)",
}

# Linestyle carries a second channel of identity so greyscale figures still
# separate variants that share a color family, and marks the reference-only
# curve (unif-kkm) as dashed. Every "own-base 0.5x self-comparison" row
# (kabc-0.5x, kabc-sg-0.5x, acb-0.5x) shares "-." as a unifying "this is a
# 0.5x-of-its-own-tau variant" signal, distinct from unif-kkm's separate
# budget-tier styling below.
ALGO_LINESTYLES = {
    "kabc": "-",
    "kabc-0.5x": "-.",
    "kabc-sg": "--",
    "kabc-sg-0.5x": "-.",
    "acb": "-",
    "acb-0.5x": "-.",
    "acb-gaussian": "-",  # independent base, like kabc/kabc-sg/acb -- not a 0.5x-derived variant
    "tas-fw": "-",
    "tas-fw-0.1x": "-.",
    "tas-fw-0.2x": "-.",
    "tas-fw-calibrated": "-",
    "unif-kkm-1x": ":",
    "unif-kkm-0.5x": ":",
    "unif-kkm-highfixed": ":",
    "unif-kkm-midfixed": ":",
    "unif-kkm-lowfixed": ":",
    "unif-kkm-naive": ":",
}

ALGO_MARKERS = {
    "kabc": "o",
    "kabc-0.5x": "h",
    "kabc-sg": "s",
    "kabc-sg-0.5x": "p",
    "acb": "D",
    "acb-0.5x": "<",
    "acb-gaussian": "^",
    "tas-fw": "v",
    "tas-fw-0.1x": "3",
    "tas-fw-0.2x": ">",
    "tas-fw-calibrated": "P",
    "unif-kkm-1x": "*",
    "unif-kkm-0.5x": "d",
    "unif-kkm-highfixed": "1",
    "unif-kkm-midfixed": "2",
    "unif-kkm-lowfixed": "x",
    "unif-kkm-naive": "+",
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
