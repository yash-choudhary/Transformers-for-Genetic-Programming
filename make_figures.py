"""Generate the report's figures from committed result files.

    python make_figures.py                 # all figures
    python make_figures.py step_floor      # just one

Every figure is driven by data on disk rather than typed in, so a figure cannot
drift away from the numbers in the tables.

------------------------------------------------------------------------------
EDIT THE STYLE BLOCK BELOW. Nothing else needs changing to restyle the figures.
------------------------------------------------------------------------------
"""
import glob
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter, ScalarFormatter

# ============================================================ STYLE =========

# Font. "Times New Roman" or "Cambria" to match a Word body font; the default
# sans is safer if a font is missing (matplotlib falls back silently and the
# figure ends up in a different face from the rest of the document).
FONT = "DejaVu Sans"
BASE_SIZE = 9.5

# Titles are OFF because the report puts a numbered caption under each figure;
# a title inside the image duplicates it and reads as a slide, not a report.
# Set True if you want stand-alone figures (for slides, say).
DRAW_TITLES = False

# "pdf" is vector: it stays sharp at any zoom and prints properly. Word accepts
# PNG most reliably, so both are written by default.
FORMATS = ("png", "pdf")
DPI = 400

# Figure widths in inches. 6.3 fills a 1-inch-margin A4/Letter text column.
W_FULL, W_HALF = 6.3, 5.2

# Colours. Muted enough to print, distinct enough on screen. Every series is
# ALSO distinguished by marker or hatch, so the figures survive greyscale
# printing -- worth keeping if you change these.
COL = {
    "tsgp":  "#2b6cb0",   # blue
    "std":   "#c05621",   # burnt orange
    "third": "#4a5568",   # slate
    "extra": "#2f855a",   # green
    "rule":  "#a0aec0",   # light grey for reference lines
    "note":  "#4a5568",
}

GRID_ALPHA = 0.22
OUT = "figures"

# ============================================================================

plt.rcParams.update({
    "font.family": FONT,
    "font.size": BASE_SIZE,
    "axes.titlesize": BASE_SIZE + 1,
    "axes.labelsize": BASE_SIZE,
    "xtick.labelsize": BASE_SIZE - 1,
    "ytick.labelsize": BASE_SIZE - 1,
    "legend.fontsize": BASE_SIZE - 1,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.alpha": GRID_ALPHA,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.6,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "pdf.fonttype": 42,      # embed real text, not outlines - stays selectable
})

SHORT = {"1030_ERA": "ERA", "1027_ESL": "ESL",
         "690_visualizing_galaxy": "Galaxy", "1029_LEV": "LEV",
         "529_pollen": "pollen"}
ORDER = list(SHORT)


def title(ax, text):
    if DRAW_TITLES:
        ax.set_title(text)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in FORMATS:
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  {name}: {', '.join(FORMATS)}")


def load_runs(d):
    out = defaultdict(list)
    for f in glob.glob(os.path.join(d, "runs", "*.json")):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        out[r["dataset"]].append(r)
    return out


def linlog(ax, axis="x", ticks=None):
    """Log scale with plain numbers instead of 10^n, and no minor labels."""
    (ax.set_xscale if axis == "x" else ax.set_yscale)("log")
    a = ax.get_xaxis() if axis == "x" else ax.get_yaxis()
    if ticks is not None:
        (ax.set_xticks if axis == "x" else ax.set_yticks)(ticks)
    a.set_major_formatter(ScalarFormatter())
    a.set_minor_formatter(NullFormatter())


# ---------------------------------------------------------------- figures ==
def step_floor():
    """Achieved distance against requested SD: the floor, and where the
    paper's operating point falls relative to it."""
    d = json.load(open("diagnostics/stepfloor_linear_final.json"))["floor"]
    req = sorted(float(k) for k in d)
    med = [d[k]["median"] for k in sorted(d, key=float)]

    fig, ax = plt.subplots(figsize=(W_HALF, 3.1))
    ax.plot(req, req, ls=(0, (4, 3)), color=COL["rule"], lw=1.1,
            label="if the request were honoured")
    ax.axhspan(0.60, 0.95, color=COL["tsgp"], alpha=0.07, lw=0)
    ax.plot(req, med, "o-", color=COL["tsgp"], ms=5.5, mfc="white", mew=1.4,
            label="achieved (median of 256 parents)")
    ax.axvline(0.1, color=COL["std"], lw=1.1, ls=":")
    ax.annotate("operating point\nSD$_d$ = 0.1", xy=(0.1, 0.28),
                xytext=(0.0022, 0.008), fontsize=BASE_SIZE - 1.5,
                color=COL["std"], ha="left",
                arrowprops=dict(arrowstyle="->", color=COL["std"], lw=0.9))
    ax.text(1.2e-4, 1.25, "floor $\\approx$ 0.7", fontsize=BASE_SIZE - 1.5,
            color=COL["tsgp"])
    linlog(ax, "x", [1e-4, 1e-3, 1e-2, 1e-1, 1])
    ax.set_yscale("log")
    ax.set_ylim(5e-5, 6)
    ax.set_xlabel("requested semantic distance  SD$_d$")
    ax.set_ylabel("achieved parent–offspring distance")
    title(ax, "The operator cannot take a step as small as it is asked for")
    ax.legend(frameon=False, loc="lower right")
    save(fig, "fig_step_floor")


def locality_control():
    """The control: distance to the offspring's own parent against distance to
    an unrelated parent."""
    sweep = json.load(open("diagnostics/baseline_adamw.json"))["sweep"]
    keys = sorted(sweep, key=float)
    own = [sweep[k]["d_own"] for k in keys]
    other = [sweep[k]["d_other"] for k in keys]
    x = np.arange(len(keys))
    w = 0.36

    fig, ax = plt.subplots(figsize=(W_HALF, 3.1))
    ax.bar(x - w / 2, own, w, color=COL["tsgp"], label="to its own parent")
    ax.bar(x + w / 2, other, w, facecolor="white", edgecolor=COL["third"],
           hatch="///", lw=0.9, label="to an unrelated parent")
    for i, (a, b) in enumerate(zip(own, other)):
        ax.text(i, max(a, b) * 1.35, f"{a / b:.2f}", ha="center",
                fontsize=BASE_SIZE - 2, color=COL["note"])
    ax.set_yscale("log")
    # Headroom above the tallest bar for the ratio labels, and again above
    # those for the legend, so neither lands on the data or on a tick label.
    ax.set_ylim(0.35, 4000)
    ax.set_yticks([1, 10, 100])
    ax.get_yaxis().set_major_formatter(ScalarFormatter())
    ax.get_yaxis().set_minor_formatter(NullFormatter())
    ax.set_xticks(x)
    ax.set_xticklabels([f"{float(k):g}" for k in keys])
    ax.set_xlabel("requested semantic distance  SD$_d$")
    ax.set_ylabel("semantic distance")
    ax.text(-0.42, 620, "ratio shown above each pair",
            fontsize=BASE_SIZE - 2.5, color=COL["note"])
    title(ax, "Offspring land nearer their own parent than a stranger")
    ax.legend(frameon=False, loc="upper right", ncol=1)
    save(fig, "fig_locality_control")


def size_generations(dataset="1027_ESL"):
    """stdGP grows its solutions across a run; TSGP does not.

    Drawn from the full 30-run grid rather than the two instrumented runs: a
    median over two runs is not a median worth plotting.
    """
    def series(method):
        out = []
        for f in glob.glob(f"results_v7/runs/{dataset}__{method}__*.json"):
            h = json.load(open(f)).get("best_size_history")
            if h:
                out.append(h)
        return np.array(out, dtype=float) if out else None

    sd, ts = series("stdgp"), series("tsgp")
    if sd is None or ts is None:
        print("  (skipped size_generations: no per-generation history)")
        return

    fig, ax = plt.subplots(figsize=(W_HALF, 3.0))
    g = np.arange(sd.shape[1])
    for arr, col, lab, ls in ((sd, COL["std"], "standard GP", "-"),
                              (ts, COL["tsgp"], "TSGP", (0, (5, 2)))):
        lo, mid, hi = np.percentile(arr, [25, 50, 75], axis=0)
        ax.fill_between(g, lo, hi, color=col, alpha=0.15, lw=0)
        ax.plot(g, mid, color=col, lw=2.0, ls=ls, label=lab)
    ax.set_xlabel("generation")
    ax.set_ylabel("size of best solution (nodes)")
    ax.set_xlim(0, sd.shape[1] - 1)
    ax.set_ylim(bottom=0)
    ax.text(0.02, 0.94, f"median and interquartile range, {sd.shape[0]} runs",
            transform=ax.transAxes, fontsize=BASE_SIZE - 2.5,
            color=COL["note"], va="top")
    title(ax, "Standard GP builds structure across a run; TSGP does not")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.90))
    save(fig, "fig_size_generations")


# the published medians, Anthes, Sobania and Rothlauf (2025), Table 2
PAPER_RMSE = {"1030_ERA": (0.797, 0.817), "1027_ESL": (0.379, 0.502),
              "690_visualizing_galaxy": (0.327, 0.337),
              "1029_LEV": (0.672, 0.703), "529_pollen": (0.518, 0.514)}


def rmse_vs_paper():
    """Our medians beside the published ones, four bars per dataset.

    Replaces a version that printed a rotated number on every bar; the values
    are in Table 6.1 immediately below, so the bars only have to carry the
    comparison.
    """
    ours = {}
    for ds in ORDER:
        row = {}
        for method in ("tsgp", "stdgp"):
            v = [json.load(open(f))["test_rmse"] for f in
                 glob.glob(f"results_v7/runs/{ds}__{method}__*.json")]
            if v:
                row[method] = float(np.median(v))
        if len(row) == 2:
            ours[ds] = row
    if len(ours) < len(ORDER):
        print("  (skipped rmse_vs_paper: incomplete results_v7 grid)")
        return

    fig, ax = plt.subplots(figsize=(W_FULL, 3.0))
    x = np.arange(len(ORDER))
    w = 0.2
    bars = (("TSGP (ours)", COL["tsgp"], None,
             [ours[d]["tsgp"] for d in ORDER]),
            ("TSGP (paper)", COL["tsgp"], "//",
             [PAPER_RMSE[d][0] for d in ORDER]),
            ("standard GP (ours)", COL["std"], None,
             [ours[d]["stdgp"] for d in ORDER]),
            ("standard GP (paper)", COL["std"], "//",
             [PAPER_RMSE[d][1] for d in ORDER]))
    for i, (lab, col, hatch, vals) in enumerate(bars):
        ax.bar(x + (i - 1.5) * w, vals, w * 0.88, label=lab, color=col,
               alpha=1.0 if hatch is None else 0.28, hatch=hatch,
               edgecolor=col, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[d] for d in ORDER])
    ax.set_ylabel("median test RMSE")
    ax.set_ylim(0, max(max(v[3]) for v in bars) * 1.32)
    ax.grid(axis="x", visible=False)
    ax.text(0.0, 1.02, "lower is better", transform=ax.transAxes,
            fontsize=BASE_SIZE - 2.5, color=COL["note"])
    title(ax, "Median test RMSE, ours against the paper")
    ax.legend(frameon=False, ncol=2, loc="upper right",
              bbox_to_anchor=(1.0, 1.05), columnspacing=1.2)
    save(fig, "fig_rmse_vs_paper")


def semantic_step(dataset="1027_ESL", target_norm=15.6, run="run00"):
    """The annealing result: stdGP's step shrinks across a run, TSGP's does not.

    Every generation is drawn rather than six sampled ones, and the values are
    read off the axis instead of printed beside each marker, which is what made
    the earlier version of this figure collide with its own tick labels. The
    run drawn is the one the report's numbers are quoted from (28.2 down to
    1.0), so the figure and the text continue to agree.
    """
    def series(method):
        f = f"results_instr_base/{dataset}__{method}__{run}.json"
        if not os.path.exists(f):
            return None
        h = json.load(open(f)).get("pair_distance_median")
        return np.array(h, dtype=float) if h else None

    sd, ts = series("stdgp"), series("tsgp")
    if sd is None or ts is None:
        print("  (skipped semantic_step: run instrument.py first)")
        return
    # generation 0 is the initial population: no parent, so no step to measure
    sd, ts = sd[1:], ts[1:]

    def smooth(v, w=5):
        pad = np.pad(v, (w // 2, w // 2), mode="edge")
        return np.array([np.nanmedian(pad[i:i + w]) for i in range(len(v))])

    fig, ax = plt.subplots(figsize=(W_HALF, 3.0))
    g = np.arange(1, len(sd) + 1)
    ax.axhline(target_norm, color=COL["rule"], lw=1.0, ls=(0, (4, 3)))
    ax.text(len(sd) * 0.5, target_norm - 0.8,
            f"magnitude of the target signal ({target_norm})", ha="center",
            va="top", fontsize=BASE_SIZE - 2.5, color=COL["note"])
    # the generation-to-generation series is noisy enough to read as hatching;
    # the faint line is the measurement, the bold one a five-generation median
    for m, col, lab, ls in ((ts, COL["tsgp"], "TSGP", (0, (5, 2))),
                            (sd, COL["std"], "standard GP", "-")):
        ax.plot(g, m, color=col, lw=0.9, alpha=0.30)
        ax.plot(g, smooth(m), color=col, lw=2.1, ls=ls, label=lab)
    ax.annotate(f"{sd[0]:.1f}", (g[0], sd[0]), color=COL["std"],
                textcoords="offset points", xytext=(5, 2),
                fontsize=BASE_SIZE - 2)
    ax.annotate(f"{sd[-1]:.1f}", (g[-1], sd[-1]), color=COL["std"], ha="right",
                textcoords="offset points", xytext=(-3, 7),
                fontsize=BASE_SIZE - 2)
    ax.set_xlabel("generation")
    ax.set_ylabel("parent-to-offspring semantic distance")
    ax.set_xlim(1, len(sd))
    ax.set_ylim(0, np.nanmax([sd, ts]) * 1.12)
    title(ax, "Semantic step size across generations")
    ax.legend(frameon=False, loc="upper right", ncol=2,
              bbox_to_anchor=(1.0, 1.02))
    save(fig, "fig_semantic_step")


# ------------------------------------------------------- schematic helpers
def _boxer(fig, ax):
    """Return box/arrow helpers that size a box to the text it contains, so
    editing a label can never push text outside its border."""
    def box(cx, cy, text, ec, fs, fc="white", pad_x=0.012, pad_y=0.030,
            weight="normal"):
        t = ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
                    zorder=3, linespacing=1.45, fontweight=weight)
        fig.canvas.draw()
        bb = t.get_window_extent().transformed(ax.transData.inverted())
        ax.add_patch(plt.Rectangle(
            (bb.x0 - pad_x, bb.y0 - pad_y), bb.width + 2 * pad_x,
            bb.height + 2 * pad_y, facecolor=fc, edgecolor=ec, lw=1.0,
            zorder=2))
        return dict(l=bb.x0 - pad_x, r=bb.x1 + pad_x,
                    b=bb.y0 - pad_y, t=bb.y1 + pad_y, cx=cx, cy=cy)

    def varrow(x, y1, y2, label=None):
        ax.annotate("", xy=(x, y2), xytext=(x, y1),
                    arrowprops=dict(arrowstyle="->", lw=0.9,
                                    color=COL["third"]))
        if label:
            ax.text(x + 0.012, (y1 + y2) / 2, label, ha="left", va="center",
                    fontsize=BASE_SIZE - 3, color=COL["note"])

    def harrow(x1, x2, y, label=None):
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="->", lw=0.9,
                                    color=COL["third"]))
        if label:
            ax.text((x1 + x2) / 2, y + 0.022, label, ha="center",
                    fontsize=BASE_SIZE - 3, color=COL["note"])
    return box, varrow, harrow


def transformer_architecture():
    """The encoder-decoder transformer, at the level of detail the report needs.

    Stack labels are rotated down the left edge of each dashed region rather
    than sitting above it: a horizontal label there collides with whatever box
    is above, and the collision only appears once the boxes are auto-sized.
    """
    fig, ax = plt.subplots(figsize=(W_FULL, 4.6))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    box, varrow, harrow = _boxer(fig, ax)
    fs = BASE_SIZE - 2.5
    xe, xd = 0.30, 0.755

    # dashed regions cover only the repeated layers, not the stack's in/outputs
    ax.add_patch(plt.Rectangle((0.145, 0.215), 0.31, 0.375, facecolor="#f6f9fc",
                               edgecolor=COL["third"], lw=0.8, ls=(0, (4, 3)),
                               zorder=1))
    ax.add_patch(plt.Rectangle((0.595, 0.175), 0.32, 0.545, facecolor="#f6f9fc",
                               edgecolor=COL["third"], lw=0.8, ls=(0, (4, 3)),
                               zorder=1))
    ax.text(0.128, 0.403, "ENCODER  $\\times N$", rotation=90, ha="center",
            va="center", fontsize=fs, color=COL["note"], fontweight="bold")
    # on the RIGHT of the decoder region: on the left it sits exactly where the
    # cross-attention arrow's label goes
    ax.text(0.932, 0.448, "DECODER  $\\times N$", rotation=90, ha="center",
            va="center", fontsize=fs, color=COL["note"], fontweight="bold")

    box(xe, 0.075, "input tokens\n+ positional encoding", COL["third"], fs)
    varrow(xe, 0.135, 0.245)
    e1 = box(xe, 0.305, "multi-head\nself-attention", COL["tsgp"], fs)
    varrow(xe, e1["t"], 0.445, "add & norm")
    e2 = box(xe, 0.505, "feed-forward", COL["tsgp"], fs)
    varrow(xe, e2["t"], 0.645, "add & norm")
    box(xe, 0.705, "encoded representation", COL["third"], fs)

    box(xd, 0.075, "output tokens so far\n+ positional encoding", COL["third"],
        fs)
    varrow(xd, 0.135, 0.205)
    d1 = box(xd, 0.265, "masked multi-head\nself-attention", COL["extra"], fs)
    varrow(xd, d1["t"], 0.375, "add & norm")
    d2 = box(xd, 0.435, "cross-attention", COL["extra"], fs)
    varrow(xd, d2["t"], 0.545, "add & norm")
    d3 = box(xd, 0.605, "feed-forward", COL["extra"], fs)
    varrow(xd, d3["t"], 0.755, "add & norm")
    box(xd, 0.815, "linear + softmax\n$\\rightarrow$ next-token distribution",
        COL["third"], fs)

    harrow(0.455, 0.665, 0.435)
    ax.text(0.560, 0.472, "keys, values", ha="center", fontsize=fs - 0.5,
            color=COL["note"])
    save(fig, "fig_transformer_architecture")


def tsgp_model():
    """The specific model this study trains, with its actual configuration.

    Laid out over two rows. A single row of seven boxes does not fit the text
    column: the boxes overlap, and because each arrow is drawn between two
    auto-sized borders the overlap silently reverses the arrows.
    """
    fig, ax = plt.subplots(figsize=(W_FULL, 3.6))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    box, varrow, harrow = _boxer(fig, ax)
    fs = BASE_SIZE - 2.5
    y1, y2 = 0.78, 0.40

    a1 = box(0.145, y1, "parent expression\n$\\rightarrow$ prefix tokens",
             COL["third"], fs)
    a2 = box(0.500, y1, "embedding, d = 128\n+ positional encoding\n"
                        "+ desired distance SD$_d$", COL["tsgp"], fs)
    a3 = box(0.845, y1, "encoder $\\times 2$\n8 heads", COL["tsgp"], fs)
    harrow(a1["r"] + 0.01, a2["l"] - 0.01, y1)
    harrow(a2["r"] + 0.01, a3["l"] - 0.01, y1)

    b1 = box(0.145, y2, "decoder $\\times 2$\ncross-attends the encoder",
             COL["extra"], fs)
    b2 = box(0.470, y2, "linear\n$\\rightarrow$ 22 tokens", COL["third"], fs)
    b3 = box(0.685, y2, "syntax control\nmask invalid", COL["std"], fs)
    b4 = box(0.905, y2, "sample\nnext token", COL["std"], fs)
    harrow(b1["r"] + 0.01, b2["l"] - 0.01, y2)
    harrow(b2["r"] + 0.01, b3["l"] - 0.01, y2)
    harrow(b3["r"] + 0.01, b4["l"] - 0.01, y2)

    def route(points, dashed=False):
        """Orthogonal polyline with an arrowhead on the last segment. A curved
        connector between two rows sweeps across the middle of the figure and
        crosses whatever boxes are there; right-angled routing cannot."""
        style = dict(color=COL["third"], lw=0.9,
                     ls=(0, (3, 2)) if dashed else "-")
        for (x1, y1_), (x2, y2_) in zip(points[:-1], points[1:-1] + [points[-1]]):
            ax.plot([x1, x2], [y1_, y2_], zorder=1, **style)
        ax.annotate("", xy=points[-1], xytext=points[-2],
                    arrowprops=dict(arrowstyle="-|>", lw=0.9,
                                    color=COL["third"],
                                    ls=(0, (3, 2)) if dashed else "-"))

    # encoder output drops to the decoder row, routed clear of both rows
    ymid = (a3["b"] + b1["t"]) / 2
    route([(a3["cx"], a3["b"] - 0.012), (a3["cx"], ymid),
           (b1["cx"], ymid), (b1["cx"], b1["t"] + 0.012)])
    ax.text(0.505, ymid + 0.028, "keys, values", ha="center", fontsize=fs - 0.5,
            color=COL["note"])

    # the sampled token is appended and the decoder runs again
    ylow = b1["b"] - 0.13
    route([(b4["cx"], b4["b"] - 0.012), (b4["cx"], ylow),
           (b1["cx"], ylow), (b1["cx"], b1["b"] - 0.012)], dashed=True)
    ax.text(0.520, ylow - 0.055, "appended, then decoded again until the tree "
                                 "is complete", ha="center", fontsize=fs - 0.5,
            color=COL["note"], style="italic")

    ax.text(0.5, 0.955, "934,000 parameters  ·  vocabulary of 22 tokens  ·  "
                        "sequences capped at 100  ·  AdamW, lr $10^{-3}$, "
                        "8 epochs",
            ha="center", fontsize=fs, color=COL["note"])
    ax.text(0.5, 0.030, "trained once on 5M semantically similar expression "
                        "pairs, then reused unchanged for every search",
            ha="center", fontsize=fs, color=COL["tsgp"])
    save(fig, "fig_tsgp_model")


def size_vs_accuracy():
    """The accuracy/compactness trade at the centre of the extension."""
    std, tsgp = load_runs("results_clf"), load_runs("results_clf_k1")
    ml = {}
    p = "results_clf_ml/ml_baselines.json"
    if os.path.exists(p):
        for r in json.load(open(p)):
            ml.setdefault(r["dataset"], defaultdict(list))[r["model"]].append(r)

    fig, axes = plt.subplots(1, 5, figsize=(W_FULL * 1.55, 3.1))
    for ax, ds in zip(axes, ORDER):
        pts = []
        if ds in std:
            pts.append(("standard GP", np.median([r["size"] for r in std[ds]]),
                        np.median([r["test_acc"] for r in std[ds]]),
                        COL["std"], "s"))
        if ds in tsgp:
            pts.append(("TSGP", np.median([r["size"] for r in tsgp[ds]]),
                        np.median([r["test_acc"] for r in tsgp[ds]]),
                        COL["tsgp"], "o"))
        for key, lab, col, mk in (("logreg", "logistic regression",
                                   COL["third"], "^"),
                                  ("tree-d3", "decision tree (d3)",
                                   COL["extra"], "D")):
            if ds in ml and key in ml[ds]:
                pts.append((lab, np.median([r["size"] for r in ml[ds][key]]),
                            np.median([r["acc"] for r in ml[ds][key]]),
                            col, mk))
        maj = float(np.median([r["majority_baseline"] for r in std.get(ds, [])]
                              or [np.nan]))
        if np.isfinite(maj):
            ax.axhline(maj, color=COL["rule"], lw=0.9, ls=(0, (4, 3)))
        for lab, sz, acc, col, mk in pts:
            ax.scatter(sz, acc, s=46, color=col, marker=mk, zorder=3,
                       edgecolor="white", linewidth=0.7, label=lab)
        # five panels across one text width: three ticks is as many as fit
        # without the numbers running into one another
        linlog(ax, "x", [5, 25, 100])
        ax.set_xlim(4, 260)
        if np.isfinite(maj):
            top = max([p[2] for p in pts] + [maj]) + 0.035
            ax.set_ylim(maj - 0.025, min(top, 1.01))
        ax.set_title(SHORT[ds], fontsize=BASE_SIZE + 0.5)
        ax.set_xlabel("solution size (nodes)")
    axes[0].set_ylabel("test accuracy")
    # reserve the strip at the foot of the figure rather than hanging the
    # legend below it, which put it across the axis labels
    fig.tight_layout(rect=(0, 0.17, 1, 1), w_pad=1.4)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.055))
    fig.text(0.5, 0.008, "dashed line: majority-class baseline",
             ha="center", fontsize=BASE_SIZE - 2, color=COL["note"])
    if DRAW_TITLES:
        fig.suptitle("Accuracy against solution size, median-split task", y=1.05)
    save(fig, "fig_size_vs_accuracy")


def task_difficulty():
    """Why a second benchmark construction was needed."""
    med_p = "results_clf_ml/ml_baselines.json"
    mid_p = "results_clf_ml/ml_baselines_middle.json"
    if not (os.path.exists(med_p) and os.path.exists(mid_p)):
        print("  (skipped task_difficulty: run baselines_ml.py for both tasks)")
        return

    def margins(path):
        by = defaultdict(lambda: defaultdict(list))
        for r in json.load(open(path)):
            by[r["dataset"]][r["model"]].append(r["acc"])
        return {ds: {k: np.median(v) - np.median(m["majority"])
                     for k, v in m.items() if k != "majority"}
                for ds, m in by.items()}

    med, mid = margins(med_p), margins(mid_p)
    models = [("logreg", "logistic regression", COL["third"], "///"),
              ("tree-d3", "decision tree (d3)", COL["extra"], None),
              ("rf-100", "random forest", COL["tsgp"], None)]
    x = np.arange(len(ORDER))
    w = 0.26

    fig, axes = plt.subplots(1, 2, figsize=(W_FULL * 1.35, 2.9), sharey=True)
    for ax, data, lab in ((axes[0], med, "Median-split task"),
                          (axes[1], mid, "Middle-band task")):
        for j, (key, name, col, hatch) in enumerate(models):
            vals = [data.get(ds, {}).get(key, 0) for ds in ORDER]
            ax.bar(x + (j - 1) * w, vals, w, label=name,
                   facecolor="white" if hatch else col,
                   edgecolor=col, hatch=hatch, lw=0.9)
        ax.axhline(0, color="#2d3748", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT[d] for d in ORDER])
        ax.set_title(lab, fontsize=BASE_SIZE + 0.5)
    axes[0].set_ylabel("accuracy above majority baseline")
    axes[1].legend(frameon=False, loc="upper right")
    save(fig, "fig_task_difficulty")


def decision_value():
    """How a real-valued expression becomes a classifier."""
    fig, ax = plt.subplots(figsize=(W_FULL, 2.25))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Boxes are sized from their text rather than hand-placed, so changing a
    # label cannot push the text outside its border.
    def box(cx, cy, text, ec, fs, pad_x=0.018, pad_y=0.055):
        t = ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
                    zorder=3, linespacing=1.5)
        fig.canvas.draw()
        bb = t.get_window_extent().transformed(ax.transData.inverted())
        ax.add_patch(plt.Rectangle(
            (bb.x0 - pad_x, bb.y0 - pad_y),
            bb.width + 2 * pad_x, bb.height + 2 * pad_y,
            facecolor="white", edgecolor=ec, lw=1.1, zorder=2))
        return bb.x0 - pad_x, bb.x1 + pad_x

    def arrow(x1, x2, y, label):
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="->", lw=1.0,
                                    color=COL["third"]))
        ax.text((x1 + x2) / 2, y + 0.07, label, ha="center",
                fontsize=BASE_SIZE - 2.5, color=COL["note"])

    fs = BASE_SIZE - 1.5
    _, r1 = box(0.135, 0.66, "expression tree\nadd(mul(x0, 0.3), x2)",
                COL["third"], fs)
    l2, r2 = box(0.475, 0.66, "$f(\\mathbf{x}) \\in \\mathbb{R}$\n"
                              "decision value", COL["tsgp"], fs)
    arrow(r1 + 0.012, l2 - 0.012, 0.66, "evaluate")
    l3, _ = box(0.815, 0.79, "$f(\\mathbf{x}) > 0 \\;\\Rightarrow\\; +1$",
                COL["extra"], fs)
    box(0.815, 0.53, "$f(\\mathbf{x}) < 0 \\;\\Rightarrow\\; -1$",
        COL["std"], fs)
    arrow(r2 + 0.012, l3 - 0.012, 0.66, "sign")

    ax.text(0.5, 0.26, "$f(\\mathbf{x}) = 0$ is the decision boundary;   "
                       "$|f(\\mathbf{x})|$ is confidence",
            ha="center", fontsize=BASE_SIZE - 1, color=COL["note"])
    ax.text(0.5, 0.08, "semantics stay continuous, so Euclidean semantic "
                       "distance keeps its meaning",
            ha="center", fontsize=BASE_SIZE - 1, color=COL["tsgp"])
    save(fig, "fig_decision_value")


def semantic_distance():
    """What semantics and semantic distance actually are."""
    rng = np.random.default_rng(3)
    probe = np.arange(1, 9)
    f1 = np.array([1.2, -0.4, 0.9, 2.1, -1.1, 0.3, 1.7, -0.6])
    f2 = f1 + rng.normal(0, 0.28, size=8)
    f3 = np.array([-1.4, 2.2, -0.7, 0.4, 1.9, -1.8, 0.2, 2.4])

    fig, axes = plt.subplots(1, 2, figsize=(W_FULL * 1.15, 2.7),
                             gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    for p, a, b in zip(probe, f1, f2):
        ax.plot([p, p], [a, b], color=COL["rule"], lw=0.8, zorder=1)
    ax.plot(probe, f1, "o-", color=COL["tsgp"], ms=5, label="$f_i$")
    ax.plot(probe, f2, "s--", color=COL["tsgp"], alpha=0.55, ms=4.5,
            label="$f_j$  (near)")
    ax.plot(probe, f3, "^:", color=COL["std"], ms=4.5, label="$f_k$  (far)")
    ax.set_xlabel("probe input")
    ax.set_ylabel("output value")
    ax.legend(frameon=False, ncol=3, loc="lower center")
    ax.set_ylim(-2.6, 2.9)
    title(ax, "Semantics: outputs on fixed probe inputs")

    ax = axes[1]
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.80, "$SD(f_i, f_j) = \\| \\, s(f_i) - s(f_j) \\, \\|_2$",
            ha="center", fontsize=BASE_SIZE + 2.5)
    ax.text(0.5, 0.55, f"$SD(f_i, f_j) = {np.linalg.norm(f1 - f2):.2f}$"
                       "\u2003(near)",
            ha="center", fontsize=BASE_SIZE + 0.5, color=COL["tsgp"])
    ax.text(0.5, 0.40, f"$SD(f_i, f_k) = {np.linalg.norm(f1 - f3):.2f}$"
                       "\u2003(far)",
            ha="center", fontsize=BASE_SIZE + 0.5, color=COL["std"])
    ax.text(0.5, 0.13, "Two expressions are semantically similar when these\n"
                       "vectors are close, however differently they are built.",
            ha="center", fontsize=BASE_SIZE - 1, color=COL["note"])
    save(fig, "fig_semantic_distance")


FIGURES = {
    "transformer_architecture": transformer_architecture,
    "tsgp_model": tsgp_model,
    "semantic_distance": semantic_distance,
    "size_generations": size_generations,
    "semantic_step": semantic_step,
    "rmse_vs_paper": rmse_vs_paper,
    "step_floor": step_floor,
    "locality_control": locality_control,
    "decision_value": decision_value,
    "task_difficulty": task_difficulty,
    "size_vs_accuracy": size_vs_accuracy,
}

if __name__ == "__main__":
    want = sys.argv[1:] or list(FIGURES)
    unknown = [w for w in want if w not in FIGURES]
    if unknown:
        print(f"unknown figure(s): {unknown}\navailable: {list(FIGURES)}")
        raise SystemExit(2)
    print(f"writing to {OUT}/  (titles {'on' if DRAW_TITLES else 'off'}, "
          f"formats {list(FORMATS)})")
    for name in want:
        try:
            FIGURES[name]()
        except Exception as e:
            print(f"  FAILED {name}: {type(e).__name__}: {e}")
