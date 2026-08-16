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
        ax.text(i, max(a, b) * 1.25, f"{a / b:.2f}", ha="center",
                fontsize=BASE_SIZE - 2, color=COL["note"])
    ax.set_yscale("log")
    ax.set_ylim(0.4, 200)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{float(k):g}" for k in keys])
    ax.set_xlabel("requested semantic distance  SD$_d$")
    ax.set_ylabel("semantic distance")
    title(ax, "Offspring land nearer their own parent than a stranger")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "fig_locality_control")


def size_generations():
    """stdGP grows its solutions across a run; TSGP does not."""
    def series(pat):
        return [json.load(open(f))["best_size"]
                for f in glob.glob(f"results_instr_base/{pat}")]
    sd, ts = series("*stdgp*.json"), series("*tsgp*.json")
    if not (sd and ts):
        print("  (skipped size_generations: no instrumented runs)")
        return

    fig, ax = plt.subplots(figsize=(W_HALF, 3.0))
    for s in sd:
        ax.plot(s, color=COL["std"], alpha=0.25, lw=0.8)
    for s in ts:
        ax.plot(s, color=COL["tsgp"], alpha=0.25, lw=0.8)
    ax.plot(np.median(np.array(sd), axis=0), color=COL["std"], lw=2.2,
            label="standard GP")
    ax.plot(np.median(np.array(ts), axis=0), color=COL["tsgp"], lw=2.2,
            ls=(0, (5, 2)), label="TSGP")
    ax.set_xlabel("generation")
    ax.set_ylabel("size of best solution (nodes)")
    ax.set_xlim(0, 50)
    title(ax, "Standard GP builds structure across a run; TSGP does not (ESL)")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "fig_size_generations")


def size_vs_accuracy():
    """The accuracy/compactness trade at the centre of the extension."""
    std, tsgp = load_runs("results_clf"), load_runs("results_clf_k1")
    ml = {}
    p = "results_clf_ml/ml_baselines.json"
    if os.path.exists(p):
        for r in json.load(open(p)):
            ml.setdefault(r["dataset"], defaultdict(list))[r["model"]].append(r)

    fig, axes = plt.subplots(1, 5, figsize=(W_FULL * 1.55, 2.5))
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
        linlog(ax, "x", [5, 10, 25, 50, 100, 200])
        ax.set_xlim(4, 260)
        if np.isfinite(maj):
            top = max([p[2] for p in pts] + [maj]) + 0.035
            ax.set_ylim(maj - 0.025, min(top, 1.01))
        ax.set_title(SHORT[ds], fontsize=BASE_SIZE + 0.5)
        ax.set_xlabel("solution size (nodes)")
    axes[0].set_ylabel("test accuracy")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.16))
    fig.text(0.5, -0.05, "dashed line: majority-class baseline",
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
    "step_floor": step_floor,
    "locality_control": locality_control,
    "size_generations": size_generations,
    "size_vs_accuracy": size_vs_accuracy,
    "task_difficulty": task_difficulty,
    "decision_value": decision_value,
    "semantic_distance": semantic_distance,
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
