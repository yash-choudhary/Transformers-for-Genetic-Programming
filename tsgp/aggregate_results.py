"""Step 3d — collect per-unit results into experiment_results.json.

Reads whatever units are present under results/runs/ and writes the same
aggregate schema the single-script version produced. Safe to run on a partial
grid: it reports what's missing and aggregates the rest.

    python -m tsgp.aggregate_results
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

from . import config
from .datasets import DATASETS, DISPLAY_NAMES
from .experiment_units import METHODS, RESULTS_DIR, load_completed


# Paper Table 2 (median test RMSE) and Table 3 (median best-solution size),
# for the two methods this grid runs.
PAPER_TABLE = {
    "1030_ERA":               {"tsgp": (0.797, 72), "stdgp": (0.817, 60)},
    "1027_ESL":               {"tsgp": (0.379, 73), "stdgp": (0.502, 12)},
    "690_visualizing_galaxy": {"tsgp": (0.327, 64), "stdgp": (0.337, 48)},
    "1029_LEV":               {"tsgp": (0.672, 69), "stdgp": (0.703, 50)},
    "529_pollen":             {"tsgp": (0.518, 58), "stdgp": (0.514, 37)},
}
# Which method the paper reports as better on each data set (Table 2).
PAPER_WINNER = {"1030_ERA": "tsgp", "1027_ESL": "tsgp",
                "690_visualizing_galaxy": "tsgp", "1029_LEV": "tsgp",
                "529_pollen": "stdgp"}


def _summarise(units):
    test_rmses = [u["test_rmse"] for u in units]
    sizes = [u["best_size"] for u in units]
    ks = sorted({u.get("step_k") for u in units if u.get("step_k") is not None})
    return {
        "n_runs": len(units),
        # k = 1 is the paper's operator; k > 1 is Sect. 5 step-size control.
        # Surfaced here so a summary table can never silently mix the two.
        "step_k": ks[0] if len(ks) == 1 else (ks or None),
        "test_rmse_median": float(np.median(test_rmses)),
        "test_rmse_mean": float(np.mean(test_rmses)),
        "test_rmse_std": float(np.std(test_rmses)),
        "test_rmses": [float(x) for x in test_rmses],
        "size_median": float(np.median(sizes)),
        "sizes": [int(x) for x in sizes],
    }


def aggregate(datasets=None, methods=None, num_runs=None,
              output_dir=RESULTS_DIR, verbose=True):
    if datasets is None:
        datasets = DATASETS
    if methods is None:
        methods = METHODS
    if num_runs is None:
        num_runs = config.NUM_RUNS

    units, missing = load_completed(datasets, methods, num_runs, output_dir)

    by_key = defaultdict(list)
    for u in units:
        by_key[(u["dataset"], u["method"])].append(u)

    all_results = {}
    for dataset in datasets:
        entry = {}
        for method in methods:
            found = sorted(by_key.get((dataset, method), []),
                           key=lambda u: u["run"])
            if found:
                entry[method] = _summarise(found)
        if entry:
            all_results[dataset] = entry

    results_path = os.path.join(output_dir, "experiment_results.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    if verbose:
        total = len(datasets) * len(methods) * num_runs
        print(f"Aggregated {len(units)}/{total} units "
              f"({len(missing)} missing)\n")
        print("MEDIAN TEST RMSE — ours vs the paper's Table 2")
        print(f"{'Dataset':<10} {'TSGP':>8} {'stdGP':>8} | "
              f"{'pTSGP':>7} {'pStdGP':>7} | {'dTSGP':>7} {'dStdGP':>7} | "
              f"{'winner':>7} {'paper':>7} {'p-value':>10}")
        print("-" * 92)
        agree = 0
        comparable = 0
        for dataset in datasets:
            r = all_results.get(dataset, {})
            t, s = r.get("tsgp"), r.get("stdgp")
            label = DISPLAY_NAMES.get(dataset, dataset)
            ref = PAPER_TABLE.get(dataset, {})
            if not t or not s:
                # A one-armed grid is still worth reporting against the paper;
                # only the head-to-head columns are unavailable.
                for name, m in (("tsgp", t), ("stdgp", s)):
                    if not m:
                        continue
                    pv = ref.get(name, (float("nan"),))[0]
                    print(f"{label:<10} {name:>8} n={m['n_runs']:<3} "
                          f"median {m['test_rmse_median']:.4f}  paper {pv:.3f}  "
                          f"delta {m['test_rmse_median'] - pv:+.3f}  "
                          f"size {m['size_median']:.0f}"
                          + (f"  k={m['step_k']}" if m.get("step_k") else ""))
                continue
            pt = ref.get("tsgp", (float('nan'),))[0]
            ps = ref.get("stdgp", (float('nan'),))[0]
            mt, ms = t["test_rmse_median"], s["test_rmse_median"]

            # Wilcoxon rank-sum, as the paper uses (Sect. 4.2, alpha = 0.05).
            try:
                from scipy.stats import ranksums
                pval = float(ranksums(t["test_rmses"], s["test_rmses"]).pvalue)
            except Exception:
                pval = float("nan")
            winner = "tsgp" if mt < ms else "stdgp"
            if pval > 0.05:
                winner_label = "n.s."
            else:
                winner_label = winner
            paper_win = PAPER_WINNER.get(dataset, "?")
            comparable += 1
            if winner == paper_win:
                agree += 1
            print(f"{label:<10} {mt:>8.4f} {ms:>8.4f} | {pt:>7.3f} {ps:>7.3f} | "
                  f"{mt-pt:>+7.3f} {ms-ps:>+7.3f} | "
                  f"{winner_label:>7} {paper_win:>7} {pval:>10.2e}")
        if comparable:
            print(f"\nDirection agrees with the paper on {agree}/{comparable} "
                  f"data sets.")
        ks = {r["tsgp"]["step_k"] for r in all_results.values()
              if r.get("tsgp") and r["tsgp"].get("step_k")}
        if ks:
            k_str = ", ".join(str(k) for k in sorted(ks, key=str))
            note = ("the paper's operator" if ks == {1} else
                    "step-size control from Sect. 5's future work, NOT the "
                    "paper's operator")
            print(f"TSGP step_k = {k_str} — {note}.")
            if ks != {1}:
                print("  k>1 spends k times the model evaluations per "
                      "generation, so this is not an equal-budget comparison "
                      "against stdGP.")

        print("\nMEDIAN BEST-SOLUTION SIZE — ours vs the paper's Table 3")
        print(f"{'Dataset':<10} {'TSGP':>8} {'stdGP':>8} | "
              f"{'pTSGP':>7} {'pStdGP':>7}")
        print("-" * 46)
        for dataset in datasets:
            r = all_results.get(dataset, {})
            t, s = r.get("tsgp"), r.get("stdgp")
            label = DISPLAY_NAMES.get(dataset, dataset)
            if not t or not s:
                continue
            ref = PAPER_TABLE.get(dataset, {})
            print(f"{label:<10} {t['size_median']:>8.0f} {s['size_median']:>8.0f} | "
                  f"{ref.get('tsgp', (0, 0))[1]:>7} {ref.get('stdgp', (0, 0))[1]:>7}")

        if missing:
            print(f"\nMissing {len(missing)} unit(s); first 10:")
            for dataset, method, run in missing[:10]:
                print(f"  {dataset} / {method} / run {run}")
            print("\nRun `python -m tsgp.run_experiments --weights <path>` "
                  "to fill these in (completed units are skipped).")
        print(f"\nMedian test RMSE written to {results_path}")

    return all_results, missing


def compare(dir_a, dir_b, datasets=None, methods=None, num_runs=None,
            label_a="A", label_b="B"):
    """Head-to-head between two result directories.

    Used to isolate the effect of a single change -- e.g. `results` was
    produced with the train/test split pinned at random_state=42 for all 30
    runs, `results_v7` with a per-run split as Sect. 4.1 specifies. Both arms
    are otherwise the same code and the same checkpoint, so the difference is
    attributable.
    """
    datasets = datasets or DATASETS
    methods = methods or METHODS
    num_runs = num_runs or config.NUM_RUNS

    a, _ = aggregate(datasets, methods, num_runs, dir_a, verbose=False)
    b, _ = aggregate(datasets, methods, num_runs, dir_b, verbose=False)

    print(f"\n{'':<10} {'':<6} {label_a:>10} {label_b:>10} {'change':>9} "
          f"{'paper':>8} {'|delta| a':>10} {'|delta| b':>10}")
    print("-" * 78)
    closer = {"a": 0, "b": 0}
    for dataset in datasets:
        for method in methods:
            ma = a.get(dataset, {}).get(method)
            mb = b.get(dataset, {}).get(method)
            if not ma or not mb:
                continue
            va, vb = ma["test_rmse_median"], mb["test_rmse_median"]
            pv = PAPER_TABLE.get(dataset, {}).get(method, (float("nan"),))[0]
            da, db = abs(va - pv), abs(vb - pv)
            if db < da:
                closer["b"] += 1
            elif da < db:
                closer["a"] += 1
            print(f"{DISPLAY_NAMES.get(dataset, dataset):<10} {method:<6} "
                  f"{va:>10.4f} {vb:>10.4f} {vb - va:>+9.4f} {pv:>8.3f} "
                  f"{da:>10.3f} {db:>10.3f}")
    print(f"\nCloser to the published value: {label_b} on {closer['b']} "
          f"cells, {label_a} on {closer['a']}.")

    print(f"\nWinner per data set (TSGP vs stdGP)")
    print(f"{'':<10} {label_a:>10} {label_b:>10} {'paper':>10}")
    print("-" * 44)
    for dataset in datasets:
        row = []
        for res in (a, b):
            t = res.get(dataset, {}).get("tsgp")
            s = res.get(dataset, {}).get("stdgp")
            row.append("-" if not t or not s else
                       ("tsgp" if t["test_rmse_median"] < s["test_rmse_median"]
                        else "stdgp"))
        print(f"{DISPLAY_NAMES.get(dataset, dataset):<10} {row[0]:>10} "
              f"{row[1]:>10} {PAPER_WINNER.get(dataset, '?'):>10}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=RESULTS_DIR)
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--methods", nargs="+", default=METHODS,
                        choices=METHODS)
    parser.add_argument("--runs", type=int, default=config.NUM_RUNS)
    parser.add_argument("--compare-with", default=None,
                        help="A second results directory to compare against, "
                             "to isolate one change (e.g. the per-run "
                             "train/test split) with everything else held "
                             "fixed.")
    parser.add_argument("--label-a", default="before")
    parser.add_argument("--label-b", default="after")
    args = parser.parse_args()

    if args.compare_with:
        compare(args.compare_with, args.output, args.datasets, args.methods,
                args.runs, args.label_a, args.label_b)
        return 0
    aggregate(args.datasets, args.methods, args.runs, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
