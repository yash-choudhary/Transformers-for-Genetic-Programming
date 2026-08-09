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


def _summarise(units):
    test_rmses = [u["test_rmse"] for u in units]
    sizes = [u["best_size"] for u in units]
    return {
        "n_runs": len(units),
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
        print(f"{'Dataset':<24} {'TSGP':<18} {'stdGP':<18}")
        print("-" * 62)
        for dataset in datasets:
            r = all_results.get(dataset, {})
            cells = []
            for method in ("tsgp", "stdgp"):
                m = r.get(method)
                cells.append(
                    f"{m['test_rmse_median']:.4f} (n={m['n_runs']})"
                    if m else "-"
                )
            label = DISPLAY_NAMES.get(dataset, dataset)
            print(f"{label:<24} {cells[0]:<18} {cells[1]:<18}")

        if missing:
            print(f"\nMissing {len(missing)} unit(s); first 10:")
            for dataset, method, run in missing[:10]:
                print(f"  {dataset} / {method} / run {run}")
            print("\nRun `python -m tsgp.run_experiments --weights <path>` "
                  "to fill these in (completed units are skipped).")
        print(f"\nMedian test RMSE written to {results_path}")

    return all_results, missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=RESULTS_DIR)
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--methods", nargs="+", default=METHODS,
                        choices=METHODS)
    parser.add_argument("--runs", type=int, default=config.NUM_RUNS)
    args = parser.parse_args()
    aggregate(args.datasets, args.methods, args.runs, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
