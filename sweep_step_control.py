"""Screen step-control configurations on one data set before committing a grid.

The annealed operator (k=8, target decaying 2.0 -> 0.1 x ||y_train||) beat the
paper's k=1 operator significantly on all five data sets, but those two numbers
were a first guess and were never tuned. This screens a small set of variants
on a single data set, paired against the same seeds, so the full grid is only
spent on a configuration that has already shown itself.

Two of the arms are controls rather than candidates:

  const-*   frac_start == frac_end, i.e. a constant target. Separates "annealing
            helps" from "merely targeting a distance helps". Without this the
            claim that the *schedule* matters is unsupported.
  k=1       the paper's operator, for reference.

Everything is written to results_sweep/<name>/ and is resumable per unit.
"""
import argparse
import itertools
import json
import os
import subprocess
import sys
from collections import defaultdict

import numpy as np

ARMS = [
    # name,            k,  frac_start, frac_end
    ("k1",             1,  None,       None),
    ("a_2p0_0p10",     8,  2.0,        0.10),   # the grid's configuration
    ("b_1p0_0p02",     8,  1.0,        0.02),   # start smaller, end smaller
    ("c_4p0_0p05",     8,  4.0,        0.05),   # wider sweep
    ("d_2p0_0p10_k16", 16, 2.0,        0.10),   # does more candidates help?
    ("e_const_0p30",   8,  0.30,       0.30),   # CONTROL: constant target
]


def run_arm(name, k, fs, fe, dataset, runs, weights, python):
    out = os.path.join("results_sweep", name)
    cmd = [python, "-m", "tsgp.run_experiments", "--weights", weights,
           "--methods", "tsgp", "--datasets", dataset, "--runs", str(runs),
           "--step-k", str(k), "--output", out, "--quiet"]
    if k > 1:
        cmd += ["--step-anneal",
                "--step-frac-start", str(fs), "--step-frac-end", str(fe)]
    print(f"\n=== {name}: k={k} frac {fs}->{fe} ===", flush=True)
    subprocess.run(cmd, check=False)
    return out


def collect(out_dir):
    vals = {}
    for f in os.listdir(os.path.join(out_dir, "runs")):
        if not f.endswith(".json"):
            continue
        r = json.load(open(os.path.join(out_dir, "runs", f)))
        vals[r["run"]] = (r["test_rmse"], r["best_size"])
    return vals


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="1027_ESL")
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--weights", default="checkpoints_adamw/tsgp_final.npy")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--only", default=None,
                   help="Comma-separated arm names, to resume a partial sweep")
    args = p.parse_args()

    arms = ARMS
    if args.only:
        keep = set(args.only.split(","))
        arms = [a for a in ARMS if a[0] in keep]

    results = {}
    for name, k, fs, fe in arms:
        out = run_arm(name, k, fs, fe, args.dataset, args.runs,
                      args.weights, args.python)
        try:
            results[name] = collect(out)
        except FileNotFoundError:
            print(f"  {name}: no results written")

    base = results.get("k1", {})
    print(f"\n{'arm':<16} {'n':>3} {'median':>8} {'vs k=1':>8} "
          f"{'better':>8} {'p':>8} {'size':>6}")
    print("-" * 62)
    from scipy.stats import wilcoxon
    for name, k, fs, fe in arms:
        v = results.get(name)
        if not v:
            continue
        runs = sorted(v)
        rm = [v[r][0] for r in runs]
        sz = [v[r][1] for r in runs]
        common = sorted(set(v) & set(base)) if base else []
        if common and name != "k1":
            a = [v[r][0] for r in common]
            b = [base[r][0] for r in common]
            better = f"{sum(1 for x, y in zip(a, b) if x < y)}/{len(common)}"
            delta = f"{np.median(a) - np.median(b):+.4f}"
            try:
                pv = f"{wilcoxon(a, b).pvalue:.4f}"
            except ValueError:
                pv = "-"
        else:
            better, delta, pv = "-", "-", "-"
        print(f"{name:<16} {len(rm):>3} {np.median(rm):>8.4f} {delta:>8} "
              f"{better:>8} {pv:>8} {np.median(sz):>6.0f}")

    print("\nIf const-0p30 matches the annealed arms, the schedule is not what")
    print("matters and a fixed target is the simpler claim to make.")


if __name__ == "__main__":
    main()
