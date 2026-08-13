"""Prototype: TSGP against standard GP on binary classification.

Trees stay real-valued and are read as decision values, thresholded at zero, so
the semantics the transformer was trained on remain meaningful and the operator
can be reused without retraining. See tsgp/classification.py for the reasoning.

    python run_classification.py --datasets irish --runs 5
"""
import argparse
import json
import os
import time

import numpy as np

from tsgp import config
from tsgp.classification import (DISPLAY_NAMES, N_SOURCE_FEATURES,
                                 default_datasets,
                                 load_classification_dataset,
                                 majority_baseline, run_stdgp_classify,
                                 run_tsgp_classify)
from tsgp.experiment_units import unit_seed, set_seeds, load_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--generations", type=int, default=None)
    p.add_argument("--weights", default="checkpoints_adamw/tsgp_final.npy")
    p.add_argument("--methods", default="tsgp,stdgp")
    p.add_argument("--step-k", type=int, default=8)
    p.add_argument("--step-anneal", action="store_true", default=True)
    p.add_argument("--step-frac-start", type=float, default=1.0)
    p.add_argument("--step-frac-end", type=float, default=0.02)
    p.add_argument("--out", default="results_classification")
    args = p.parse_args()
    if args.datasets is None:
        args.datasets = default_datasets()
    print(f"NUM_FEATURES = {config.NUM_FEATURES}   datasets: "
          f"{', '.join(args.datasets)}", flush=True)

    methods = args.methods.split(",")
    os.makedirs(os.path.join(args.out, "runs"), exist_ok=True)

    model = None
    if "tsgp" in methods:
        print(f"loading {args.weights} ...", flush=True)
        model = load_model(args.weights)

    for dataset in args.datasets:
        for method in methods:
            for run in range(args.runs):
                path = os.path.join(args.out, "runs",
                                    f"{dataset}__{method}__run{run:02d}.json")
                if os.path.exists(path):
                    continue
                seed = unit_seed(dataset, method, run)
                set_seeds(seed)
                Xtr, Xte, ytr, yte = load_classification_dataset(
                    dataset, split_seed=seed)
                t0 = time.perf_counter()
                if method == "tsgp":
                    _, out = run_tsgp_classify(
                        model, Xtr, ytr, Xte, yte,
                        generations=args.generations, step_k=args.step_k,
                        step_anneal=args.step_anneal,
                        frac_start=args.step_frac_start,
                        frac_end=args.step_frac_end)
                else:
                    _, out = run_stdgp_classify(
                        Xtr, ytr, Xte, yte, generations=args.generations)
                out.update({
                    "dataset": dataset, "method": method, "run": run,
                    "seed": seed,
                    "elapsed_sec": round(time.perf_counter() - t0, 1),
                    "majority_baseline": majority_baseline(ytr, yte),
                    "n_source_features": N_SOURCE_FEATURES.get(dataset),
                    "step_k": args.step_k if method == "tsgp" else None,
                })
                tmp = path + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(out, f, indent=2)
                os.replace(tmp, path)
                print(f"  {DISPLAY_NAMES.get(dataset, dataset):<10} {method:<6} "
                      f"run {run}: acc={out['test_acc']:.4f} "
                      f"auc={out['test_auc']:.4f} size={out['size']} "
                      f"(maj {out['majority_baseline']:.3f}) "
                      f"{out['elapsed_sec']:.0f}s", flush=True)

    summarise(args.out, args.datasets, methods)


def summarise(out_dir, datasets, methods):
    from collections import defaultdict
    from scipy.stats import ranksums

    data = defaultdict(lambda: defaultdict(list))
    maj = {}
    for f in os.listdir(os.path.join(out_dir, "runs")):
        if not f.endswith(".json"):
            continue
        r = json.load(open(os.path.join(out_dir, "runs", f)))
        data[r["dataset"]][r["method"]].append(r)
        maj[r["dataset"]] = r["majority_baseline"]

    print(f"\n{'Dataset':<10} {'method':<6} {'n':>3} {'test acc':>9} "
          f"{'test AUC':>9} {'size':>5} {'majority':>9} {'over maj':>9}")
    print("-" * 72)
    for d in datasets:
        for m in methods:
            rows = data[d].get(m, [])
            if not rows:
                continue
            acc = np.median([r["test_acc"] for r in rows])
            auc = np.median([r["test_auc"] for r in rows])
            sz = np.median([r["size"] for r in rows])
            print(f"{DISPLAY_NAMES.get(d, d):<10} {m:<6} {len(rows):>3} "
                  f"{acc:>9.4f} {auc:>9.4f} {sz:>5.0f} {maj[d]:>9.3f} "
                  f"{acc - maj[d]:>+9.4f}")
        t, s = data[d].get("tsgp"), data[d].get("stdgp")
        if t and s:
            pa = ranksums([r["test_acc"] for r in t],
                          [r["test_acc"] for r in s]).pvalue
            pu = ranksums([r["test_auc"] for r in t],
                          [r["test_auc"] for r in s]).pvalue
            print(f"{'':<10} {'tsgp vs stdgp':<22} acc p={pa:.4f}  auc p={pu:.4f}")
    print("\n'over maj' is the margin above always predicting the majority "
          "class.\nA method at or below zero has found nothing usable.")


if __name__ == "__main__":
    main()
