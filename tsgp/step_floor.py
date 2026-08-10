"""How small a semantic step can this operator actually take?

Both SD encodings bottom out at the same achieved parent-offspring distance no
matter how small a distance is requested: asking for SD_d = 0.01 and SD_d = 0.1
produce the same result. That is the signature of a *floor* -- a smallest step
the operator can reliably generate -- rather than of weak conditioning, and it
matters because the paper's operating point (SD_d = 0.1) sits below it. A search
whose smallest possible move is larger than the precision it needs cannot have
an exploitation phase, which is exactly what the ESL curves show: stdGP anneals
its step from 28 down to 1.0 while TSGP stays at 16-34 and freezes.

Two things are measured here:

  FLOOR   achieved distance as SD_d is driven far below the training range.
          If it stops falling, that value is the floor.

  MIN-OF-K  the distribution of min-over-k sampled offspring distances. Drawing
          k offspring and keeping the semantically nearest is the cheapest way
          to get underneath the floor, and it is what the paper lists as future
          work in Sect. 5 ("systematically control the step size of the
          transformer during the search"). This quantifies what each doubling
          of k buys before spending a grid on it.
"""
import argparse
import json
import os

import numpy as np

from . import config
from .operator_diagnostics import SemanticSpace, Sampler


def measure(model, pool_parents, k_max=16, seed=5, verbose=True):
    rng = np.random.default_rng(seed)
    space = SemanticSpace()
    sampler = Sampler(model, space.pset, rng)

    sems, parents = [], []
    for p in pool_parents:
        s = space.of(p)
        if s is not None:
            sems.append(s)
            parents.append(p)
    sems = np.array(sems)
    report = {"n_parents": len(parents)}

    if verbose:
        print(f"parents: {len(parents)}   "
              f"(training pairs sit at SD median "
              f"{0.164}, p25 0.049)")
        print(f"\n--- FLOOR: achieved distance vs requested ---")
        print(f"{'SD_d':>10} {'median':>9} {'p25':>9} {'p10':>9}")

    floor = {}
    for sd_v in [1e-4, 1e-3, 1e-2, 0.1, 1.0]:
        offs = sampler(parents, sd_v)
        d = []
        for i, o in enumerate(offs):
            so = space.of(o)
            if so is not None:
                d.append(np.linalg.norm(so - sems[i]))
        if not d:
            continue
        d = np.array(d)
        floor[sd_v] = {"median": float(np.median(d)),
                       "p25": float(np.percentile(d, 25)),
                       "p10": float(np.percentile(d, 10))}
        if verbose:
            print(f"{sd_v:>10.4f} {np.median(d):>9.3f} "
                  f"{np.percentile(d, 25):>9.3f} {np.percentile(d, 10):>9.3f}")
    report["floor"] = {str(k): v for k, v in floor.items()}

    # ---- min-of-k at the paper's operating point -------------------------
    if verbose:
        print(f"\n--- MIN-OF-K at SD_d = {config.TSGP_SD_DESIRED} ---")
    draws = []
    for _ in range(k_max):
        offs = sampler(parents, config.TSGP_SD_DESIRED)
        row = []
        for i, o in enumerate(offs):
            so = space.of(o)
            row.append(np.linalg.norm(so - sems[i]) if so is not None
                       else np.inf)
        draws.append(row)
    D = np.array(draws)                      # [k_max, n_parents]

    if verbose:
        print(f"{'k':>4} {'median min-of-k':>16} {'vs k=1':>8}")
    mink = {}
    base = None
    for k in [1, 2, 4, 8, 16]:
        if k > k_max:
            break
        m = np.min(D[:k], axis=0)
        m = m[np.isfinite(m)]
        med = float(np.median(m))
        base = med if k == 1 else base
        mink[k] = med
        if verbose:
            print(f"{k:>4} {med:>16.3f} {base / med if med > 0 else float('nan'):>7.2f}x")
    report["min_of_k"] = {str(k): v for k, v in mink.items()}

    if verbose:
        print("\nInterpretation: if the FLOOR table stops falling as SD_d "
              "drops, the\noperator cannot honour requests below that value "
              "and step size has to\nbe controlled by selection instead. "
              "MIN-OF-K says how much that buys.")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--parents-json", default="diagnostics/pool_parents.json")
    parser.add_argument("--k-max", type=int, default=16)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    from .experiment_units import load_model

    model = load_model(args.weights)
    print(f"checkpoint: {args.weights}  (encoding {model.sd_encoding})")
    with open(args.parents_json) as f:
        parents = json.load(f)

    rep = measure(model, parents, k_max=args.k_max)
    out = args.out or (os.path.splitext(args.weights)[0] + "_stepfloor.json")
    with open(out, "w") as f:
        json.dump(rep, f, indent=2)
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
