"""Build a training pool for the classification operator.

Two things separate this from the regression pool in data_generator.py, and
both come from measurement rather than preference.

**The semantic regime.** In regression, matching a standardised target pins the
scale of f: outputs have to be about the size of y. Under logistic loss nothing
does -- 2*f always scores better than f -- so a classification search drifts to
ever larger magnitudes. Measured on irish, the population's median semantic norm
climbs 19.5 -> 27.3 -> 57.7 over generations, with the best solution at 74,
against a regression pool that sits at ~10. On the regression side the operator
was near-random on parents with norms in that range, so a pool built from
regression problems leaves the classification search out of distribution for
essentially its whole run. The fix is to generate the pool from synthetic
*classification* problems solved under the same logistic loss, so the pool spans
the magnitudes the real search visits.

**The dimensionality.** PMLB has two 4-feature binary classification sets and
neither is usable, so the operator has to be wider than the paper's d=4 for the
benchmarks to mean anything. Run this with TSGP_NUM_FEATURES=8.

The synthetic problems deliberately mix linear and mildly non-linear decision
boundaries. A pool built only from linear boundaries would teach the operator
that decision functions are weighted sums, which is not true of the benchmarks
and would bias every offspring it proposes.
"""
import csv
import os
import pickle
import random

import numpy as np

from . import config
from .classification import logistic_loss
from .data_generator import (compute_all_semantics, coverage_by_norm,
                             find_semantic_pairs, report_pool_coverage)
from .primitives import create_pset, setup_datagen_toolbox
from .tokenizer import tree_to_tokens

BOUNDARY_KINDS = ("linear", "quadratic", "interaction")


def generate_synthetic_classification_problem(num_samples=200,
                                              num_features=None,
                                              kind=None, rng=None):
    """A random binary problem with a known decision boundary.

    Returns X standardised and y in {-1, +1}. The intercept is set from the
    realised margin so classes stay roughly balanced regardless of the boundary
    drawn -- an all-one-class problem would teach the operator nothing.
    """
    rng = rng or np.random
    n = num_features or config.NUM_FEATURES
    kind = kind or rng.choice(BOUNDARY_KINDS)

    X = rng.standard_normal((num_samples, n))
    w = rng.standard_normal(n)
    margin = X @ w

    if kind == "quadratic":
        v = rng.standard_normal(n)
        margin = margin + (X ** 2) @ v * 0.5
    elif kind == "interaction":
        i, j = rng.choice(n, size=2, replace=False)
        margin = margin + X[:, i] * X[:, j] * rng.standard_normal()

    # centre so the split is near 50/50, then add label noise
    margin = margin - np.median(margin)
    margin = margin + rng.standard_normal(num_samples) * rng.uniform(0.1, 0.6)
    y = np.where(margin >= 0, 1.0, -1.0)

    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    return X, y


def run_stdgp_for_functions(X, y, pop_size=None, generations=None,
                            verbose=False):
    """stdGP under logistic loss, collecting every unique function it visits.

    Sect. 3.1's phase one, with the objective swapped. Every generation's
    population is recorded, not just the last, so the pool spans the whole
    trajectory from random initial trees to fitted decision functions -- which
    is exactly the range the real search moves through.
    """
    pop_size = pop_size or config.DATAGEN_GP_POP_SIZE
    generations = generations or config.DATAGEN_GP_GENERATIONS

    pset = create_pset(num_features=X.shape[1])
    toolbox = setup_datagen_toolbox(pset)
    toolbox.register("evaluate", logistic_loss, X=X, y=y)

    pop = toolbox.population(n=pop_size)
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)

    unique = {}
    for ind in pop:
        unique.setdefault(str(ind), toolbox.clone(ind))

    for gen in range(1, generations + 1):
        offspring = [toolbox.clone(i) for i in toolbox.select(pop, len(pop))]
        for i in range(0, len(offspring) - 1, 2):
            if random.random() < config.GP_CROSSOVER_PROB:
                offspring[i], offspring[i + 1] = toolbox.mate(
                    offspring[i], offspring[i + 1])
                del offspring[i].fitness.values
                del offspring[i + 1].fitness.values
        for i in range(len(offspring)):
            if random.random() < config.GP_MUTATION_PROB:
                offspring[i], = toolbox.mutate(offspring[i])
                del offspring[i].fitness.values
        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = toolbox.evaluate(ind)
        pop[:] = offspring
        for ind in pop:
            unique.setdefault(str(ind), toolbox.clone(ind))

        if verbose and gen % 10 == 0:
            best = min(pop, key=lambda i: i.fitness.values[0])
            print(f"    gen {gen}: loss={best.fitness.values[0]:.4f} "
                  f"unique={len(unique):,}", flush=True)

    return list(unique.values()), pset


def generate(output_dir, num_problems=None, sd_max=None, verbose=True):
    num_problems = num_problems or config.NUM_SYNTHETIC_PROBLEMS
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(0)

    functions, token_seqs, pset = [], [], None
    for idx in range(num_problems):
        kind = BOUNDARY_KINDS[idx % len(BOUNDARY_KINDS)]
        X, y = generate_synthetic_classification_problem(
            num_samples=200, kind=kind, rng=rng)
        if verbose:
            print(f"problem {idx + 1}/{num_problems} ({kind}) ...", flush=True)
        found, pset = run_stdgp_for_functions(X, y, verbose=verbose)
        for f in found:
            toks = tree_to_tokens(f)
            if len(toks) <= config.TRANSFORMER_MAX_SEQ_LEN - 2:
                functions.append(f)
                token_seqs.append(toks)
        if verbose:
            print(f"  kept {len(found):,}; pool now {len(functions):,}",
                  flush=True)

    if verbose:
        print(f"\npool: {len(functions):,} functions. Computing semantics ...",
              flush=True)
    semantics, valid = compute_all_semantics(functions, pset)
    if verbose:
        print(f"finite semantics: {len(valid):,}")
        report_pool_coverage(semantics, valid,
                             os.path.join(output_dir, "pool_coverage.txt"))
        print()
        cov = coverage_by_norm(semantics)
        with open(os.path.join(output_dir, "pool_coverage.txt"), "a") as f:
            f.write("\n\n" + cov + "\n")

    pairs = find_semantic_pairs(functions, semantics, valid, sd_max=sd_max)

    data = [{"input_tokens": token_seqs[i],
             "output_tokens": token_seqs[j],
             "sd": float(sd)} for i, j, sd in pairs]

    pkl = os.path.join(output_dir, "training_pairs.pkl")
    with open(pkl, "wb") as f:
        pickle.dump(data, f, protocol=4)
    with open(os.path.join(output_dir, "training_pairs.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["input_tokens", "output_tokens", "sd"])
        for d in data:
            w.writerow([" ".join(d["input_tokens"]),
                        " ".join(d["output_tokens"]), f"{d['sd']:.6f}"])
    if verbose:
        print(f"\nwrote {len(data):,} pairs to {pkl}")
    return data


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Generate the classification operator's training pairs. "
                    "Run with TSGP_NUM_FEATURES=8.")
    p.add_argument("--output", default="data/training_clf")
    p.add_argument("--problems", type=int,
                   default=config.NUM_SYNTHETIC_PROBLEMS)
    p.add_argument("--sd-max", type=float, default=None,
                   help="Pair filter. The paper uses 100 for regression; the "
                        "classification pool's norms are larger, so check "
                        "pool_coverage.txt and raise this if most functions "
                        "end up with no neighbour inside it.")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args()

    print(f"NUM_FEATURES = {config.NUM_FEATURES} "
          f"(set TSGP_NUM_FEATURES to change)")
    generate(a.output, num_problems=a.problems, sd_max=a.sd_max,
             verbose=not a.quiet)
