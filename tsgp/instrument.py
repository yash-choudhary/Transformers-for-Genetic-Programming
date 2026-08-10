"""Per-generation instrumentation of the TSGP and stdGP searches.

The replication study so far compared final numbers against the paper's
Tables 2 and 3. But the paper also publishes three *curves* -- Fig. 2 (training
RMSE over generations), Fig. 3 (size of the best solution over generations) and
Fig. 4 (parent-offspring semantic distance over generations) -- and those say
far more about where a search goes wrong than any endpoint does.

This module reruns both searches while recording, every generation:

  * best-so-far training RMSE               -> Fig. 2
  * size of the best-so-far solution        -> Fig. 3
  * median parent->offspring semantic distance, measured exactly as Sect. 4.4
    specifies (Euclidean distance between semantics approximated on the *test
    set of the target problem*, counting only variations where the offspring
    is structurally different from its parent)  -> Fig. 4

plus two diagnostics the paper does not publish but which distinguish a
working search from a stalled one:

  * improvement rate: fraction of offspring that beat their own parent
  * population semantic diversity: mean pairwise distance within the population

The searches themselves are unchanged; this only observes them.
"""
import json
import os
import random

import numpy as np
from deap import creator

from . import config
from .primitives import (create_pset, setup_deap, evaluate_individual,
                         evaluate_semantics)
from .tsgp_search import TSGPSearchOperator, _update_best_ever


def _semantics_matrix(individuals, toolbox, X):
    """Semantics of each individual on X. Rows that blow up become None."""
    out = []
    for ind in individuals:
        s = evaluate_semantics(ind, toolbox, X)
        out.append(s if np.all(np.isfinite(s)) else None)
    return out


def _pair_distances(parents, children, toolbox, X):
    """Sect. 4.4: Euclidean distance between parent and offspring semantics.

    Only "successful variations where the offspring is structurally different
    from its parent" are counted.
    """
    ps = _semantics_matrix(parents, toolbox, X)
    cs = _semantics_matrix(children, toolbox, X)
    dists = []
    for parent, child, sp, sc in zip(parents, children, ps, cs):
        if sp is None or sc is None or child is None:
            continue
        if str(parent) == str(child):
            continue
        d = float(np.linalg.norm(sp - sc))
        if np.isfinite(d):
            dists.append(d)
    return dists


def _population_diversity(pop, toolbox, X, rng, sample=40):
    sel = pop if len(pop) <= sample else [pop[i] for i in
                                          rng.choice(len(pop), sample,
                                                     replace=False)]
    sems = [s for s in _semantics_matrix(sel, toolbox, X) if s is not None]
    if len(sems) < 2:
        return float("nan")
    M = np.array(sems)
    # mean pairwise Euclidean distance, upper triangle only
    d = np.linalg.norm(M[:, None, :] - M[None, :, :], axis=-1)
    iu = np.triu_indices(len(M), k=1)
    return float(np.mean(d[iu]))


def _blank_log():
    return {
        "train_rmse": [],          # best-so-far  (Fig. 2)
        "best_size": [],           # best-so-far  (Fig. 3)
        "pop_median_size": [],
        "pair_distance_median": [],   # Fig. 4
        "pair_distance_p25": [],
        "pair_distance_p75": [],
        "improvement_rate": [],
        "pop_diversity": [],
        "n_valid_offspring": [],
    }


def _record(log, best_ever, pop, dists, improved, n_valid, diversity):
    log["train_rmse"].append(float(best_ever.fitness.values[0]))
    log["best_size"].append(int(len(best_ever)))
    log["pop_median_size"].append(float(np.median([len(i) for i in pop])))
    if dists:
        log["pair_distance_median"].append(float(np.median(dists)))
        log["pair_distance_p25"].append(float(np.percentile(dists, 25)))
        log["pair_distance_p75"].append(float(np.percentile(dists, 75)))
    else:
        for k in ("pair_distance_median", "pair_distance_p25",
                  "pair_distance_p75"):
            log[k].append(float("nan"))
    log["improvement_rate"].append(float(improved))
    log["pop_diversity"].append(float(diversity))
    log["n_valid_offspring"].append(int(n_valid))


def _stdgp_generation(pop, toolbox, pop_size):
    """One plain stdGP generation, used for the warm-up diagnostic."""
    selected = toolbox.select(pop, pop_size)
    parents_snapshot = [toolbox.clone(i) for i in selected]
    offspring = [toolbox.clone(i) for i in selected]
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
    varied = [i for i, ind in enumerate(offspring) if not ind.fitness.valid]
    for ind in offspring:
        if not ind.fitness.valid:
            ind.fitness.values = toolbox.evaluate(ind)
    improved = sum(1 for i in varied
                   if offspring[i].fitness.values[0]
                   < parents_snapshot[i].fitness.values[0])
    return offspring, [parents_snapshot[i] for i in varied], \
        [offspring[i] for i in varied], improved


def _closest_of_k(search_op, selected, k, toolbox, X_train, pset):
    """Sample k offspring per parent, keep the semantically nearest one.

    DIAGNOSTIC ONLY -- this is not the paper's operator. It exists to test one
    specific claim: that TSGP fails here because its variation step is a fixed
    jump of roughly the magnitude of the whole signal, while stdGP's step
    anneals from 28 down to 1.0 over the run and so gets an exploitation phase.

    If shrinking the step is enough to make TSGP competitive, the fault is in
    step-size control and a better-conditioned model should fix it properly.
    If it is not, the step size is not the binding constraint and no amount of
    SD conditioning will help.

    Semantics are taken on the training inputs, which the search is entitled to
    see.
    """
    from deap import creator

    batches = [search_op.sample_offspring_batch(selected) for _ in range(k)]
    parent_sems = [evaluate_semantics(p, toolbox, X_train) for p in selected]

    chosen = []
    for i in range(len(selected)):
        best, best_d = None, float("inf")
        for b in batches:
            tree = b[i]
            if tree is None or len(tree) == 0:
                continue
            cand = creator.Individual(tree)
            s = evaluate_semantics(cand, toolbox, X_train)
            if not np.all(np.isfinite(s)):
                continue
            d = float(np.linalg.norm(s - parent_sems[i]))
            if np.isfinite(d) and d < best_d:
                best, best_d = tree, d
        chosen.append(best)
    return chosen


def run_tsgp_instrumented(model, X_train, y_train, X_test, y_test,
                          pop_size=None, generations=None, seed=0,
                          warmup=0, closest_of_k=1, verbose=True):
    """TSGP search with per-generation instrumentation.

    ``warmup`` is a DIAGNOSTIC, not part of the method: it runs that many plain
    stdGP generations before handing the population to the transformer. The
    operator is semantically local only on parents that resemble its training
    pool (fitted functions with output norms around 10); on the wild functions
    Ramped Half-and-Half produces (norms of 20-1900) it is close to a random
    draw. Because replacement is fully generational, a search that starts wild
    may never reach the region where the operator works. A warm-up puts the
    population there and isolates that effect.
    """
    pop_size = pop_size or config.GP_POP_SIZE
    generations = generations or config.GP_GENERATIONS
    rng = np.random.default_rng(seed)

    pset = create_pset()
    toolbox = setup_deap(pset)
    toolbox.register("evaluate", evaluate_individual,
                     toolbox=toolbox, X=X_train, y=y_train)
    search_op = TSGPSearchOperator(model, pset)

    pop = toolbox.population(n=pop_size)
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)
    best_ever = _update_best_ever(pop, None, toolbox)

    log = _blank_log()
    log["warmup"] = warmup
    log["closest_of_k"] = closest_of_k
    _record(log, best_ever, pop, [], float("nan"), pop_size,
            _population_diversity(pop, toolbox, X_test, rng))

    for gen in range(1, generations + 1):
        if gen <= warmup:
            pop[:], kp, kc, improved = _stdgp_generation(pop, toolbox, pop_size)
            best_ever = _update_best_ever(pop, best_ever, toolbox)
            _record(log, best_ever, pop,
                    _pair_distances(kp, kc, toolbox, X_test),
                    improved / max(len(kc), 1), len(kc),
                    _population_diversity(pop, toolbox, X_test, rng))
            continue

        selected = toolbox.select(pop, pop_size)
        if closest_of_k > 1:
            child_trees = _closest_of_k(search_op, selected, closest_of_k,
                                        toolbox, X_train, pset)
        else:
            child_trees = search_op.sample_offspring_batch(selected)

        offspring, kept_parents, kept_children = [], [], []
        improved, n_valid = 0, 0
        for parent, child_tree in zip(selected, child_trees):
            accepted = None
            if child_tree is not None and len(child_tree) > 0:
                child = creator.Individual(child_tree)
                try:
                    child.fitness.values = toolbox.evaluate(child)
                    if child.fitness.values[0] < 1e6:
                        accepted = child
                except Exception:
                    accepted = None
            if accepted is not None:
                n_valid += 1
                if accepted.fitness.values[0] < parent.fitness.values[0]:
                    improved += 1
                kept_parents.append(parent)
                kept_children.append(accepted)
                offspring.append(accepted)
            else:
                clone = toolbox.clone(parent)
                clone.fitness.values = parent.fitness.values
                offspring.append(clone)

        dists = _pair_distances(kept_parents, kept_children, toolbox, X_test)
        pop[:] = offspring
        best_ever = _update_best_ever(pop, best_ever, toolbox)
        _record(log, best_ever, pop, dists,
                improved / max(n_valid, 1), n_valid,
                _population_diversity(pop, toolbox, X_test, rng))

        if verbose and (gen % 10 == 0 or gen == 1):
            print(f"  gen {gen:>2}: rmse={log['train_rmse'][-1]:.4f} "
                  f"size={log['best_size'][-1]:>3} "
                  f"popsize={log['pop_median_size'][-1]:>5.1f} "
                  f"dist={log['pair_distance_median'][-1]:>8.3f} "
                  f"impr={log['improvement_rate'][-1]:.3f}", flush=True)

    log["test_rmse"] = float(evaluate_individual(best_ever, toolbox,
                                                 X_test, y_test)[0])
    log["final_size"] = int(len(best_ever))
    log["best_expression"] = str(best_ever)
    return log


def run_stdgp_instrumented(X_train, y_train, X_test, y_test,
                           pop_size=None, generations=None, seed=0,
                           verbose=True):
    pop_size = pop_size or config.GP_POP_SIZE
    generations = generations or config.GP_GENERATIONS
    rng = np.random.default_rng(seed)

    pset = create_pset()
    toolbox = setup_deap(pset)
    toolbox.register("evaluate", evaluate_individual,
                     toolbox=toolbox, X=X_train, y=y_train)

    pop = toolbox.population(n=pop_size)
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)
    best_ever = _update_best_ever(pop, None, toolbox)

    log = _blank_log()
    _record(log, best_ever, pop, [], float("nan"), pop_size,
            _population_diversity(pop, toolbox, X_test, rng))

    for gen in range(1, generations + 1):
        selected = toolbox.select(pop, pop_size)
        # Keep an untouched copy of each parent so parent->offspring pairs can
        # be matched positionally after the in-place DEAP operators run.
        parents_snapshot = [toolbox.clone(ind) for ind in selected]
        offspring = [toolbox.clone(ind) for ind in selected]

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

        varied = [i for i, ind in enumerate(offspring) if not ind.fitness.valid]
        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = toolbox.evaluate(ind)

        improved = sum(
            1 for i in varied
            if offspring[i].fitness.values[0] < parents_snapshot[i].fitness.values[0]
        )
        dists = _pair_distances([parents_snapshot[i] for i in varied],
                                [offspring[i] for i in varied],
                                toolbox, X_test)

        pop[:] = offspring
        best_ever = _update_best_ever(pop, best_ever, toolbox)
        _record(log, best_ever, pop, dists,
                improved / max(len(varied), 1), len(varied),
                _population_diversity(pop, toolbox, X_test, rng))

        if verbose and (gen % 10 == 0 or gen == 1):
            print(f"  gen {gen:>2}: rmse={log['train_rmse'][-1]:.4f} "
                  f"size={log['best_size'][-1]:>3} "
                  f"popsize={log['pop_median_size'][-1]:>5.1f} "
                  f"dist={log['pair_distance_median'][-1]:>8.3f} "
                  f"impr={log['improvement_rate'][-1]:.3f}", flush=True)

    log["test_rmse"] = float(evaluate_individual(best_ever, toolbox,
                                                 X_test, y_test)[0])
    log["final_size"] = int(len(best_ever))
    log["best_expression"] = str(best_ever)
    return log


# ---------------------------------------------------------------------------
# The paper's published reference points, so a run can be scored immediately.
# Table 2 (median test RMSE), Table 3 (median best size), Sect. 4.2 (gen-10
# training RMSE on ESL).
# ---------------------------------------------------------------------------
PAPER = {
    "1027_ESL":               {"tsgp_rmse": 0.379, "stdgp_rmse": 0.502,
                               "tsgp_size": 73, "stdgp_size": 12,
                               "tsgp_gen10_train": 0.381, "stdgp_gen10_train": 0.565},
    "1030_ERA":               {"tsgp_rmse": 0.797, "stdgp_rmse": 0.817,
                               "tsgp_size": 72, "stdgp_size": 60},
    "690_visualizing_galaxy": {"tsgp_rmse": 0.327, "stdgp_rmse": 0.337,
                               "tsgp_size": 64, "stdgp_size": 48},
    "1029_LEV":               {"tsgp_rmse": 0.672, "stdgp_rmse": 0.703,
                               "tsgp_size": 69, "stdgp_size": 50},
    "529_pollen":             {"tsgp_rmse": 0.518, "stdgp_rmse": 0.514,
                               "tsgp_size": 58, "stdgp_size": 37},
}


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Instrument the TSGP/stdGP search and compare the "
                    "per-generation behaviour against the paper's figures.")
    parser.add_argument("--dataset", default="1027_ESL",
                        help="PMLB key. ESL is the sharpest discriminator: "
                             "the paper has TSGP at 0.379 against stdGP 0.502.")
    parser.add_argument("--weights",
                        default="checkpoints_adamw/tsgp_final.npy")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--out", default="results_instrumented")
    parser.add_argument("--methods", default="tsgp,stdgp")
    parser.add_argument("--warmup", type=int, default=0,
                        help="DIAGNOSTIC: run this many plain stdGP "
                             "generations before handing over to the "
                             "transformer, to test whether the operator works "
                             "once the population is inside its training "
                             "distribution.")
    parser.add_argument("--closest-of-k", type=int, default=1,
                        help="DIAGNOSTIC: sample k offspring per parent and "
                             "keep the semantically nearest. Tests whether "
                             "TSGP's failure is step-size control -- stdGP's "
                             "step anneals 28 -> 1.0 over a run while the "
                             "transformer's stays fixed at 16-34. Not the "
                             "paper's operator; k=1 is.")
    args = parser.parse_args()

    from .datasets import load_and_prepare_dataset, DISPLAY_NAMES
    from .experiment_units import load_model, set_seeds, unit_seed

    methods = args.methods.split(",")
    os.makedirs(args.out, exist_ok=True)

    model = None
    if "tsgp" in methods:
        print(f"loading {args.weights} ...", flush=True)
        model = load_model(args.weights)

    name = DISPLAY_NAMES.get(args.dataset, args.dataset)
    print(f"\ndataset {name}  ({args.runs} runs/method)")

    all_logs = {m: [] for m in methods}
    for run in range(args.runs):
        for method in methods:
            seed = unit_seed(args.dataset, method, run)
            set_seeds(seed)
            # Sect. 4.1: a 50/50 split per run. The prior code pinned
            # random_state=42, so all 30 "independent" runs shared one split.
            X_tr, X_te, y_tr, y_te = load_and_prepare_dataset(
                args.dataset, split_seed=seed)
            print(f"\n[{method}] run {run} (seed {seed})", flush=True)
            if method == "tsgp":
                log = run_tsgp_instrumented(model, X_tr, y_tr, X_te, y_te,
                                            generations=args.generations,
                                            seed=seed, warmup=args.warmup,
                                            closest_of_k=args.closest_of_k)
            else:
                log = run_stdgp_instrumented(X_tr, y_tr, X_te, y_te,
                                             generations=args.generations,
                                             seed=seed)
            log["run"] = run
            log["seed"] = seed
            all_logs[method].append(log)
            with open(os.path.join(args.out,
                                   f"{args.dataset}__{method}__run{run:02d}.json"),
                      "w") as f:
                json.dump(log, f, indent=2)

    summarise(all_logs, args.dataset)


def summarise(all_logs, dataset):
    ref = PAPER.get(dataset, {})
    print("\n" + "=" * 78)
    print(f"SUMMARY — {dataset}")
    print("=" * 78)

    for method, logs in all_logs.items():
        if not logs:
            continue
        test = np.median([l["test_rmse"] for l in logs])
        size = np.median([l["final_size"] for l in logs])
        curves = np.array([l["train_rmse"] for l in logs])
        sizes = np.array([l["best_size"] for l in logs])
        dist = np.array([l["pair_distance_median"] for l in logs])
        impr = np.array([l["improvement_rate"] for l in logs])

        pr = ref.get(f"{method}_rmse")
        ps = ref.get(f"{method}_size")
        print(f"\n{method.upper()}")
        print(f"  median test RMSE   {test:.4f}"
              + (f"   paper {pr:.3f}   delta {test - pr:+.3f}" if pr else ""))
        print(f"  median best size   {size:.0f}"
              + (f"   paper {ps}" if ps else ""))
        g10 = ref.get(f"{method}_gen10_train")
        if len(curves[0]) > 10:
            print(f"  gen-10 train RMSE  {np.median(curves[:, 10]):.4f}"
                  + (f"   paper {g10:.3f}" if g10 else ""))
        print(f"  train RMSE by gen  " + "  ".join(
            f"g{g}={np.median(curves[:, g]):.3f}"
            for g in [0, 5, 10, 20, 30, 50] if g < curves.shape[1]))
        print(f"  best size by gen   " + "  ".join(
            f"g{g}={np.median(sizes[:, g]):.0f}"
            for g in [0, 5, 10, 20, 30, 50] if g < sizes.shape[1]))
        print(f"  parent->child dist " + "  ".join(
            f"g{g}={np.nanmedian(dist[:, g]):.2f}"
            for g in [1, 5, 10, 20, 30, 50] if g < dist.shape[1]))
        print(f"  improvement rate   " + "  ".join(
            f"g{g}={np.nanmedian(impr[:, g]):.3f}"
            for g in [1, 5, 10, 20, 30, 50] if g < impr.shape[1]))
        last_improve = [int(np.max(np.where(np.diff(c) < -1e-9)[0]) + 1)
                        if np.any(np.diff(c) < -1e-9) else 0 for c in curves]
        n_improve = [int(np.sum(np.diff(c) < -1e-9)) for c in curves]
        print(f"  final improvement at gen {np.median(last_improve):.1f}; "
              f"{np.median(n_improve):.1f} of {curves.shape[1]-1} gens improved")


if __name__ == "__main__":
    main()
