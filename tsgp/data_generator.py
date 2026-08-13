import csv
import os
import pickle
import random
import numpy as np
import faiss
from deap import algorithms, gp

from . import config
from .primitives import (create_pset, setup_datagen_toolbox,
                         evaluate_individual, evaluate_semantics, tree_to_prefix)
from .tokenizer import tree_to_tokens


def generate_synthetic_problem(num_samples=200, num_features=None, noise_std=0.1):
    if num_features is None:
        num_features = config.NUM_FEATURES
    X = np.random.randn(num_samples, num_features)
    coefficients = np.random.randn(num_features)
    intercept = np.random.randn()
    y = X @ coefficients + intercept
    y += np.random.randn(num_samples) * noise_std

    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X = (X - X_mean) / X_std

    y_mean = y.mean()
    y_std = y.std() + 1e-8
    y = (y - y_mean) / y_std

    return X, y


def run_stdgp_for_functions(X, y, pop_size=None, generations=None, verbose=False):
    if pop_size is None:
        pop_size = config.DATAGEN_GP_POP_SIZE
    if generations is None:
        generations = config.DATAGEN_GP_GENERATIONS

    pset = create_pset(num_features=X.shape[1])
    toolbox = setup_datagen_toolbox(pset)
    toolbox.register("evaluate", evaluate_individual,
                     toolbox=toolbox, X=X, y=y)

    pop = toolbox.population(n=pop_size)
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)

    unique_functions = {}
    for ind in pop:
        key = str(ind)
        if key not in unique_functions:
            unique_functions[key] = toolbox.clone(ind)

    for gen in range(1, generations + 1):
        offspring = toolbox.select(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

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
            key = str(ind)
            if key not in unique_functions:
                unique_functions[key] = toolbox.clone(ind)

        if verbose and gen % 20 == 0:
            best = min(pop, key=lambda x: x.fitness.values[0])
            print(f"  Gen {gen}: best RMSE={best.fitness.values[0]:.4f}, "
                  f"unique={len(unique_functions)}")

    return list(unique_functions.values()), pset


def compute_all_semantics(functions, pset, num_samples=None):
    if num_samples is None:
        num_samples = config.NUM_SEMANTIC_SAMPLES

    from .primitives import setup_deap
    toolbox = setup_deap(pset)

    X_semantic = np.random.randn(num_samples, config.NUM_FEATURES)

    semantics = []
    valid_indices = []
    for i, func in enumerate(functions):
        sem = evaluate_semantics(func, toolbox, X_semantic)
        if np.all(np.isfinite(sem)) and np.std(sem) > 1e-10:
            semantics.append(sem)
            valid_indices.append(i)

    semantics = np.array(semantics, dtype=np.float32)
    return semantics, valid_indices


def find_semantic_pairs(functions, semantics, valid_indices, k=None,
                        sd_max=None):
    if k is None:
        k = config.KNN_K
    if sd_max is None:
        sd_max = config.SD_MAX_THRESHOLD

    n = semantics.shape[0]
    dim = semantics.shape[1]

    if n > 100_000:
        nlist = min(int(np.sqrt(n)), 4096)
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist)
        index.train(semantics)
        index.add(semantics)
        index.nprobe = min(nlist, 64)
    else:
        index = faiss.IndexFlatL2(dim)
        index.add(semantics)

    distances, indices = index.search(semantics, k + 1)

    # Walk the pool in a random order rather than insertion order. The pool is
    # built by appending synthetic problems 1..50 in sequence, so the previous
    # version -- which returned the moment TARGET_NUM_PAIRS was reached while
    # iterating i in order -- silently dropped every problem past the point
    # where the cap bound. The saved 5M-pair file hit the cap exactly, so the
    # later synthetic problems contributed nothing at all.
    order = np.random.permutation(n)

    pairs = []
    excluded_no_neighbour = 0
    for i in order:
        kept_any = False
        for j_pos in range(k + 1):
            j = indices[i, j_pos]
            if j == i or j < 0:
                continue
            sd = np.sqrt(max(distances[i, j_pos], 0.0))
            if sd > 0 and sd < sd_max:
                pairs.append((valid_indices[i], valid_indices[j], sd))
                kept_any = True
        if not kept_any:
            excluded_no_neighbour += 1
        if len(pairs) >= config.TARGET_NUM_PAIRS:
            break

    # Coverage is the number that matters for whether the operator will work on
    # what the GP search actually feeds it: a function whose nearest neighbour
    # is further than sd_max contributes no training pair at all, so the model
    # never learns to vary anything that looks like it.
    seen = min(n, len(order))
    print(f"  pairs: {len(pairs):,} from {seen:,} pool functions visited; "
          f"{excluded_no_neighbour:,} had no neighbour within SD<{sd_max} "
          f"({excluded_no_neighbour / max(seen, 1) * 100:.1f}% uncovered)")
    return pairs


def coverage_by_norm(semantics, k=None, sd_max_values=(100.0, 250.0, 1000.0)):
    """What fraction of the pool gets a training pair, broken down by norm.

    Global coverage is the wrong number to act on. A pool can look badly
    uncovered while every function the search actually visits is fine, because
    the uncovered tail is degenerate blow-ups from protected division that no
    search would ever select. What matters is coverage over the norm range the
    real search operates in -- measured at 20-75 for classification.
    """
    k = k or config.KNN_K
    n, dim = semantics.shape
    index = faiss.IndexFlatL2(dim)
    index.add(semantics)
    distances, indices = index.search(semantics, k + 1)

    norms = np.linalg.norm(semantics, axis=1)
    # nearest non-self neighbour distance for each function
    nn = np.full(n, np.inf)
    for i in range(n):
        for j_pos in range(k + 1):
            j = indices[i, j_pos]
            if j == i or j < 0:
                continue
            d = np.sqrt(max(distances[i, j_pos], 0.0))
            if d > 0:
                nn[i] = min(nn[i], d)
                break

    buckets = [(0, 10), (10, 25), (25, 50), (50, 100), (100, 250),
               (250, 1e9)]
    lines = ["coverage by semantic norm (fraction with a neighbour inside "
             "each SD threshold):",
             f"  {'norm range':>14} {'n':>9} " +
             " ".join(f"{'SD<' + str(int(s)):>9}" for s in sd_max_values)]
    for lo, hi in buckets:
        m = (norms >= lo) & (norms < hi)
        if m.sum() == 0:
            continue
        cells = " ".join(f"{np.mean(nn[m] < s):>9.3f}" for s in sd_max_values)
        lines.append(f"  {f'{lo}-{hi:g}':>14} {m.sum():>9,} {cells}")
    text = "\n".join(lines)
    print(text)
    return text


def report_pool_coverage(semantics, valid_indices, out_path=None):
    """Describe the semantic regime the pool actually covers.

    The GP search starts from Ramped Half-and-Half, whose functions have very
    large output norms; the pool is made of functions that survived stdGP
    selection against standardised targets, whose norms are small. If those two
    distributions do not overlap, the transformer is queried out of
    distribution for the whole search and its offspring are close to random
    draws -- which is what we measure on the current checkpoint.
    """
    norms = np.linalg.norm(semantics, axis=1)
    q = [1, 5, 25, 50, 75, 95, 99]
    lines = ["pool semantic norms ||s(f)||:"]
    lines += [f"  p{p:<3} {np.percentile(norms, p):12.3f}" for p in q]
    lines.append(f"  mean {norms.mean():.3f}   max {norms.max():.3f}")
    for thr in (10, 50, 100, 500):
        lines.append(f"  fraction with norm > {thr:<4}: "
                     f"{np.mean(norms > thr):.4f}")
    text = "\n".join(lines)
    print(text)
    if out_path:
        with open(out_path, "w") as f:
            f.write(text + "\n")
    return norms


def generate_training_data(output_dir, num_problems=None, verbose=True):
    if num_problems is None:
        num_problems = config.NUM_SYNTHETIC_PROBLEMS

    os.makedirs(output_dir, exist_ok=True)

    all_functions = []
    all_token_seqs = []
    pset = None

    for prob_idx in range(num_problems):
        if verbose:
            print(f"Synthetic problem {prob_idx + 1}/{num_problems}")

        X, y = generate_synthetic_problem(
            num_samples=200,
            noise_std=random.uniform(0.05, 0.3)
        )

        functions, pset = run_stdgp_for_functions(
            X, y,
            pop_size=config.DATAGEN_GP_POP_SIZE,
            generations=config.DATAGEN_GP_GENERATIONS,
            verbose=verbose
        )

        for func in functions:
            tokens = tree_to_tokens(func)
            if len(tokens) <= config.TRANSFORMER_MAX_SEQ_LEN - 2:
                all_functions.append(func)
                all_token_seqs.append(tokens)

        if verbose:
            print(f"  Collected {len(functions)} unique, "
                  f"total so far: {len(all_functions)}")

    if verbose:
        print(f"\nTotal functions collected: {len(all_functions)}")
        print("Computing semantics...")

    semantics, valid_indices = compute_all_semantics(all_functions, pset)
    if verbose:
        print(f"Valid functions with finite semantics: {len(valid_indices)}")
        report_pool_coverage(semantics, valid_indices,
                             os.path.join(output_dir, "pool_coverage.txt"))
        print("Finding semantic pairs...")

    pairs = find_semantic_pairs(all_functions, semantics, valid_indices)
    if verbose:
        print(f"Total semantic pairs: {len(pairs)}")

    training_data = []
    for orig_i, orig_j, sd in pairs:
        input_tokens = all_token_seqs[orig_i]
        output_tokens = all_token_seqs[orig_j]
        training_data.append({
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            # Plain float, not the np.float32 that FAISS returns: a numpy
            # scalar makes the pickle depend on the writer's numpy version
            # (numpy >= 1.26 emits numpy._core references that older numpy
            # cannot resolve), which breaks loading in a different env.
            "sd": float(sd),
        })

    pkl_path = os.path.join(output_dir, "training_pairs.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(training_data, f)

    csv_path = os.path.join(output_dir, "training_pairs.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["input_tokens", "output_tokens", "sd"])
        for d in training_data:
            writer.writerow([
                " ".join(d["input_tokens"]),
                " ".join(d["output_tokens"]),
                f"{d['sd']:.6f}",
            ])

    if verbose:
        print(f"Saved {len(training_data)} training pairs to {pkl_path}")
        print(f"Saved CSV to {csv_path}")

    return training_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate the transformer's semantic training pairs.")
    parser.add_argument("--output", default="data/training",
                        help="Directory for training_pairs.pkl/.csv. Point "
                             "this at a new directory to keep an existing "
                             "generation intact.")
    parser.add_argument("--problems", type=int,
                        default=config.NUM_SYNTHETIC_PROBLEMS,
                        help="Number of synthetic SR problems")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    generate_training_data(args.output, num_problems=args.problems,
                           verbose=not args.quiet)
