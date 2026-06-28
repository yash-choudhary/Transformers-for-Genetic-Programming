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

    pairs = []
    for i in range(n):
        for j_pos in range(k + 1):
            j = indices[i, j_pos]
            if j == i:
                continue
            sd = np.sqrt(distances[i, j_pos])
            if sd > 0 and sd < sd_max:
                orig_i = valid_indices[i]
                orig_j = valid_indices[j]
                pairs.append((orig_i, orig_j, sd))
            if len(pairs) >= config.TARGET_NUM_PAIRS:
                return pairs
    return pairs


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
            "sd": sd,
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
    data = generate_training_data("data/training", verbose=True)
