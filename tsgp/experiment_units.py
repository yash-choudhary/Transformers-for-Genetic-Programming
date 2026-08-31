"""The experiment grid, expressed as independently-restartable units.

A *unit* is one (dataset, method, run) triple. Each unit writes its own JSON
file under ``results/runs/``, so a failure anywhere only costs the unit that
was in flight — everything already on disk is skipped next time.

Units are seeded deterministically from their own identity, so re-running a
unit reproduces it exactly and a resumed grid is indistinguishable from one
that ran start to finish.
"""
import json
import os
import random
import time
import zlib
from datetime import datetime

import numpy as np

from . import config
from .datasets import DATASETS, load_and_prepare_dataset

RESULTS_DIR = "results"
RUNS_SUBDIR = "runs"
METHODS = ["tsgp", "stdgp"]


def runs_dir(output_dir=RESULTS_DIR):
    return os.path.join(output_dir, RUNS_SUBDIR)


def unit_name(dataset, method, run):
    return f"{dataset}__{method}__run{run:02d}"


def unit_path(dataset, method, run, output_dir=RESULTS_DIR):
    return os.path.join(runs_dir(output_dir),
                        unit_name(dataset, method, run) + ".json")


def is_done(dataset, method, run, output_dir=RESULTS_DIR):
    """True only if the unit's file exists and parses.

    A truncated file (killed mid-write on an older run) counts as not done, so
    it gets redone rather than silently poisoning the aggregate.
    """
    path = unit_path(dataset, method, run, output_dir)
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            json.load(f)
        return True
    except (json.JSONDecodeError, OSError):
        return False


def iter_units(datasets=None, methods=None, num_runs=None):
    if datasets is None:
        datasets = DATASETS
    if methods is None:
        methods = METHODS
    if num_runs is None:
        num_runs = config.NUM_RUNS
    for dataset in datasets:
        for run in range(num_runs):
            for method in methods:
                yield dataset, method, run


def unit_seed(dataset, method, run):
    key = f"{dataset}|{method}|{run}".encode()
    return zlib.crc32(key) & 0x7FFFFFFF


def set_seeds(seed):
    random.seed(seed)          # DEAP's variation operators use this
    np.random.seed(seed)       # token sampling in TSGPSearchOperator
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def _write_atomic(path, payload):
    """Write via a temp file + rename so a crash can't leave partial JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def load_model(weights_path, normalize_sd=None):
    """Build the transformer and load .npy (current) or .weights.h5 (legacy).

    `normalize_sd` must match how the checkpoint was trained; it changes no
    weights, only how SD is fed in.
    """
    from .transformer_model import create_model
    model = create_model(normalize_sd=normalize_sd)
    if weights_path.endswith(".npy"):
        data = np.load(weights_path, allow_pickle=True)
        model.set_weights(list(data))
    else:
        model.load_weights(weights_path)
    return model


def run_unit(dataset, method, run, model=None, output_dir=RESULTS_DIR,
             temperature=None, verbose=True):
    """Execute one unit and persist its result. Returns the payload."""
    from .tsgp_search import run_tsgp, run_stdgp_baseline

    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; expected one of {METHODS}")
    if method == "tsgp" and model is None:
        raise ValueError("method='tsgp' requires a loaded model")

    seed = unit_seed(dataset, method, run)
    set_seeds(seed)

    X_train, X_test, y_train, y_test = load_and_prepare_dataset(dataset)

    started = time.perf_counter()
    if method == "tsgp":
        best_ind, stats = run_tsgp(model, X_train, y_train, X_test, y_test,
                                   temperature=temperature, verbose=verbose)
    else:
        best_ind, stats = run_stdgp_baseline(X_train, y_train, X_test, y_test,
                                             verbose=verbose)
    elapsed = time.perf_counter() - started

    payload = {
        "dataset": dataset,
        "method": method,
        "run": run,
        "seed": seed,
        "test_rmse": float(stats["test_rmse"]),
        "train_rmse": float(best_ind.fitness.values[0]),
        "best_size": int(len(best_ind)),
        "best_expression": str(best_ind),
        # best-so-far series (paper Fig. 2 / Table 2 basis)
        "train_rmse_history": [float(x) for x in stats["train_rmse"]],
        "best_size_history": [int(x) for x in stats["best_size"]],
        # current-population series (diagnostic: what drives the search)
        "train_rmse_pop_history": [float(x) for x in stats["train_rmse_pop"]],
        "best_size_pop_history": [int(x) for x in stats["best_size_pop"]],
        "elapsed_sec": round(elapsed, 2),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "gp_pop_size": config.GP_POP_SIZE,
        "gp_generations": config.GP_GENERATIONS,
        # Recorded so a results directory is never ambiguous about which
        # sampling temperature produced it.
        "temperature": (None if method == "stdgp" else
                        (config.TSGP_TEMPERATURE if temperature is None
                         else temperature)),
    }
    _write_atomic(unit_path(dataset, method, run, output_dir), payload)
    return payload


def load_completed(datasets=None, methods=None, num_runs=None,
                   output_dir=RESULTS_DIR):
    """Return (results, missing) across the requested grid."""
    results, missing = [], []
    for dataset, method, run in iter_units(datasets, methods, num_runs):
        path = unit_path(dataset, method, run, output_dir)
        try:
            with open(path) as f:
                results.append(json.load(f))
        except (json.JSONDecodeError, OSError):
            missing.append((dataset, method, run))
    return results, missing
