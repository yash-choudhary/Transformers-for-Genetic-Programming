import os
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pmlb

from . import config
from .transformer_model import create_model
from .tsgp_search import run_tsgp, run_stdgp_baseline


def _load_model_weights(model, path):
    """Load weights from .npy (new) or legacy .weights.h5 format."""
    if path.endswith(".npy"):
        data = np.load(path, allow_pickle=True)
        model.set_weights(list(data))
    else:
        model.load_weights(path)


DATASETS = ["ERA", "ESL", "Galaxy", "LEV", "pollen"]


def load_and_prepare_dataset(name):
    X, y = pmlb.fetch_data(name, return_X_y=True)
    X = X[:, :config.NUM_FEATURES]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42)

    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)

    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

    return X_train, X_test, y_train, y_test


def run_all_experiments(model_weights_path, output_dir="results",
                        num_runs=None, verbose=True):
    if num_runs is None:
        num_runs = config.NUM_RUNS
    os.makedirs(output_dir, exist_ok=True)

    model = create_model()
    _load_model_weights(model, model_weights_path)

    all_results = {}

    for dataset_name in DATASETS:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Dataset: {dataset_name}")
            print(f"{'='*60}")

        X_train, X_test, y_train, y_test = load_and_prepare_dataset(
            dataset_name)
        if verbose:
            print(f"Train: {X_train.shape}, Test: {X_test.shape}")

        tsgp_results = []
        stdgp_results = []

        for run in range(num_runs):
            if verbose:
                print(f"\n--- Run {run + 1}/{num_runs} ---")

            if verbose:
                print("TSGP:")
            _, tsgp_stats = run_tsgp(
                model, X_train, y_train, X_test, y_test,
                verbose=verbose)
            tsgp_results.append(tsgp_stats)

            if verbose:
                print("\nstdGP:")
            _, stdgp_stats = run_stdgp_baseline(
                X_train, y_train, X_test, y_test,
                verbose=verbose)
            stdgp_results.append(stdgp_stats)

        tsgp_test_rmses = [r["test_rmse"] for r in tsgp_results]
        stdgp_test_rmses = [r["test_rmse"] for r in stdgp_results]
        tsgp_sizes = [r["best_size"][-1] for r in tsgp_results]
        stdgp_sizes = [r["best_size"][-1] for r in stdgp_results]

        all_results[dataset_name] = {
            "tsgp": {
                "test_rmse_median": float(np.median(tsgp_test_rmses)),
                "test_rmse_mean": float(np.mean(tsgp_test_rmses)),
                "test_rmse_std": float(np.std(tsgp_test_rmses)),
                "test_rmses": [float(x) for x in tsgp_test_rmses],
                "size_median": float(np.median(tsgp_sizes)),
                "sizes": [int(x) for x in tsgp_sizes],
            },
            "stdgp": {
                "test_rmse_median": float(np.median(stdgp_test_rmses)),
                "test_rmse_mean": float(np.mean(stdgp_test_rmses)),
                "test_rmse_std": float(np.std(stdgp_test_rmses)),
                "test_rmses": [float(x) for x in stdgp_test_rmses],
                "size_median": float(np.median(stdgp_sizes)),
                "sizes": [int(x) for x in stdgp_sizes],
            },
        }

        if verbose:
            print(f"\n{dataset_name} Summary:")
            print(f"  TSGP  — Median Test RMSE: "
                  f"{np.median(tsgp_test_rmses):.4f}, "
                  f"Median Size: {np.median(tsgp_sizes):.0f}")
            print(f"  stdGP — Median Test RMSE: "
                  f"{np.median(stdgp_test_rmses):.4f}, "
                  f"Median Size: {np.median(stdgp_sizes):.0f}")

    results_path = os.path.join(output_dir, "experiment_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    if verbose:
        print(f"\n{'='*60}")
        print("Final Results (Median Test RMSE):")
        print(f"{'Dataset':<12} {'TSGP':<10} {'stdGP':<10}")
        print("-" * 32)
        for name in DATASETS:
            r = all_results[name]
            print(f"{name:<12} "
                  f"{r['tsgp']['test_rmse_median']:<10.4f} "
                  f"{r['stdgp']['test_rmse_median']:<10.4f}")
        print(f"\nResults saved to {results_path}")

    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True,
                        help="Path to trained model weights")
    parser.add_argument("--output", default="results",
                        help="Output directory for results")
    parser.add_argument("--runs", type=int, default=config.NUM_RUNS,
                        help="Number of independent runs per dataset")
    args = parser.parse_args()
    run_all_experiments(args.weights, args.output, args.runs)
