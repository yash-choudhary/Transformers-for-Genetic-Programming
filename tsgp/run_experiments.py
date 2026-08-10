"""Step 3c — run the full benchmark grid, resumably.

Walks every (dataset, method, run) unit, skipping any that already has a
result on disk and carrying on past units that fail. Re-running the same
command picks up exactly where it left off, so a crash, a dropped network
connection, or a Ctrl-C costs at most the unit in flight.

    python -m tsgp.run_experiments --weights checkpoints/tsgp_final.npy

Narrow the grid while testing:

    python -m tsgp.run_experiments --weights checkpoints/tsgp_final.npy \
        --datasets 1030_ERA --runs 1
"""
import argparse
import sys
import time
import traceback

from . import config
from .aggregate_results import aggregate
from .datasets import DATASETS, DISPLAY_NAMES, load_and_prepare_dataset  # noqa: F401
from .experiment_units import (METHODS, RESULTS_DIR, is_done, iter_units,
                               load_model, run_unit)


def run_all_experiments(model_weights_path, output_dir=RESULTS_DIR,
                        num_runs=None, datasets=None, methods=None,
                        force=False, temperature=None, normalize_sd=None,
                        step_k=None, step_anneal=None, verbose=True):
    if num_runs is None:
        num_runs = config.NUM_RUNS
    if datasets is None:
        datasets = DATASETS
    if methods is None:
        methods = METHODS

    units = list(iter_units(datasets, methods, num_runs))
    pending = [u for u in units
               if force or not is_done(*u, output_dir=output_dir)]

    print(f"{len(units)} units in grid, {len(units) - len(pending)} already "
          f"done, {len(pending)} to run.")
    if not pending:
        print("Nothing to do.")
        # The summary is the deliverable, so it prints regardless of --quiet;
        # `verbose` only controls per-generation GP chatter.
        return aggregate(datasets, methods, num_runs, output_dir)

    model = None
    if any(method == "tsgp" for _, method, _ in pending):
        print(f"Loading model weights from {model_weights_path}")
        model = load_model(model_weights_path, normalize_sd=normalize_sd)
        # The encoding is detected from the checkpoint's weight shapes, so say
        # which one was actually loaded rather than which one config prefers.
        if model.sd_encoding == "binned":
            print(f"SD conditioning: {config.TRANSFORMER_SD_NUM_BINS} log1p "
                  f"bins -> learned embedding")
        else:
            print(f"SD conditioning: rank-1 Dense on the "
                  f"{'log1p-standardised' if model.normalize_sd else 'raw'} "
                  f"scalar")

    failures = []
    started = time.perf_counter()

    for idx, (dataset, method, run) in enumerate(pending, start=1):
        label = DISPLAY_NAMES.get(dataset, dataset)
        elapsed = time.perf_counter() - started
        eta = (elapsed / (idx - 1) * (len(pending) - idx + 1)) if idx > 1 else None
        header = (f"[{idx}/{len(pending)}] {label} / {method} / run {run}")
        if eta is not None:
            header += f"   (elapsed {elapsed/60:.1f}m, eta {eta/60:.1f}m)"
        print(f"\n{'='*70}\n{header}\n{'='*70}")

        try:
            payload = run_unit(dataset, method, run, model=model,
                               output_dir=output_dir, temperature=temperature,
                               step_k=step_k, step_anneal=step_anneal,
                               verbose=verbose)
            print(f"-> test_rmse={payload['test_rmse']:.4f} "
                  f"size={payload['best_size']} "
                  f"({payload['elapsed_sec']:.1f}s)")
        except KeyboardInterrupt:
            print("\nInterrupted. Completed units are saved; "
                  "re-run the same command to resume.")
            break
        except Exception:
            # One bad unit must not cost the whole grid.
            print(f"FAILED: {dataset} / {method} / run {run}")
            traceback.print_exc()
            failures.append((dataset, method, run))

    if failures:
        print(f"\n{len(failures)} unit(s) failed:")
        for dataset, method, run in failures:
            print(f"  {dataset} / {method} / run {run}")
        print("Re-run the same command to retry only these.")

    print()
    return aggregate(datasets, methods, num_runs, output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",
                        help="Path to trained model weights "
                             "(required unless --methods stdgp)")
    parser.add_argument("--output", default=RESULTS_DIR,
                        help="Output directory for results")
    parser.add_argument("--runs", type=int, default=config.NUM_RUNS,
                        help="Number of independent runs per dataset")
    parser.add_argument("--datasets", nargs="+", default=DATASETS,
                        choices=DATASETS)
    parser.add_argument("--methods", nargs="+", default=METHODS,
                        choices=METHODS)
    parser.add_argument("--force", action="store_true",
                        help="Recompute units that already have results")
    parser.add_argument("--temperature", type=float, default=None,
                        help=f"Sampling temperature for the TSGP operator "
                             f"(default {config.TSGP_TEMPERATURE}). Paper-silent "
                             f"parameter; recorded in every result file.")
    parser.add_argument("--sd-normalize", action="store_true",
                        help="Checkpoint was trained with normalised SD "
                             "conditioning; must match training.")
    parser.add_argument("--step-k", type=int, default=None,
                        help=f"Sample k offspring per parent and keep the "
                             f"semantically nearest (default "
                             f"{config.TSGP_STEP_K}). k=1 is the paper's "
                             f"operator. k>1 is the step-size control the "
                             f"paper lists as future work in Sect. 5, needed "
                             f"because a single sample cannot honour SD_d=0.1 "
                             f"-- its achieved distance floors at ~0.7. "
                             f"Recorded in every result file.")
    parser.add_argument("--step-anneal", action="store_true",
                        help="With --step-k > 1, pick the candidate whose "
                             "parent distance is closest to a target that "
                             "decays across the run, instead of the nearest "
                             "candidate. Imitates the annealing stdGP gets "
                             "free from crossover (28 -> 1.0 on ESL) and "
                             "which TSGP lacks. Sect. 5 future work, not the "
                             "paper's operator.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if "tsgp" in args.methods and not args.weights:
        parser.error("--weights is required unless you pass --methods stdgp")

    run_all_experiments(args.weights, args.output, args.runs,
                        datasets=args.datasets, methods=args.methods,
                        force=args.force, temperature=args.temperature,
                        normalize_sd=True if args.sd_normalize else None,
                        step_k=args.step_k,
                        step_anneal=True if args.step_anneal else None,
                        verbose=not args.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
