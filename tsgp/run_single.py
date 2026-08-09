"""Step 3b — run ONE experiment unit (one dataset, one method, one run).

The smallest restartable piece of the grid. Use this to test a single
configuration, to redo a unit that failed, or to drive the grid from an
external scheduler.

    python -m tsgp.run_single --dataset 1030_ERA --method tsgp --run 0 \
        --weights checkpoints/tsgp_final.npy

    python -m tsgp.run_single --dataset 1030_ERA --method stdgp --run 0
"""
import argparse
import sys

from .datasets import DATASETS
from .experiment_units import (METHODS, RESULTS_DIR, is_done, load_model,
                               run_unit, unit_path, unit_seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--run", type=int, required=True,
                        help="Run index (0-based)")
    parser.add_argument("--weights",
                        help="Path to model weights; required for --method tsgp")
    parser.add_argument("--output", default=RESULTS_DIR)
    parser.add_argument("--force", action="store_true",
                        help="Recompute even if this unit already has a result")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Sampling temperature for the TSGP operator")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    path = unit_path(args.dataset, args.method, args.run, args.output)
    if not args.force and is_done(args.dataset, args.method, args.run,
                                  args.output):
        print(f"Already done: {path}  (use --force to recompute)")
        return 0

    if args.method == "tsgp" and not args.weights:
        parser.error("--weights is required when --method tsgp")

    model = load_model(args.weights) if args.method == "tsgp" else None

    print(f"{args.dataset} / {args.method} / run {args.run} "
          f"(seed {unit_seed(args.dataset, args.method, args.run)})")
    payload = run_unit(args.dataset, args.method, args.run, model=model,
                       output_dir=args.output, temperature=args.temperature,
                       verbose=not args.quiet)

    print(f"\ntest_rmse={payload['test_rmse']:.4f}  "
          f"size={payload['best_size']}  "
          f"elapsed={payload['elapsed_sec']:.1f}s")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
