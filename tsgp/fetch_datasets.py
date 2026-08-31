"""Step 3a — download and cache the PMLB benchmark datasets.

Run this once before the experiments so a flaky network can't take down a
compute run hours in. Imports no TensorFlow, so it starts instantly.

    python -m tsgp.fetch_datasets
"""
import argparse
import sys

from .datasets import (DATASETS, DISPLAY_NAMES, PMLB_CACHE_DIR,
                       load_and_prepare_dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DATASETS,
                        help="Subset of PMLB dataset names to fetch")
    parser.add_argument("--cache-dir", default=PMLB_CACHE_DIR)
    args = parser.parse_args()

    failed = []
    for name in args.datasets:
        label = DISPLAY_NAMES.get(name, name)
        print(f"{name} ({label}) ...")
        try:
            X_train, X_test, y_train, y_test = load_and_prepare_dataset(
                name, cache_dir=args.cache_dir)
        except Exception as exc:
            print(f"  FAILED: {exc}")
            failed.append(name)
            continue
        print(f"  train={X_train.shape}  test={X_test.shape}  "
              f"y_train mean={y_train.mean():+.3f} std={y_train.std():.3f}")

    print()
    if failed:
        print(f"{len(failed)} dataset(s) failed: {', '.join(failed)}")
        print("Re-run this script to retry only what's missing "
              "(cached datasets are not re-downloaded).")
        return 1
    print(f"All {len(args.datasets)} datasets cached under {args.cache_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
