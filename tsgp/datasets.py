"""PMLB dataset loading, kept free of any TensorFlow import.

Importing this module is cheap, so the dataset-fetching step doesn't pay the
~10s TF startup cost before it can touch the network.
"""
import time

import pmlb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from . import config


# PMLB dataset names are ID-prefixed; the bare names from the paper
# (ERA, ESL, Galaxy, LEV, pollen) are not valid fetch_data keys.
# All five are regression tasks with exactly 4 continuous features.
DATASETS = [
    "1030_ERA",
    "1027_ESL",
    "690_visualizing_galaxy",
    "1029_LEV",
    "529_pollen",
]

# Paper's short names, for reporting.
DISPLAY_NAMES = {
    "1030_ERA": "ERA",
    "1027_ESL": "ESL",
    "690_visualizing_galaxy": "Galaxy",
    "1029_LEV": "LEV",
    "529_pollen": "pollen",
}

PMLB_CACHE_DIR = "data/pmlb_cache"


def fetch_dataset(name, cache_dir=PMLB_CACHE_DIR, attempts=5, base_delay=2.0):
    """Fetch a PMLB dataset, retrying on transient network failures.

    PMLB downloads from GitHub on a cache miss; a dropped connection there
    would otherwise abort whatever experiment happened to be running.
    """
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            return pmlb.fetch_data(name, return_X_y=True,
                                   local_cache_dir=cache_dir)
        except Exception as exc:  # network errors surface in several flavours
            last_error = exc
            if attempt == attempts:
                break
            delay = base_delay * (2 ** (attempt - 1))
            print(f"  fetch of {name} failed ({type(exc).__name__}: {exc}); "
                  f"retry {attempt}/{attempts - 1} in {delay:.0f}s")
            time.sleep(delay)
    raise RuntimeError(
        f"Could not fetch dataset {name} after {attempts} attempts. "
        f"Last error: {last_error}"
    )


def load_and_prepare_dataset(name, cache_dir=PMLB_CACHE_DIR):
    X, y = fetch_dataset(name, cache_dir=cache_dir)
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
