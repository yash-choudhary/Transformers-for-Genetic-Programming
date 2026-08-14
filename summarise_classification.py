"""Compare the three classification arms.

  results_clf           stdGP baseline
  results_clf_transfer  the REGRESSION-trained operator, applied to
                        classification without retraining (the transfer control)
  results_clf_new       the operator trained on classification-regime data

The majority-class baseline is printed with every row. Without it an accuracy
figure is uninterpretable: on a set that is 60% one class, 0.60 means the search
found nothing.
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np

try:
    from scipy.stats import ranksums
except ImportError:
    ranksums = None

ARMS = [("stdGP", "results_clf"),
        ("TSGP-transfer", "results_clf_transfer"),
        ("TSGP-clf", "results_clf_new")]

ORDER = ["1030_ERA", "1027_ESL", "690_visualizing_galaxy", "1029_LEV",
         "529_pollen"]
SHORT = {"1030_ERA": "ERA", "1027_ESL": "ESL",
         "690_visualizing_galaxy": "Galaxy", "1029_LEV": "LEV",
         "529_pollen": "pollen"}


def load(d):
    out = defaultdict(list)
    for f in glob.glob(os.path.join(d, "runs", "*.json")):
        try:
            r = json.load(open(f))
        except (json.JSONDecodeError, OSError):
            continue
        out[r["dataset"]].append(r)
    return out


data = {name: load(d) for name, d in ARMS if os.path.isdir(d)}
maj = {}
for arm in data.values():
    for ds, rows in arm.items():
        if rows:
            maj[ds] = np.median([r["majority_baseline"] for r in rows])

print(f"{'dataset':<9} {'arm':<14} {'n':>3} {'acc':>8} {'AUC':>8} "
      f"{'size':>5} {'majority':>9} {'over maj':>9}")
print("-" * 72)
for ds in ORDER:
    if ds not in maj:
        continue
    for name, _ in ARMS:
        rows = data.get(name, {}).get(ds, [])
        if not rows:
            continue
        acc = np.median([r["test_acc"] for r in rows])
        auc = np.median([r["test_auc"] for r in rows])
        sz = np.median([r["size"] for r in rows])
        print(f"{SHORT[ds]:<9} {name:<14} {len(rows):>3} {acc:>8.4f} "
              f"{auc:>8.4f} {sz:>5.0f} {maj[ds]:>9.3f} {acc - maj[ds]:>+9.4f}")
    if ranksums:
        std = [r["test_acc"] for r in data.get("stdGP", {}).get(ds, [])]
        for name in ("TSGP-transfer", "TSGP-clf"):
            arm = [r["test_acc"] for r in data.get(name, {}).get(ds, [])]
            if arm and std:
                p = ranksums(arm, std).pvalue
                verdict = ("n.s." if p > 0.05 else
                           ("TSGP" if np.median(arm) > np.median(std)
                            else "stdGP"))
                print(f"{'':<9} {name + ' vs stdGP':<14} "
                      f"{'':>3} p={p:<8.4f} -> {verdict}")
        a = [r["test_acc"] for r in data.get("TSGP-clf", {}).get(ds, [])]
        b = [r["test_acc"] for r in data.get("TSGP-transfer", {}).get(ds, [])]
        if a and b:
            p = ranksums(a, b).pvalue
            better = "clf-trained" if np.median(a) > np.median(b) else "transfer"
            print(f"{'':<9} {'clf vs transfer':<14} {'':>3} p={p:<8.4f} "
                  f"-> {better}")
    print()

print("'over maj' is the margin above always predicting the majority class.")
print("An arm at or below zero has found nothing usable.")
print("\nTSGP arms use step control (k=8), so they spend 8x the model")
print("evaluations per generation -- not an equal-budget comparison with stdGP.")
