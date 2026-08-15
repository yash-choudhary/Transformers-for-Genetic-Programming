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

import argparse

_p = argparse.ArgumentParser(description=__doc__)
_p.add_argument("--stdgp", default="results_clf")
_p.add_argument("--tsgp", default="results_clf_new")
_p.add_argument("--transfer", default="results_clf_transfer")
_p.add_argument("--k1", default="results_clf_k1",
                help="Equal-budget k=1 arm, included when present.")
_args = _p.parse_args()

ARMS = [("stdGP", _args.stdgp),
        ("TSGP-transfer", _args.transfer),
        ("TSGP-k1", _args.k1),
        ("TSGP-clf", _args.tsgp)]

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
data = {k: v for k, v in data.items() if v}

# Refuse to compare arms built from different label constructions. The
# median-split and middle-band tasks share dataset names but are different
# problems, and mixing them produces a large, entirely spurious difference --
# which is exactly what happened the first time this script was pointed at the
# middle-band directories while still defaulting to the median transfer arm.
def _task_of(rows):
    tasks = {r.get("clf_task") for arm in rows.values() for r in arm}
    tasks.discard(None)
    return tasks


_seen = {name: _task_of(arm) for name, arm in data.items()}
_all = set().union(*_seen.values()) if _seen else set()
if len(_all) > 1:
    print(f"REFUSING TO SUMMARISE: arms come from different label "
          f"constructions {sorted(_all)}.")
    for name, t in _seen.items():
        print(f"  {name:<15} {sorted(t) or ['(unrecorded)']}")
    print("\nPass --stdgp/--tsgp/--transfer/--k1 pointing at one task's "
          "directories.")
    raise SystemExit(2)

# Majority baselines differ between the two constructions, so a mismatch shows
# up there too even for older files written before clf_task was recorded.
_maj_by_arm = {}
for name, arm in data.items():
    for ds, rows in arm.items():
        if rows:
            _maj_by_arm.setdefault(ds, {})[name] = np.median(
                [r["majority_baseline"] for r in rows])
for ds, per_arm in _maj_by_arm.items():
    if len(per_arm) > 1 and (max(per_arm.values()) - min(per_arm.values())) > 0.05:
        print(f"WARNING: majority baseline for {ds} differs across arms "
              f"({ {k: round(v, 3) for k, v in per_arm.items()} }) -- these "
              f"are probably different label constructions.")
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
        for name in ("TSGP-transfer", "TSGP-k1", "TSGP-clf"):
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
