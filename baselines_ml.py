"""Standard ML baselines on the same classification benchmarks.

Comparing TSGP only against standard GP leaves the obvious question unanswered:
is a compact symbolic model actually *good*, or would an off-the-shelf
classifier match it with less effort? Interpretability is the whole argument for
symbolic regression, so the honest comparison includes a model that is also
readable (a shallow decision tree, a logistic regression) and one that is not
but sets the accuracy ceiling (a random forest).

Same splits, same seeds and same thresholding as the GP runs, so the numbers sit
directly alongside them.
"""
import json
import os

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier

from tsgp.classification import (CLF_DATASETS_MEDIAN,
                                 load_classification_dataset)
from tsgp.experiment_units import unit_seed

SHORT = {"1030_ERA": "ERA", "1027_ESL": "ESL",
         "690_visualizing_galaxy": "Galaxy", "1029_LEV": "LEV",
         "529_pollen": "pollen"}


def models(seed):
    return {
        # size = number of decision nodes, comparable in spirit to GP nodes
        "tree-d3": DecisionTreeClassifier(max_depth=3, random_state=seed),
        "tree-full": DecisionTreeClassifier(random_state=seed),
        "logreg": LogisticRegression(max_iter=2000),
        "rf-100": RandomForestClassifier(n_estimators=100, random_state=seed,
                                         n_jobs=-1),
        "majority": DummyClassifier(strategy="most_frequent"),
    }


def size_of(name, model):
    if name.startswith("tree"):
        return int(model.tree_.node_count)
    if name == "logreg":
        return int(np.sum(np.abs(model.coef_) > 1e-9)) + 1
    if name == "rf-100":
        return int(sum(t.tree_.node_count for t in model.estimators_))
    return 0


def main(runs=30, out="results_clf_ml"):
    os.makedirs(out, exist_ok=True)
    rows = []
    for ds in CLF_DATASETS_MEDIAN:
        for run in range(runs):
            seed = unit_seed(ds, "ml", run)
            Xtr, Xte, ytr, yte = load_classification_dataset(
                ds, split_seed=seed)
            for name, m in models(seed).items():
                m.fit(Xtr, ytr)
                pred = m.predict(Xte)
                try:
                    score = (m.predict_proba(Xte)[:, 1]
                             if hasattr(m, "predict_proba")
                             else m.decision_function(Xte))
                    auc = roc_auc_score(yte, score)
                except Exception:
                    auc = float("nan")
                rows.append({"dataset": ds, "model": name, "run": run,
                             "acc": float(accuracy_score(yte, pred)),
                             "auc": float(auc), "size": size_of(name, m)})
    with open(os.path.join(out, "ml_baselines.json"), "w") as f:
        json.dump(rows, f)

    print(f"{'dataset':<9} {'model':<10} {'acc':>8} {'AUC':>8} {'size':>7}")
    print("-" * 46)
    for ds in CLF_DATASETS_MEDIAN:
        for name in models(0):
            sel = [r for r in rows if r["dataset"] == ds and r["model"] == name]
            print(f"{SHORT[ds]:<9} {name:<10} "
                  f"{np.median([r['acc'] for r in sel]):>8.4f} "
                  f"{np.nanmedian([r['auc'] for r in sel]):>8.4f} "
                  f"{np.median([r['size'] for r in sel]):>7.0f}")
        print()
    return rows


if __name__ == "__main__":
    main()
