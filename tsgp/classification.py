"""Extending TSGP from symbolic regression to binary classification.

The method's whole mechanism rests on two things: semantics defined as a
real-valued output vector, and semantic distance defined as the Euclidean
distance between two such vectors. A tree that emits a class label would break
both -- distance would collapse to a Hamming count over labels, which is coarse,
mostly flat, and would strip the k-nearest-neighbour pairing of the signal the
operator was trained on.

So the tree stays real-valued and its output is read as a **decision value**:
the predicted class is sign(f(x)), with labels encoded as -1/+1. Semantics
remain continuous, Euclidean distance keeps its meaning, and the transformer --
which only ever sees token sequences and output vectors -- can be reused
unchanged. That is what makes it possible to test the operator's transfer to a
new task without regenerating the training pool or retraining.

Fitness is the mean logistic loss on the decision value rather than raw
accuracy. Accuracy is piecewise-constant in the tree's parameters, so it gives
selection almost nothing to work with between the few points where a sample
flips side of the boundary; the logistic loss is a smooth surrogate for the same
objective and keeps the search gradient-like. Accuracy and ROC AUC are computed
for reporting only, never for selection.
"""
import random

import numpy as np
from deap import creator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from . import config
from .datasets import fetch_dataset, PMLB_CACHE_DIR
from .primitives import (create_pset, setup_deap, evaluate_semantics_fast)
from .tsgp_search import (TSGPSearchOperator, sample_with_step_control,
                          step_target, _better)

# Choosing benchmarks is harder here than on the regression side, and the
# constraint is worth stating plainly because it limits what this prototype can
# claim. The trained operator's terminal set is fixed at x0..x3, so a problem
# must be expressed in four features. PMLB contains exactly **two** 4-feature
# binary classification sets, and neither is usable as a benchmark:
# analcatdata_lawsuit (264 samples, 73% imbalanced) and analcatdata_aids (50
# samples).
#
# So, as on the regression side -- where the paper itself takes "the first four
# features" -- we take the first four features of wider problems. That is a
# real information loss, not a formality, so the list is ordered by how little
# is discarded, and the majority-class baseline is reported for every result.
# A dataset where TSGP and stdGP both sit at the majority baseline tells us
# nothing about the operator.
# --- the benchmark set, and why it is constructed rather than taken off the
# --- shelf -----------------------------------------------------------------
#
# PMLB contains exactly two 4-feature binary classification sets and neither is
# usable: analcatdata_lawsuit (264 samples at 73% imbalance) and
# analcatdata_aids (50 samples). Widening the operator to 8 features was the
# obvious answer, but the only wide sets available locally have 20 to 168
# features, where taking "the first eight" is arbitrary feature selection --
# GAMETES is constructed so that only two specific attributes carry signal, and
# Hill_Valley and clean are shape and molecular data whose leading columns mean
# nothing on their own. Results from those would sit at the majority baseline
# and say nothing about the operator.
#
# Instead the five regression benchmarks are converted into binary problems by
# splitting the target at its training-set median. This is a construction and
# must be described as one, but it is a good one for this study: the features
# are the same real-world features, every problem is exactly 4-dimensional so
# nothing is discarded, the classes are balanced by construction, and -- most
# usefully -- the classification and regression results are then measured on
# identical data, so the operator's behaviour can be compared across the two
# tasks directly rather than across different benchmarks.
CLF_DATASETS_MEDIAN = [
    "1030_ERA",
    "1027_ESL",
    "690_visualizing_galaxy",
    "1029_LEV",
    "529_pollen",
]

CLF_DATASETS_D4 = [
    "irish",       # 500 samples,  5 features -> drop 1, balanced
    "diabetes",    # 768 samples,  8 features -> first 4, 9% imbalance
    "breast_w",    # 699 samples,  9 features -> first 4, 10% imbalance
]

# At d=8 the point is to stop discarding features, so these are all >= 8 wide.
# diabetes is an exact fit; the rest lose one feature except australian.
# diabetes and breast_w appear in both lists so the d=4 and d=8 operators can
# be compared on common ground.
CLF_DATASETS_D8 = [
    "diabetes",    # 768 samples,  8 features -> exact fit
    "breast_w",    # 699 samples,  9 features -> first 8
    "threeOf9",    # 512 samples,  9 features -> first 8
    "xd6",         # 973 samples,  9 features -> first 8
    "australian",  # 690 samples, 14 features -> first 8
]

N_SOURCE_FEATURES = {
    "irish": 5, "phoneme": 5, "diabetes": 8, "breast_w": 9,
    "threeOf9": 9, "xd6": 9, "australian": 14, "magic": 10,
}


def default_datasets():
    """Benchmark list matching the operator's dimensionality."""
    if config.NUM_FEATURES >= 8:
        return CLF_DATASETS_D8
    return CLF_DATASETS_MEDIAN


# Kept for backwards compatibility with the d=4 prototype runs.
CLF_DATASETS = CLF_DATASETS_D4

DISPLAY_NAMES = {n: n for n in set(CLF_DATASETS_D4) | set(CLF_DATASETS_D8)}


def _balanced_threshold(y):
    """The cut point among observed values giving the most even split.

    With a continuous target this lands on the median. With a discrete ordinal
    one -- which all five of these are -- it picks whichever observed value
    splits closest to 50/50, which the median need not do when a large share of
    the mass sits on a single value.
    """
    candidates = np.unique(y)[:-1]          # a cut above the max splits nothing
    if len(candidates) == 0:
        return float(np.median(y))
    fracs = np.array([np.mean(y > c) for c in candidates])
    return float(candidates[np.argmin(np.abs(fracs - 0.5))])


def load_classification_dataset(name, cache_dir=PMLB_CACHE_DIR, split_seed=42):
    """Fetch, split 50/50 and standardise one PMLB classification benchmark.

    Mirrors the regression protocol: first four features, per-run 50/50 split,
    scaler fitted on the training half only. The target is mapped to -1/+1
    rather than 0/1 so that the margin y*f(x) is the natural quantity and
    ||y_train|| = sqrt(m), which keeps the step-size schedule -- expressed as a
    fraction of ||y_train|| -- on the same footing as in the regression runs.
    """
    X, y = fetch_dataset(name, cache_dir=cache_dir)
    # The operator's terminal set is exactly x0..x{d-1}, so the matrix has to
    # be exactly that wide: truncate wider problems, and zero-pad narrower ones
    # so the unused variables exist but carry no signal. Padding is recorded in
    # N_SOURCE_FEATURES so a result can never hide how much was discarded or
    # invented.
    d = config.NUM_FEATURES
    if X.shape[1] >= d:
        X = X[:, :d]
    else:
        X = np.hstack([X, np.zeros((X.shape[0], d - X.shape[1]))])

    classes = np.unique(y)
    if name in CLF_DATASETS_MEDIAN or len(classes) > 2:
        # Continuous or many-valued target: threshold it into a binary problem.
        #
        # A plain median split is not good enough here. These targets are
        # discrete ordinal ratings with heavy ties at the median, so "> median"
        # can land badly off balance -- on LEV it gives a 21/79 split, whose
        # majority baseline of 0.79 would swamp any signal the search finds.
        # The threshold is instead chosen from the observed values as the one
        # that comes closest to an even split. It is selected on the training
        # half only, so nothing about the test half leaks into the task
        # definition.
        X_train, X_test, y_train_raw, y_test_raw = train_test_split(
            X, y, test_size=0.5, random_state=split_seed)
        thr = _balanced_threshold(y_train_raw)
        y_train = np.where(y_train_raw > thr, 1.0, -1.0)
        y_test = np.where(y_test_raw > thr, 1.0, -1.0)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test, y_train, y_test

    if len(classes) != 2:
        raise ValueError(f"{name} has {len(classes)} classes; this prototype "
                         f"is binary only")
    y = np.where(y == classes[0], -1.0, 1.0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=split_seed, stratify=y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


def logistic_loss(individual, X, y):
    """Mean logistic loss of the decision value. Lower is better.

    log(1 + exp(-y*f)) via logaddexp, which is stable for the very large
    magnitudes protected division can produce.
    """
    try:
        f = evaluate_semantics_fast(individual, X)
        loss = float(np.mean(np.logaddexp(0.0, -y * f)))
        if not np.isfinite(loss):
            return (1e6,)
        return (loss,)
    except Exception:
        return (1e6,)


def accuracy(individual, X, y):
    f = evaluate_semantics_fast(individual, X)
    # sign(0) is 0, which would count as neither class; break the tie to +1.
    pred = np.where(f >= 0, 1.0, -1.0)
    return float(np.mean(pred == y))


def roc_auc(individual, X, y):
    """Rank-based AUC of the decision value; ties get averaged ranks."""
    f = evaluate_semantics_fast(individual, X)
    pos, neg = f[y > 0], f[y < 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    # average ranks within ties so AUC is well defined for constant trees
    vals = np.concatenate([pos, neg])
    _, inv, counts = np.unique(vals, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    r_pos = ranks[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _stats_row(best, X_tr, y_tr, X_te, y_te):
    return {
        "train_loss": float(best.fitness.values[0]),
        "train_acc": accuracy(best, X_tr, y_tr),
        "test_acc": accuracy(best, X_te, y_te),
        "test_auc": roc_auc(best, X_te, y_te),
        "size": int(len(best)),
    }


def _setup(X_train, y_train):
    pset = create_pset()
    toolbox = setup_deap(pset)
    toolbox.register("evaluate", logistic_loss, X=X_train, y=y_train)
    return pset, toolbox


def run_stdgp_classify(X_train, y_train, X_test, y_test,
                       pop_size=None, generations=None, verbose=False):
    pop_size = pop_size or config.GP_POP_SIZE
    generations = generations or config.GP_GENERATIONS
    pset, toolbox = _setup(X_train, y_train)

    pop = toolbox.population(n=pop_size)
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)
    best = min(pop, key=lambda i: i.fitness.values[0])
    best = toolbox.clone(best)

    history = [best.fitness.values[0]]
    for gen in range(1, generations + 1):
        offspring = [toolbox.clone(i) for i in toolbox.select(pop, pop_size)]
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

        champ = min(pop, key=lambda i: i.fitness.values[0])
        if _better(champ, best):
            best = toolbox.clone(champ)
        history.append(best.fitness.values[0])
        if verbose and gen % 10 == 0:
            print(f"  gen {gen:>2}: loss={best.fitness.values[0]:.4f} "
                  f"acc={accuracy(best, X_train, y_train):.3f}", flush=True)

    out = _stats_row(best, X_train, y_train, X_test, y_test)
    out["train_loss_history"] = [float(x) for x in history]
    out["expression"] = str(best)
    return best, out


def run_tsgp_classify(model, X_train, y_train, X_test, y_test,
                      pop_size=None, generations=None, step_k=None,
                      step_anneal=None, frac_start=None, frac_end=None,
                      verbose=False):
    """TSGP search with the classification objective.

    The transformer is used exactly as trained on regression semantics -- this
    is the transfer test. Only the fitness function differs from run_tsgp.
    """
    pop_size = pop_size or config.GP_POP_SIZE
    generations = generations or config.GP_GENERATIONS
    step_k = config.TSGP_STEP_K if step_k is None else step_k
    step_anneal = (config.TSGP_STEP_ANNEAL if step_anneal is None
                   else step_anneal)

    pset, toolbox = _setup(X_train, y_train)
    search_op = TSGPSearchOperator(model, pset)
    if step_k > 1:
        search_op.batch_size = 400

    pop = toolbox.population(n=pop_size)
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)
    best = toolbox.clone(min(pop, key=lambda i: i.fitness.values[0]))

    history = [best.fitness.values[0]]
    for gen in range(1, generations + 1):
        selected = toolbox.select(pop, pop_size)
        target = (step_target(gen, generations, y_train, frac_start, frac_end)
                  if (step_anneal and step_k > 1) else None)
        children = sample_with_step_control(search_op, selected, step_k,
                                            toolbox, X_train, target=target)

        offspring = []
        for parent, tree in zip(selected, children):
            accepted = None
            if tree is not None and len(tree) > 0:
                cand = creator.Individual(tree)
                cand.fitness.values = toolbox.evaluate(cand)
                if cand.fitness.values[0] < 1e6:
                    accepted = cand
            if accepted is None:
                accepted = toolbox.clone(parent)
                accepted.fitness.values = parent.fitness.values
            offspring.append(accepted)
        pop[:] = offspring

        champ = min(pop, key=lambda i: i.fitness.values[0])
        if _better(champ, best):
            best = toolbox.clone(champ)
        history.append(best.fitness.values[0])
        if verbose and gen % 10 == 0:
            print(f"  gen {gen:>2}: loss={best.fitness.values[0]:.4f} "
                  f"acc={accuracy(best, X_train, y_train):.3f}", flush=True)

    out = _stats_row(best, X_train, y_train, X_test, y_test)
    out["train_loss_history"] = [float(x) for x in history]
    out["expression"] = str(best)
    return best, out


def majority_baseline(y_train, y_test):
    """Accuracy of always predicting the training majority class.

    Without this a classification result is uninterpretable: on a set that is
    80% one class, 0.80 accuracy means the search found nothing.
    """
    maj = 1.0 if (y_train > 0).sum() >= (y_train <= 0).sum() else -1.0
    return float(np.mean(y_test == maj))


def find_classification_datasets(max_features=4, binary_only=True):
    """Re-derive CLF_DATASETS from PMLB. Slow: it fetches every candidate."""
    import pmlb
    found = []
    for name in pmlb.classification_dataset_names:
        try:
            df = pmlb.fetch_data(name, local_cache_dir=PMLB_CACHE_DIR)
        except Exception:
            continue
        if df.shape[1] - 1 != max_features:
            continue
        n_cls = df["target"].nunique()
        if binary_only and n_cls != 2:
            continue
        counts = df["target"].value_counts()
        found.append({"name": name, "samples": len(df), "classes": n_cls,
                      "minority_frac": float(counts.min() / len(df))})
    return found
