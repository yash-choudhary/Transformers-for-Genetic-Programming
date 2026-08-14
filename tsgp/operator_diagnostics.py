"""Does the trained transformer behave like a semantic variation operator?

The v1-v6 study judged the operator by token-level accuracy and by offspring
size, and concluded from those that it was broken. Both are poor instruments:

  * Token accuracy is bounded well below 1 by the task itself. Semantic
    k-nearest-neighbour pairing is one-to-many -- at near-zero SD the median
    pair still differs in 22% of its tokens and there are zero identical pairs
    in five million -- so a model that emits a *different* valid neighbour is
    penalised exactly like one that emits nonsense.
  * Offspring size is a critical branching process. In prefix notation a
    sampled tree has mean size 1/(1-2p) in the operator probability p, so a
    one-point error in p halves the tree. Size moves violently for reasons
    unrelated to semantic quality.

What actually matters is whether an offspring lands near *its own* parent, and
whether the requested semantic distance steers how near. Both need a control
condition -- distance to an unrelated parent -- to mean anything, and that
control is what the earlier diagnostics lacked.

Three checks, each with an explicit gate:

  LOCALITY   d(offspring, own parent) / d(offspring, unrelated parent).
             1.0 means the parent was ignored. Lower is better.
  RESPONSE   achieved distance as SD_d is swept. Must be monotonic and span a
             useful range, or step size cannot be controlled during search.
  REGIME     locality as a function of the parent's semantic norm. The GP
             search starts on Ramped Half-and-Half functions whose norms are
             10-100x the training pool's, so an operator that only works on
             pool-like parents will not work for the first generations -- and
             with fully generational replacement it may never recover.
"""
import json
import os

import numpy as np
import tensorflow as tf

from . import config
from .primitives import create_pset, setup_deap, evaluate_semantics
from .syntax_control import SyntaxController, apply_syntax_mask
from .tokenizer import (encode, decode, tree_to_tokens, tokens_to_tree,
                        SOS_ID, EOS_ID, PAD_ID, VOCAB_SIZE)

# Reference values measured on the training set the model was fitted to. These
# are NOT universal: the regression pool sits at SD median 0.164, the
# classification pool at 6.637, because SD is an absolute distance and
# classification functions have far larger semantic norms. Reading an achieved
# distance against the wrong reference makes a healthy operator look broken, so
# pass the right pool's values when diagnosing a classification checkpoint.
TRAIN_PAIR_SD_MEDIAN = 0.164
TRAIN_TARGET_SIZE_MEDIAN = 35
CLF_PAIR_SD_MEDIAN = 6.637
CLF_TARGET_SIZE_MEDIAN = 29

GATES = {
    "locality_pool": 0.30,      # offspring must be clearly nearer its own parent
    "response_span": 3.0,       # achieved distance must move at least this much
    "locality_rhh": 0.60,       # must also hold on start-of-search functions
}


class Sampler:
    """Batched ancestral sampling under the syntax mask."""

    def __init__(self, model, pset, rng):
        self.model = model
        self.pset = pset
        self.rng = rng

        @tf.function(reduce_retracing=True)
        def enc_fn(e, s):
            return model.encode(e, s, training=False)

        @tf.function(reduce_retracing=True)
        def dec_fn(d, eo, e, s):
            return model.final_layer(model.decode(d, eo, e, s, training=False))

        self._enc, self._dec = enc_fn, dec_fn

    def __call__(self, parent_tokens, sd_value, temperature=1.0):
        n = len(parent_tokens)
        e_tf = tf.constant(np.array([encode(p) for p in parent_tokens],
                                    dtype=np.int32))
        sd_in = tf.constant(np.full((n,), sd_value, dtype=np.float32))
        eo = self._enc(e_tf, sd_in)

        ctrls = [SyntaxController() for _ in range(n)]
        seqs = [[SOS_ID] for _ in range(n)]
        active = np.ones(n, dtype=bool)
        for _ in range(config.TRANSFORMER_MAX_SEQ_LEN - 1):
            if not active.any():
                break
            cur = max(len(seqs[i]) for i in range(n) if active[i])
            dec = np.full((n, cur), PAD_ID, dtype=np.int32)
            for i, s in enumerate(seqs):
                k = min(len(s), cur)
                dec[i, :k] = s[:k]
            logits = self._dec(tf.constant(dec), eo, e_tf, sd_in).numpy()
            for i in range(n):
                if not active[i]:
                    continue
                lg = logits[i, len(seqs[i]) - 1, :] / temperature
                lg = apply_syntax_mask(lg, ctrls[i])
                pr = np.exp(lg - lg.max())
                pr /= pr.sum() + 1e-10
                tok = self.rng.choice(VOCAB_SIZE, p=pr)
                if tok == EOS_ID or ctrls[i].is_complete():
                    active[i] = False
                    continue
                ctrls[i].update(tok)
                seqs[i].append(tok)
                if ctrls[i].is_complete():
                    active[i] = False
        return [decode(s) for s in seqs]


class SemanticSpace:
    """Semantics on one fixed set of random standardised inputs, as Sect. 3.1."""

    def __init__(self, seed=1):
        self.pset = create_pset()
        self.toolbox = setup_deap(self.pset)
        rng = np.random.default_rng(seed)
        self.X = rng.standard_normal((config.NUM_SEMANTIC_SAMPLES,
                                      config.NUM_FEATURES))

    def of(self, tokens):
        tree = tokens_to_tree(tokens, self.pset) if isinstance(tokens, list) \
            else tokens
        if tree is None:
            return None
        s = evaluate_semantics(tree, self.toolbox, self.X)
        if not np.all(np.isfinite(s)) or np.linalg.norm(s) < 1e-9:
            return None
        return s


def _locality(parents, space, sampler, sd_value, rng):
    """Returns (ratio, median own-parent distance, offspring sizes, n)."""
    sems, keep = [], []
    for i, p in enumerate(parents):
        s = space.of(p)
        if s is not None:
            sems.append(s)
            keep.append(i)
    if len(keep) < 20:
        return None
    parents = [parents[i] for i in keep]
    sems = np.array(sems)

    offs = sampler(parents, sd_value)
    perm = rng.permutation(len(parents))
    d_own, d_other, sizes = [], [], []
    for i, o in enumerate(offs):
        so = space.of(o)
        if so is None:
            continue
        d_own.append(np.linalg.norm(so - sems[i]))
        d_other.append(np.linalg.norm(so - sems[perm[i]]))
        sizes.append(len(o))
    if len(d_own) < 20:
        return None
    m_own, m_other = float(np.median(d_own)), float(np.median(d_other))
    return {
        "ratio": m_own / m_other if m_other > 0 else float("nan"),
        "d_own": m_own,
        "d_other": m_other,
        "offspring_size": float(np.median(sizes)),
        "parent_norm": float(np.median(np.linalg.norm(sems, axis=1))),
        "n": len(d_own),
    }


def rhh_parents(space, n_per_bin=100, bins=((3, 8), (9, 14), (15, 22),
                                            (23, 34), (35, 60))):
    """Parents drawn exactly as the GP search initialises its population."""
    bank = {b: [] for b in bins}
    for _ in range(60):
        if all(len(v) >= n_per_bin for v in bank.values()):
            break
        for ind in space.toolbox.population(n=300):
            size = len(ind)
            for b in bins:
                if b[0] <= size <= b[1] and len(bank[b]) < n_per_bin:
                    bank[b].append(tree_to_tokens(ind))
                    break
    return {b: v for b, v in bank.items() if len(v) >= 20}


def run(model, pool_parents, out_path=None, seed=3, verbose=True):
    rng = np.random.default_rng(seed)
    space = SemanticSpace()
    sampler = Sampler(model, space.pset, rng)
    report = {"sd_encoding": getattr(model, "sd_encoding", "?")}

    if verbose:
        print(f"SD encoding: {report['sd_encoding']}")
        print(f"\n--- LOCALITY + RESPONSE (pool parents, n={len(pool_parents)}) ---")
        print(f"{'SD_d':>8} {'d(own)':>9} {'d(other)':>9} {'ratio':>8} "
              f"{'off size':>9}")

    sweep = {}
    for sd_v in [0.01, 0.1, 1.0, 10.0, 50.0]:
        r = _locality(pool_parents, space, sampler, sd_v, rng)
        if r is None:
            continue
        sweep[sd_v] = r
        if verbose:
            print(f"{sd_v:>8.2f} {r['d_own']:>9.3f} {r['d_other']:>9.3f} "
                  f"{r['ratio']:>8.3f} {r['offspring_size']:>9.0f}")
    report["sweep"] = {str(k): v for k, v in sweep.items()}

    if verbose:
        print(f"\n  training pairs sit at SD median {TRAIN_PAIR_SD_MEDIAN}, "
              f"target size median {TRAIN_TARGET_SIZE_MEDIAN}")
        print(f"\n--- REGIME (Ramped Half-and-Half parents, as at gen 0) ---")
        print(f"{'size bin':>10} {'|s(parent)|':>12} {'d(own)':>10} "
              f"{'d(other)':>10} {'ratio':>8} {'n':>5}")

    bank = rhh_parents(space)
    regime = {}
    for b, ps in sorted(bank.items()):
        r = _locality(ps, space, sampler, config.TSGP_SD_DESIRED, rng)
        if r is None:
            continue
        regime[f"{b[0]}-{b[1]}"] = r
        if verbose:
            print(f"{f'{b[0]}-{b[1]}':>10} {r['parent_norm']:>12.2f} "
                  f"{r['d_own']:>10.3f} {r['d_other']:>10.3f} "
                  f"{r['ratio']:>8.3f} {r['n']:>5}")
    report["regime"] = regime

    # ---- gates -----------------------------------------------------------
    loc_pool = sweep.get(0.1, {}).get("ratio", float("nan"))
    lo = sweep.get(0.01, sweep.get(0.1, {})).get("d_own", float("nan"))
    hi = sweep.get(50.0, sweep.get(10.0, {})).get("d_own", float("nan"))
    span = hi / lo if lo and np.isfinite(lo) and lo > 0 else float("nan")
    loc_rhh = float(np.median([r["ratio"] for r in regime.values()])) \
        if regime else float("nan")

    checks = [
        ("LOCALITY  pool parents, ratio at SD_d=0.1", loc_pool,
         GATES["locality_pool"], "<="),
        ("RESPONSE  achieved distance span over SD_d", span,
         GATES["response_span"], ">="),
        ("REGIME    median ratio on RHH parents", loc_rhh,
         GATES["locality_rhh"], "<="),
    ]
    report["gates"] = {}
    if verbose:
        print("\n--- GATES ---")
    all_pass = True
    for label, value, gate, op in checks:
        ok = bool(np.isfinite(value)) and bool(
            (value <= gate) if op == "<=" else (value >= gate))
        all_pass = all_pass and ok
        report["gates"][label.split()[0].lower()] = {
            "value": float(value), "gate": gate, "pass": ok}
        if verbose:
            print(f"  [{'PASS' if ok else 'FAIL'}] {label:<44} "
                  f"{value:8.3f}  (need {op} {gate})")
    report["all_pass"] = bool(all_pass)
    report["notes"] = (
        "REGIME is the check the v1-v6 study never ran. The operator can score "
        "an excellent locality ratio on pool parents and still be near-random "
        "on the functions Ramped Half-and-Half produces at generation 0, whose "
        "semantic norms are 10-100x the pool's. With fully generational "
        "replacement, a search that starts there may never reach the regime "
        "where the operator works."
    )
    if verbose:
        print(f"\n  overall: {'PASS' if all_pass else 'FAIL'}")

    if out_path:
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
    return report


def main():
    import argparse
    import pickle

    parser = argparse.ArgumentParser(
        description="Gate a trained TSGP operator before spending a grid on it.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--pairs",
                        default="data/training/training_pairs_portable.pkl")
    parser.add_argument("--parents-json", default=None,
                        help="A JSON list of token lists to use as pool "
                             "parents. Avoids loading the 1.2GB pair file, "
                             "which matters when a training run is holding "
                             "most of RAM. Written by --dump-parents.")
    parser.add_argument("--dump-parents", default=None,
                        help="Sample --n-parents pool parents out of --pairs "
                             "into this JSON and exit.")
    parser.add_argument("--n-parents", type=int, default=256)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.dump_parents:
        with open(args.pairs, "rb") as f:
            data = pickle.load(f)
        rng = np.random.default_rng(0)
        idx = rng.choice(len(data), args.n_parents, replace=False)
        with open(args.dump_parents, "w") as f:
            json.dump([data[i]["input_tokens"] for i in idx], f)
        print(f"wrote {args.n_parents} parents to {args.dump_parents}")
        return

    from .experiment_units import load_model

    model = load_model(args.weights)
    print(f"checkpoint: {args.weights}")

    if args.parents_json:
        with open(args.parents_json) as f:
            pool_parents = json.load(f)
    else:
        with open(args.pairs, "rb") as f:
            data = pickle.load(f)
        rng = np.random.default_rng(0)
        idx = rng.choice(len(data), args.n_parents, replace=False)
        pool_parents = [data[i]["input_tokens"] for i in idx]
        del data

    out = args.out or (os.path.splitext(args.weights)[0] + "_diagnostics.json")
    run(model, pool_parents, out_path=out)
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
