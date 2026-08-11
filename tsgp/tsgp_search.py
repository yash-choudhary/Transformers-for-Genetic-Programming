import random
import numpy as np
import tensorflow as tf
from deap import gp, creator, base, tools

from . import config
from .primitives import (create_pset, setup_deap, evaluate_individual,
                         evaluate_semantics, evaluate_semantics_fast)
from .tokenizer import (tree_to_tokens, encode, decode, tokens_to_tree,
                        SOS_ID, EOS_ID, PAD_ID, VOCAB_SIZE)
from .syntax_control import (SyntaxController, apply_syntax_mask,
                             MASK_TABLE, MASK_INDEX)
from .transformer_model import TSGPTransformer, create_model


class TSGPSearchOperator:
    """Transformer variation operator, decoding a whole population at once.

    Decoding is the dominant cost of a TSGP run, and three things make the
    naive one-parent-at-a-time version slow:

      * every decode step re-ran the encoder, even though the parent and the
        desired SD are fixed for the whole sequence — the encoder is hoisted
        out of the loop and run once per batch;
      * every call went through eager dispatch, which on a model this small
        costs roughly 10x the actual arithmetic — the passes are wrapped in
        tf.function;
      * each pass carried a single sequence, so the fixed per-call overhead
        was paid 100 times per generation instead of once — the whole
        population is now decoded in lockstep.

    Sampling semantics are unchanged: each sequence keeps its own
    SyntaxController and is sampled from its own masked softmax, exactly as
    before. Only the order in which the RNG is consumed differs (interleaved
    by step rather than by individual), so results are not bit-identical to
    the sequential implementation at a given seed — the distribution is.
    """

    def __init__(self, model, pset, sd_desired=None, temperature=None,
                 batch_size=None):
        self.model = model
        self.pset = pset
        self.sd_desired = sd_desired or config.TSGP_SD_DESIRED
        self.temperature = (config.TSGP_TEMPERATURE if temperature is None
                            else temperature)
        # None decodes the whole population at once; set an int to cap peak
        # memory on a small GPU.
        self.batch_size = batch_size
        self._encode_fn = None
        self._decode_fn = None

    def _build_fns(self):
        if self._decode_fn is not None:
            return
        model = self.model

        @tf.function(reduce_retracing=True)
        def encode_fn(enc_input, sd_input):
            return model.encode(enc_input, sd_input, training=False)

        @tf.function(reduce_retracing=True)
        def decode_fn(dec_input, enc_output, enc_input, sd_input):
            dec_output = model.decode(dec_input, enc_output, enc_input,
                                      sd_input, training=False)
            return model.final_layer(dec_output)

        self._encode_fn = encode_fn
        self._decode_fn = decode_fn

    def sample_offspring(self, parent_individual):
        """Single-parent convenience wrapper, kept for callers and tests."""
        return self.sample_offspring_batch([parent_individual])[0]

    def sample_offspring_batch(self, parents):
        """Sample one offspring per parent. Returns a list of trees (or None)."""
        if not parents:
            return []
        chunk = self.batch_size or len(parents)
        trees = []
        for start in range(0, len(parents), chunk):
            trees.extend(self._sample_chunk(parents[start:start + chunk]))
        return trees

    def _sample_chunk(self, parents):
        self._build_fns()
        n = len(parents)

        enc_input = np.array([encode(tree_to_tokens(p)) for p in parents],
                             dtype=np.int32)
        sd_input = np.full((n,), self.sd_desired, dtype=np.float32)

        enc_input_tf = tf.constant(enc_input)
        sd_input_tf = tf.constant(sd_input)
        # Fixed for the whole sequence, so this runs once rather than per step.
        enc_output = self._encode_fn(enc_input_tf, sd_input_tf)

        controllers = [SyntaxController() for _ in range(n)]
        sequences = [[SOS_ID] for _ in range(n)]
        active = np.ones(n, dtype=bool)

        for _ in range(config.TRANSFORMER_MAX_SEQ_LEN - 1):
            if not active.any():
                break

            # Only active rows determine how far we have to decode; finished
            # rows are carried along padded and their logits ignored.
            cur_len = max(len(sequences[i]) for i in range(n) if active[i])
            dec = np.full((n, cur_len), PAD_ID, dtype=np.int32)
            for i, seq in enumerate(sequences):
                keep = min(len(seq), cur_len)
                dec[i, :keep] = seq[:keep]

            logits = self._decode_fn(tf.constant(dec), enc_output,
                                     enc_input_tf, sd_input_tf).numpy()

            # All active rows are masked and sampled in one vectorised step.
            # The previous version looped in Python and called
            # np.random.choice once per sequence per token, which dominated
            # runtime -- 80,000 calls per generation once k candidates per
            # parent are drawn. Gumbel-max draws from exactly the same
            # categorical distribution as choice(p=softmax(logits)); only the
            # order in which the RNG is consumed differs, so results are not
            # bit-identical at a fixed seed but the distribution is unchanged.
            idx = np.flatnonzero(active)
            # Each row reads its own last real position; the causal mask means
            # trailing PAD cannot influence it.
            pos = np.fromiter((len(sequences[i]) - 1 for i in idx),
                              dtype=np.intp, count=len(idx))
            step_logits = logits[idx, pos, :] / self.temperature
            kinds = np.fromiter(
                (MASK_INDEX[controllers[i].mask_kind()] for i in idx),
                dtype=np.intp, count=len(idx))
            step_logits = step_logits + MASK_TABLE[kinds]

            gumbel = np.random.gumbel(size=step_logits.shape)
            tokens = np.argmax(step_logits + gumbel, axis=1)

            for slot, i in enumerate(idx):
                token_id = int(tokens[slot])
                if token_id == EOS_ID or controllers[i].is_complete():
                    active[i] = False
                    continue
                controllers[i].update(token_id)
                sequences[i].append(token_id)
                if controllers[i].is_complete():
                    active[i] = False

        return [tokens_to_tree(decode(seq), self.pset) for seq in sequences]


def step_target(gen, generations, y_train, frac_start=None, frac_end=None):
    """Desired parent-offspring semantic distance at this generation.

    Geometric decay between frac_start and frac_end times ||y_train||, chosen
    to imitate the annealing stdGP gets for free from crossover between
    converging parents. Expressed relative to ||y_train|| so the same schedule
    is meaningful on every data set.

    frac_start == frac_end gives a constant target, which is the control that
    separates "annealing helps" from "merely targeting a distance helps".
    """
    a = config.TSGP_STEP_FRAC_START if frac_start is None else frac_start
    b = config.TSGP_STEP_FRAC_END if frac_end is None else frac_end
    norm = float(np.linalg.norm(y_train))
    frac = a * (b / a) ** (gen / max(generations, 1))
    return norm * frac


def sample_with_step_control(search_op, parents, k, toolbox, X_train,
                             target=None):
    """Sample k offspring per parent; keep the semantically nearest one.

    Why this exists. The transformer's single-sample semantic step has a floor:
    driving SD_d from 0.1 down to 0.0001 leaves the achieved parent-offspring
    distance flat at ~0.7 (see tsgp.step_floor), so the paper's operating point
    SD_d = 0.1 sits *below* what one sample can deliver. Meanwhile stdGP's step
    anneals from 28 to 1.0 across a run because crossover between converging
    parents naturally shrinks, which is what gives it an exploitation phase.
    TSGP has no such mechanism and freezes at generation 10.

    Taking the nearest of k draws gets underneath the floor -- measured medians
    0.79 / 0.42 / 0.24 / 0.15 / 0.10 for k = 1 / 2 / 4 / 8 / 16 -- and is the
    step-size control the paper lists as future work in Sect. 5. k = 1 is the
    paper's operator exactly.

    All k*len(parents) sequences are decoded in a single lockstep batch, so the
    cost is close to flat in k rather than k times a single pass. Selection is
    on semantic proximity to the parent only; fitness plays no part, so this
    adds step-size control and not a second layer of selection pressure.
    """
    n = len(parents)
    if k <= 1:
        return search_op.sample_offspring_batch(parents)

    flat = search_op.sample_offspring_batch(list(parents) * k)

    parent_sems = [evaluate_semantics_fast(p, X_train) for p in parents]
    chosen = [None] * n
    best_score = [float("inf")] * n
    for j, tree in enumerate(flat):
        i = j % n
        if tree is None or len(tree) == 0:
            continue
        sem = evaluate_semantics_fast(tree, X_train)
        if not np.all(np.isfinite(sem)):
            continue
        d = float(np.linalg.norm(sem - parent_sems[i]))
        if not np.isfinite(d):
            continue
        # target=None reproduces the greedy "nearest offspring" rule; with a
        # target the step is steered towards a size rather than minimised,
        # which is what keeps early generations exploratory.
        score = d if target is None else abs(d - target)
        if score < best_score[i]:
            best_score[i], chosen[i] = score, tree
    return chosen


def _better(candidate, incumbent):
    """True if candidate has strictly lower training RMSE than incumbent."""
    return incumbent is None or (candidate.fitness.values[0]
                                 < incumbent.fitness.values[0])


def _update_best_ever(pop, best_ever, toolbox):
    """Archive the best training solution seen so far.

    Paper Sect. 4.2: "The best performing solution on the training set is used
    to compute the RMSE on the test set" -- the best found at any point in the
    run, not whatever survives to the final population. Replacement stays
    purely generational (Table 1 specifies no elitism), so this archive is
    reporting only: it never re-enters the population and cannot influence
    selection or variation.

    The individual is cloned because DEAP's variation operators modify
    individuals in place.
    """
    champion = min(pop, key=lambda x: x.fitness.values[0])
    if _better(champion, best_ever):
        return toolbox.clone(champion)
    return best_ever


def run_tsgp(model, X_train, y_train, X_test, y_test,
             pop_size=None, generations=None, temperature=None, step_k=None,
             step_anneal=None, frac_start=None, frac_end=None, verbose=True):
    if pop_size is None:
        pop_size = config.GP_POP_SIZE
    if generations is None:
        generations = config.GP_GENERATIONS
    if step_k is None:
        step_k = config.TSGP_STEP_K
    if step_anneal is None:
        step_anneal = config.TSGP_STEP_ANNEAL

    pset = create_pset()
    toolbox = setup_deap(pset)
    toolbox.register("evaluate", evaluate_individual,
                     toolbox=toolbox, X=X_train, y=y_train)

    search_op = TSGPSearchOperator(model, pset, temperature=temperature)
    # Cap the decode batch so k * pop_size sequences cannot exhaust an 8GB card.
    # Measured at 800: 8000MiB of 8192 on the RTX 2070, which is too close to
    # the edge for a 150-unit grid to survive. 400 halves peak memory and the
    # decode is GPU-bound either way, so the throughput cost is small.
    if step_k > 1:
        search_op.batch_size = 400

    pop = toolbox.population(n=pop_size)
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)

    best_ever = _update_best_ever(pop, None, toolbox)

    # "train_rmse"/"best_size" track the archived best-so-far (what the paper
    # plots and reports); the "_pop" series track the current population, which
    # is what actually drives the search.
    stats = {
        "train_rmse": [best_ever.fitness.values[0]],
        "best_size": [len(best_ever)],
        "train_rmse_pop": [best_ever.fitness.values[0]],
        "best_size_pop": [len(best_ever)],
    }

    if verbose:
        print(f"Gen 0: Train RMSE={best_ever.fitness.values[0]:.4f}, "
              f"Size={len(best_ever)}")

    for gen in range(1, generations + 1):
        selected = toolbox.select(pop, pop_size)

        # One batched decode for the whole population, rather than pop_size
        # independent autoregressive loops. step_k = 1 is the paper's operator.
        target = (step_target(gen, generations, y_train, frac_start, frac_end)
                  if (step_anneal and step_k > 1) else None)
        child_trees = sample_with_step_control(search_op, selected, step_k,
                                               toolbox, X_train, target=target)

        offspring = []
        for parent, child_tree in zip(selected, child_trees):
            if child_tree is not None and len(child_tree) > 0:
                child = creator.Individual(child_tree)
                try:
                    child.fitness.values = toolbox.evaluate(child)
                    if child.fitness.values[0] < 1e6:
                        offspring.append(child)
                        continue
                except Exception:
                    pass

            clone = toolbox.clone(parent)
            clone.fitness.values = parent.fitness.values
            offspring.append(clone)

        pop[:] = offspring

        pop_best = min(pop, key=lambda x: x.fitness.values[0])
        best_ever = _update_best_ever(pop, best_ever, toolbox)

        stats["train_rmse"].append(best_ever.fitness.values[0])
        stats["best_size"].append(len(best_ever))
        stats["train_rmse_pop"].append(pop_best.fitness.values[0])
        stats["best_size_pop"].append(len(pop_best))

        if verbose and (gen % 5 == 0 or gen == 1):
            print(f"Gen {gen}: Train RMSE={best_ever.fitness.values[0]:.4f} "
                  f"(pop {pop_best.fitness.values[0]:.4f}), "
                  f"Size={len(best_ever)}")

    test_rmse = evaluate_individual(best_ever, toolbox, X_test, y_test)[0]
    stats["test_rmse"] = test_rmse

    if verbose:
        print(f"\nBest solution: {best_ever}")
        print(f"Train RMSE: {best_ever.fitness.values[0]:.4f}")
        print(f"Test RMSE: {test_rmse:.4f}")
        print(f"Size: {len(best_ever)}")

    return best_ever, stats


def run_stdgp_baseline(X_train, y_train, X_test, y_test,
                       pop_size=None, generations=None, verbose=True):
    if pop_size is None:
        pop_size = config.GP_POP_SIZE
    if generations is None:
        generations = config.GP_GENERATIONS

    pset = create_pset()
    toolbox = setup_deap(pset)
    toolbox.register("evaluate", evaluate_individual,
                     toolbox=toolbox, X=X_train, y=y_train)

    pop = toolbox.population(n=pop_size)
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)

    best_ever = _update_best_ever(pop, None, toolbox)

    stats = {
        "train_rmse": [best_ever.fitness.values[0]],
        "best_size": [len(best_ever)],
        "train_rmse_pop": [best_ever.fitness.values[0]],
        "best_size_pop": [len(best_ever)],
    }

    if verbose:
        print(f"Gen 0: Train RMSE={best_ever.fitness.values[0]:.4f}")

    for gen in range(1, generations + 1):
        selected = toolbox.select(pop, pop_size)
        offspring = [toolbox.clone(ind) for ind in selected]

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

        pop_best = min(pop, key=lambda x: x.fitness.values[0])
        best_ever = _update_best_ever(pop, best_ever, toolbox)

        stats["train_rmse"].append(best_ever.fitness.values[0])
        stats["best_size"].append(len(best_ever))
        stats["train_rmse_pop"].append(pop_best.fitness.values[0])
        stats["best_size_pop"].append(len(pop_best))

        if verbose and (gen % 5 == 0 or gen == 1):
            print(f"Gen {gen}: Train RMSE={best_ever.fitness.values[0]:.4f} "
                  f"(pop {pop_best.fitness.values[0]:.4f})")

    test_rmse = evaluate_individual(best_ever, toolbox, X_test, y_test)[0]
    stats["test_rmse"] = test_rmse

    if verbose:
        print(f"\nBest: Train RMSE={best_ever.fitness.values[0]:.4f}, "
              f"Test RMSE={test_rmse:.4f}, Size={len(best_ever)}")

    return best_ever, stats
