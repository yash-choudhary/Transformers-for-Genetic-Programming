import random
import numpy as np
import tensorflow as tf
from deap import gp, creator, base, tools

from . import config
from .primitives import create_pset, setup_deap, evaluate_individual
from .tokenizer import (tree_to_tokens, encode, decode, tokens_to_tree,
                        SOS_ID, EOS_ID, PAD_ID, VOCAB_SIZE)
from .syntax_control import SyntaxController, apply_syntax_mask
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

            for i in range(n):
                if not active[i]:
                    continue
                # Each row reads its own last real position; the causal mask
                # means trailing PAD cannot influence it.
                pos = len(sequences[i]) - 1
                next_logits = logits[i, pos, :] / self.temperature
                next_logits = apply_syntax_mask(next_logits, controllers[i])

                probs = np.exp(next_logits - np.max(next_logits))
                probs = probs / (probs.sum() + 1e-10)

                token_id = np.random.choice(VOCAB_SIZE, p=probs)

                if token_id == EOS_ID or controllers[i].is_complete():
                    active[i] = False
                    continue

                controllers[i].update(token_id)
                sequences[i].append(token_id)

                if controllers[i].is_complete():
                    active[i] = False

        return [tokens_to_tree(decode(seq), self.pset) for seq in sequences]


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
             pop_size=None, generations=None, temperature=None, verbose=True):
    if pop_size is None:
        pop_size = config.GP_POP_SIZE
    if generations is None:
        generations = config.GP_GENERATIONS

    pset = create_pset()
    toolbox = setup_deap(pset)
    toolbox.register("evaluate", evaluate_individual,
                     toolbox=toolbox, X=X_train, y=y_train)

    search_op = TSGPSearchOperator(model, pset, temperature=temperature)

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
        # independent autoregressive loops.
        child_trees = search_op.sample_offspring_batch(selected)

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
