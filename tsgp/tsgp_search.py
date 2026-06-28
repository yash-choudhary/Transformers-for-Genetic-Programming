import random
import numpy as np
from deap import gp, creator, base, tools

from . import config
from .primitives import create_pset, setup_deap, evaluate_individual
from .tokenizer import (tree_to_tokens, encode, decode, tokens_to_tree,
                        SOS_ID, EOS_ID, PAD_ID, VOCAB_SIZE)
from .syntax_control import SyntaxController, apply_syntax_mask
from .transformer_model import TSGPTransformer, create_model


class TSGPSearchOperator:
    def __init__(self, model, pset, sd_desired=None, temperature=1.0):
        self.model = model
        self.pset = pset
        self.sd_desired = sd_desired or config.TSGP_SD_DESIRED
        self.temperature = temperature

    def sample_offspring(self, parent_individual):
        parent_tokens = tree_to_tokens(parent_individual)
        enc_input = np.array([encode(parent_tokens)], dtype=np.int32)
        sd_input = np.array([self.sd_desired], dtype=np.float32)

        syntax_ctrl = SyntaxController()
        dec_tokens = [SOS_ID]

        for _ in range(config.TRANSFORMER_MAX_SEQ_LEN - 1):
            dec_padded = np.zeros((1, config.TRANSFORMER_MAX_SEQ_LEN),
                                 dtype=np.int32)
            dec_padded[0, :len(dec_tokens)] = dec_tokens

            logits = self.model(
                [enc_input, dec_padded, sd_input],
                training=False
            )

            next_logits = np.array(logits[0, len(dec_tokens) - 1, :])
            next_logits = next_logits / self.temperature
            next_logits = apply_syntax_mask(next_logits, syntax_ctrl)

            probs = np.exp(next_logits - np.max(next_logits))
            probs = probs / (probs.sum() + 1e-10)

            token_id = np.random.choice(VOCAB_SIZE, p=probs)

            if token_id == EOS_ID or syntax_ctrl.is_complete():
                break

            syntax_ctrl.update(token_id)
            dec_tokens.append(token_id)

            if syntax_ctrl.is_complete():
                break

        output_tokens = decode(dec_tokens)
        tree = tokens_to_tree(output_tokens, self.pset)
        return tree


def run_tsgp(model, X_train, y_train, X_test, y_test,
             pop_size=None, generations=None, verbose=True):
    if pop_size is None:
        pop_size = config.GP_POP_SIZE
    if generations is None:
        generations = config.GP_GENERATIONS

    pset = create_pset()
    toolbox = setup_deap(pset)
    toolbox.register("evaluate", evaluate_individual,
                     toolbox=toolbox, X=X_train, y=y_train)

    search_op = TSGPSearchOperator(model, pset)

    pop = toolbox.population(n=pop_size)
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)

    best_train_rmse = min(ind.fitness.values[0] for ind in pop)
    best_ind = min(pop, key=lambda x: x.fitness.values[0])

    stats = {
        "train_rmse": [best_train_rmse],
        "best_size": [len(best_ind)],
    }

    if verbose:
        print(f"Gen 0: Train RMSE={best_train_rmse:.4f}, "
              f"Size={len(best_ind)}")

    for gen in range(1, generations + 1):
        selected = toolbox.select(pop, pop_size)

        offspring = []
        for parent in selected:
            child_tree = search_op.sample_offspring(parent)

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

        best_ind = min(pop, key=lambda x: x.fitness.values[0])
        best_train_rmse = best_ind.fitness.values[0]
        stats["train_rmse"].append(best_train_rmse)
        stats["best_size"].append(len(best_ind))

        if verbose and (gen % 5 == 0 or gen == 1):
            print(f"Gen {gen}: Train RMSE={best_train_rmse:.4f}, "
                  f"Size={len(best_ind)}")

    best_ind = min(pop, key=lambda x: x.fitness.values[0])
    test_rmse = evaluate_individual(best_ind, toolbox, X_test, y_test)[0]
    stats["test_rmse"] = test_rmse

    if verbose:
        print(f"\nBest solution: {best_ind}")
        print(f"Train RMSE: {best_ind.fitness.values[0]:.4f}")
        print(f"Test RMSE: {test_rmse:.4f}")
        print(f"Size: {len(best_ind)}")

    return best_ind, stats


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

    best_train_rmse = min(ind.fitness.values[0] for ind in pop)
    best_ind = min(pop, key=lambda x: x.fitness.values[0])

    stats = {
        "train_rmse": [best_train_rmse],
        "best_size": [len(best_ind)],
    }

    if verbose:
        print(f"Gen 0: Train RMSE={best_train_rmse:.4f}")

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

        best_ind = min(pop, key=lambda x: x.fitness.values[0])
        best_train_rmse = best_ind.fitness.values[0]
        stats["train_rmse"].append(best_train_rmse)
        stats["best_size"].append(len(best_ind))

        if verbose and (gen % 5 == 0 or gen == 1):
            print(f"Gen {gen}: Train RMSE={best_train_rmse:.4f}")

    best_ind = min(pop, key=lambda x: x.fitness.values[0])
    test_rmse = evaluate_individual(best_ind, toolbox, X_test, y_test)[0]
    stats["test_rmse"] = test_rmse

    if verbose:
        print(f"\nBest: Train RMSE={best_ind.fitness.values[0]:.4f}, "
              f"Test RMSE={test_rmse:.4f}, Size={len(best_ind)}")

    return best_ind, stats
