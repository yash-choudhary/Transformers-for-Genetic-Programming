import operator
import random
import numpy as np
from deap import gp, creator, base, tools

from . import config


def protdiv(left, right):
    if abs(right) < 1e-6:
        return 1.0
    return left / right


def _erc_generator():
    return random.choice(config.ERC_VALUES)


def create_pset(num_features=None):
    if num_features is None:
        num_features = config.NUM_FEATURES
    pset = gp.PrimitiveSet("MAIN", num_features)
    pset.addPrimitive(operator.add, 2, name="add")
    pset.addPrimitive(operator.sub, 2, name="sub")
    pset.addPrimitive(operator.mul, 2, name="mul")
    pset.addPrimitive(protdiv, 2, name="protdiv")
    pset.addEphemeralConstant("ERC", _erc_generator)
    for i in range(num_features):
        pset.renameArguments(**{f"ARG{i}": f"x{i}"})
    return pset


def setup_deap(pset):
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset,
                     min_=config.GP_INIT_DEPTH_MIN,
                     max_=config.GP_INIT_DEPTH_MAX)
    toolbox.register("individual", tools.initIterate, creator.Individual,
                     toolbox.expr)
    toolbox.register("population", tools.initRepeat, list,
                     toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("select", tools.selTournament,
                     tournsize=config.GP_TOURNAMENT_SIZE)
    # Paper Sect. 4.1: subtree crossover with a 10% internal node bias toward
    # selecting terminal nodes. DEAP's termpb is the probability of picking a
    # terminal, so termpb=0.1 gives the 90/10 internal/terminal split.
    toolbox.register("mate", gp.cxOnePointLeafBiased,
                     termpb=config.GP_INTERNAL_NODE_BIAS)
    toolbox.register("expr_mut", gp.genFull,
                     min_=config.GP_MUTATION_DEPTH_MIN,
                     max_=config.GP_MUTATION_DEPTH_MAX)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut,
                     pset=pset)
    toolbox.decorate("mate", gp.staticLimit(
        key=operator.attrgetter("height"),
        max_value=config.GP_MAX_TREE_DEPTH))
    toolbox.decorate("mutate", gp.staticLimit(
        key=operator.attrgetter("height"),
        max_value=config.GP_MAX_TREE_DEPTH))
    return toolbox


def evaluate_individual(individual, toolbox, X, y):
    func = toolbox.compile(expr=individual)
    try:
        y_pred = np.array([func(*row) for row in X])
        y_pred = np.nan_to_num(y_pred, nan=1e6, posinf=1e6, neginf=-1e6)
        y_pred = np.clip(y_pred, -1e6, 1e6)
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))
        if np.isnan(rmse) or np.isinf(rmse):
            rmse = 1e6
    except Exception:
        rmse = 1e6
    return (rmse,)


def evaluate_semantics(individual, toolbox, X):
    func = toolbox.compile(expr=individual)
    try:
        semantics = np.array([func(*row) for row in X], dtype=np.float64)
        semantics = np.nan_to_num(semantics, nan=0.0, posinf=1e6, neginf=-1e6)
        semantics = np.clip(semantics, -1e6, 1e6)
    except Exception:
        semantics = np.full(len(X), 0.0)
    return semantics


def tree_to_prefix(individual):
    return [node.name if isinstance(node, gp.Primitive)
            else node.format() if isinstance(node, gp.Ephemeral)
            else node.value
            for node in individual]


def setup_datagen_toolbox(pset):
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset,
                     min_=config.GP_INIT_DEPTH_MIN,
                     max_=config.GP_INIT_DEPTH_MAX)
    toolbox.register("individual", tools.initIterate, creator.Individual,
                     toolbox.expr)
    toolbox.register("population", tools.initRepeat, list,
                     toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    toolbox.register("select", _double_tournament, fitness_size=5,
                     parsimony_size=1.4, fitness_first=True)

    # Same crossover bias as the search runs -- Sect. 4.1 states the data
    # generation uses the Table 1 GP parameters.
    toolbox.register("mate", gp.cxOnePointLeafBiased,
                     termpb=config.GP_INTERNAL_NODE_BIAS)
    toolbox.register("expr_mut", gp.genFull,
                     min_=config.GP_MUTATION_DEPTH_MIN,
                     max_=config.GP_MUTATION_DEPTH_MAX)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut,
                     pset=pset)
    toolbox.decorate("mate", gp.staticLimit(
        key=operator.attrgetter("height"),
        max_value=config.GP_MAX_TREE_DEPTH))
    toolbox.decorate("mutate", gp.staticLimit(
        key=operator.attrgetter("height"),
        max_value=config.GP_MAX_TREE_DEPTH))
    return toolbox


def _double_tournament(individuals, k, fitness_size, parsimony_size,
                       fitness_first):
    return tools.selDoubleTournament(individuals, k,
                                     fitness_size=fitness_size,
                                     parsimony_size=parsimony_size,
                                     fitness_first=fitness_first)
