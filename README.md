# TSGP — Transformer Semantic Genetic Programming

An implementation of **Transformer Semantic Genetic Programming (TSGP)** for symbolic regression, as described in:

> Anthes, P., Sobania, D., & Rothlauf, F. (2025). *Transformer Semantic Genetic Programming for Symbolic Regression.* arXiv:2501.18479

TSGP replaces the standard syntactic variation operators of Genetic Programming (crossover, mutation) with a single **transformer model** that has learned, from millions of synthetic examples, what it means for two mathematical expressions to be *semantically similar* (i.e. to produce similar outputs). Once trained, the transformer acts as a drop-in variation operator inside a normal GP evolutionary loop: given a parent expression tree, it generates an offspring tree that behaves similarly to the parent but is not constrained to be built the same way (unlike e.g. geometric semantic GP, which is forced to use linear combinations and therefore bloats).

This README documents the full pipeline end‑to‑end: the data structures, how each module connects to the others, the math/algorithms behind each step, and how to actually run the system.

---

## Table of Contents

1. [High-Level Pipeline](#high-level-pipeline)
2. [Repository Layout](#repository-layout)
3. [Core Concepts](#core-concepts)
4. [Module-by-Module Walkthrough](#module-by-module-walkthrough)
   - [config.py](#configpy)
   - [primitives.py](#primitivespy)
   - [tokenizer.py](#tokenizerpy)
   - [data_generator.py](#data_generatorpy)
   - [transformer_model.py](#transformer_modelpy)
   - [syntax_control.py](#syntax_controlpy)
   - [train_transformer.py](#train_transformerpy)
   - [tsgp_search.py](#tsgp_searchpy)
   - [run_experiments.py](#run_experimentspy)
5. [End-to-End Data Flow](#end-to-end-data-flow)
6. [How to Run Everything](#how-to-run-everything)
7. [Apple Silicon GPU Setup](#apple-silicon-gpu-setup)
8. [Design Decisions and Gotchas](#design-decisions-and-gotchas)

---

## High-Level Pipeline

TSGP has three phases that map directly onto the three things you asked to implement:

```
┌──────────────────────────┐     ┌──────────────────────────┐     ┌──────────────────────────┐
│  1. SEMANTIC PAIR         │     │  2. TRANSFORMER MODEL     │     │  3. TSGP SEARCH           │
│     GENERATOR              │ ──▶ │     (trained once)        │ ──▶ │     (DEAP GP loop)        │
│  data_generator.py         │     │  transformer_model.py     │     │  tsgp_search.py            │
│                            │     │  train_transformer.py     │     │  run_experiments.py        │
└──────────────────────────┘     └──────────────────────────┘     └──────────────────────────┘
   stdGP on 50 synthetic           Encoder-decoder transformer       Transformer replaces
   regression problems  ──▶        learns: given f_i + desired       crossover/mutation;
   collect ~millions of            semantic distance SD, predict     evolves a population
   unique expression trees ──▶     a semantically-close f_o,         on a real (PMLB)
   k-NN search in semantic         token by token                    black-box dataset
   space ──▶ 5M (f_i, f_o, SD)
   training pairs
```

The key insight that makes this work: **semantics of a function are approximated by its output vector** on a fixed set of random inputs. Two functions are "semantically similar" if those output vectors are close in Euclidean space — *regardless of how differently the underlying expression trees are structured*. The transformer is trained to learn this mapping (tree → tree, conditioned on desired output similarity) so that at inference time it can generate novel offspring without being restricted to a fixed set of structural operators.

---

## Repository Layout

```
Code/
├── tsgp/
│   ├── __init__.py            # empty (package marker)
│   ├── config.py               # all hyperparameters in one place
│   ├── primitives.py           # GP primitive set, DEAP toolbox setup, fitness/semantics eval
│   ├── tokenizer.py            # vocabulary + tree↔token-sequence conversion
│   ├── data_generator.py       # Part 1: synthetic data → stdGP → k-NN → training pairs
│   ├── transformer_model.py    # Part 2: Keras/TF encoder-decoder transformer
│   ├── syntax_control.py       # grammar-constrained decoding (valid-tree guarantee)
│   ├── train_transformer.py    # training loop, checkpointing, resume support
│   ├── tsgp_search.py          # Part 3: TSGP variation operator + GP evolutionary loop
│   └── run_experiments.py      # benchmark runner across PMLB datasets
├── data/                       # generated data (gitignored)
│   └── training/
│       ├── training_pairs.pkl
│       └── training_pairs.csv
├── checkpoints/                # model weights per epoch (gitignored)
├── results/                    # experiment_results.json
├── requirements.txt
└── README.md                   # this file
```

> The Colab notebook (`train_colab.ipynb`) is intentionally **not** covered here — it's a self-contained, copy-pasted version of the training code for running on a faster GPU and isn't part of the shipped pipeline.

---

## Core Concepts

Before diving into code, three ideas recur throughout the codebase:

### 1. Expression trees are DEAP `PrimitiveTree` objects

A symbolic regression solution like `x0 + (x1 * 0.3)` is represented internally as a [DEAP](https://deap.readthedocs.io/) `gp.PrimitiveTree` — a flat list of nodes that represents a tree in **prefix (pre-order) notation**:

```
add(x0, mul(x1, 0.3))   ⟶   [add, x0, mul, x1, 0.3]
```

This prefix-list representation is exactly what allows trees to be linearized into a token sequence for the transformer, and reconstructed back into a tree afterward.

### 2. Semantics = output vector

The "semantics" `s(f)` of a function `f` is *not* its structure — it's the vector of outputs it produces on a fixed set of standardized random inputs:

```
s(f) = [f(x⁽¹⁾), f(x⁽²⁾), ..., f(x⁽ᵐ⁾)]   where x⁽ⁱ⁾ ~ N(0, 1)
```

Two functions are semantically similar if `‖s(fᵢ) - s(fⱼ)‖₂` (their **semantic distance**, SD) is small. This is the GSGP definition of semantics (Moraglio et al., 2012) — see `primitives.evaluate_semantics`.

### 3. The transformer is conditioned on a *desired* semantic distance

Unlike a plain seq2seq model, the encoder and decoder both receive an extra scalar input: the semantic distance `SD` between the input function and the *target* output function. This lets the trained model be steered at inference time — "give me an offspring that's about this semantically far from the parent" — which is what lets TSGP control exploration step size (`SD_d = 0.1` in the paper).

---

## Module-by-Module Walkthrough

### `config.py`

A single source of truth for every hyperparameter in the system, taken directly from the paper's Table 1 and Section 4.1. Nothing here is computed — it's all constants, so changing an experiment setting means changing one line in this file.

```python
NUM_FEATURES = 4                       # d=4 dimensional problems (paper's PMLB selection)

PRIMITIVE_SET_FUNCTIONS = ["add", "sub", "mul", "protdiv"]
TERMINAL_VARIABLES = [f"x{i}" for i in range(NUM_FEATURES)]
ERC_MIN, ERC_MAX, ERC_STEP = -0.5, 0.5, 0.1     # ephemeral random constants

GP_POP_SIZE = 100                      # TSGP/stdGP search population
GP_GENERATIONS = 50
GP_TOURNAMENT_SIZE = 5
GP_MAX_TREE_DEPTH = 17
NUM_RUNS = 30                          # independent runs per benchmark dataset

NUM_SYNTHETIC_PROBLEMS = 50            # data-generation phase
DATAGEN_GP_POP_SIZE = 2000
DATAGEN_GP_GENERATIONS = 100
KNN_K = 3                              # k-nearest-neighbors for semantic pairing
SD_MAX_THRESHOLD = 100.0
TARGET_NUM_PAIRS = 5_000_000

TRANSFORMER_NUM_HEADS = 8
TRANSFORMER_HIDDEN_DIM = 128
TRANSFORMER_NUM_ENCODER_LAYERS = 2
TRANSFORMER_NUM_DECODER_LAYERS = 2
TRANSFORMER_MAX_SEQ_LEN = 100
TRANSFORMER_EPOCHS = 8
TRANSFORMER_BATCH_SIZE = 256

TSGP_SD_DESIRED = 0.1                  # step size used during GP search
```

Every other module imports `config` and reads from it rather than hardcoding values — if you want to e.g. run a quicker smoke test, you create a second config or override the relevant function arguments (most functions accept overrides as kwargs, falling back to `config` when `None`).

---

### `primitives.py`

Defines the **GP primitive set** (the building blocks expression trees are made of) and the DEAP plumbing needed to create, evaluate, and evolve populations of trees. This is shared by both the data generator (Part 1) and the TSGP search (Part 3) — both use the *exact same* function/terminal set, which matters because the transformer is only ever trained on trees built from this primitive set.

**Primitive set** (`create_pset`):

| Token | Arity | Meaning |
|---|---|---|
| `add`, `sub`, `mul` | 2 | standard arithmetic |
| `protdiv` | 2 | **protected division** — returns `1.0` if `|right| < 1e-6` instead of raising `ZeroDivisionError` |
| `x0..x3` | 0 | input variables (terminals) |
| `ERC` | 0 | ephemeral random constant, sampled once per occurrence from `{-0.5, -0.4, ..., 0.5}` |

```python
def protdiv(left, right):
    if abs(right) < 1e-6:
        return 1.0
    return left / right
```

**Two DEAP toolboxes** are built from the same primitive set but configured differently for their two use cases:

- `setup_deap(pset)` — used by **TSGP search and stdGP baseline**. Standard tournament selection (size 5), one-point crossover, uniform mutation, height-limited to 17 (`GP_MAX_TREE_DEPTH`).
- `setup_datagen_toolbox(pset)` — used by **data generation only**. Uses `selDoubleTournament` (double tournament selection) instead of plain tournament selection:

```python
toolbox.register("select", _double_tournament, fitness_size=5,
                 parsimony_size=1.4, fitness_first=True)
```

Double tournament selection runs *two* tournaments in sequence: first picks the winner of a fitness-based tournament among a larger group, then a second tournament based on tree size (parsimony) decides the final pick. This penalizes overly complex trees while still prioritizing accuracy, which is exactly what the paper specifies for generating a *diverse but not bloated* pool of training functions.

**Fitness evaluation** (used during all GP runs):

```python
def evaluate_individual(individual, toolbox, X, y):
    func = toolbox.compile(expr=individual)
    y_pred = np.array([func(*row) for row in X])
    y_pred = np.nan_to_num(y_pred, nan=1e6, posinf=1e6, neginf=-1e6)  # guard against blowups
    y_pred = np.clip(y_pred, -1e6, 1e6)
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    return (rmse,)   # DEAP expects a tuple
```

**Semantics evaluation** (used only during data generation, to compute `s(f)` for the k-NN search):

```python
def evaluate_semantics(individual, toolbox, X):
    func = toolbox.compile(expr=individual)
    semantics = np.array([func(*row) for row in X], dtype=np.float64)
    semantics = np.nan_to_num(semantics, nan=0.0, posinf=1e6, neginf=-1e6)
    return np.clip(semantics, -1e6, 1e6)
```

Both functions guard numerically unstable trees (e.g. division-heavy trees that explode) by clipping rather than discarding — this keeps the population well-formed during selection instead of crashing GP runs.

---

### `tokenizer.py`

Converts between three representations of a function:

```
DEAP PrimitiveTree  ⟷  list[str] tokens  ⟷  list[int] token IDs (model input/output)
```

**Vocabulary construction** — built once at import time, entirely from `config`:

```python
SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN]      # <PAD>, <SOS>, <EOS>
FUNCTION_TOKENS = config.PRIMITIVE_SET_FUNCTIONS          # add, sub, mul, protdiv
VARIABLE_TOKENS = config.TERMINAL_VARIABLES               # x0, x1, x2, x3
ERC_TOKENS = [str(v) for v in config.ERC_VALUES]          # "-0.5", "-0.4", ..., "0.5"

VOCAB = SPECIAL_TOKENS + FUNCTION_TOKENS + VARIABLE_TOKENS + ERC_TOKENS
```

With 4 features and ERCs from -0.5 to 0.5 step 0.1 (11 values), the vocabulary has `3 + 4 + 4 + 11 = 22` tokens — a tiny vocabulary by NLP standards, which is part of why the model can be so small (934K params).

**Arity table** — every token's arity is precomputed so the syntax controller (see below) can validate sequences without re-deriving it:

```python
ARITY = {}
for t in FUNCTION_TOKENS:
    ARITY[TOKEN_TO_ID[t]] = 2     # add/sub/mul/protdiv all take 2 args
for t in VARIABLE_TOKENS + ERC_TOKENS:
    ARITY[TOKEN_TO_ID[t]] = 0     # terminals take 0 args
```

**Tree → tokens** (prefix traversal, which DEAP's `PrimitiveTree.__iter__` already gives you for free since the underlying list *is* prefix order):

```python
def tree_to_tokens(individual):
    tokens = []
    for node in individual:
        if isinstance(node, gp.Primitive):
            tokens.append(node.name)              # e.g. "add"
        elif isinstance(node, gp.Terminal):
            if node.name == "ERC":
                tokens.append(_snap_erc(node.value))  # round to nearest 0.1, e.g. "0.3"
            else:
                tokens.append(node.value)          # e.g. "x0"
    return tokens
```

`_snap_erc` exists because DEAP's ephemeral constants are floats that may drift slightly from the discrete grid (e.g. floating point `0.30000000000000004`) — snapping ensures every ERC maps cleanly onto a vocabulary token.

**Encoding for the model** (padding/truncating to `TRANSFORMER_MAX_SEQ_LEN`):

```python
def encode(tokens, max_len=100):                 # for the ENCODER input — no SOS/EOS
    ids = [TOKEN_TO_ID.get(t, PAD_ID) for t in tokens][:max_len]
    return ids + [PAD_ID] * (max_len - len(ids))

def encode_with_sos_eos(tokens, max_len=100):     # for the DECODER target — wrapped in SOS/EOS
    ids = ([SOS_ID] + [TOKEN_TO_ID.get(t, PAD_ID) for t in tokens] + [EOS_ID])[:max_len]
    return ids + [PAD_ID] * (max_len - len(ids))
```

**Tokens → tree** (used at inference time, to turn the transformer's sampled output back into something DEAP/GP can evaluate and evolve):

```python
def tokens_to_tree(tokens, pset):
    expr = []
    for token in tokens:
        # match against pset's primitives, then terminals, then fall back to parsing as an ERC float
        ...
    tree = gp.PrimitiveTree(expr)
    gp.compile(tree, pset)     # validates the tree actually compiles
    return tree
```

This function returns `None` if the token sequence doesn't form a valid, compilable tree — but in practice this should never happen because `syntax_control.py` prevents the model from ever sampling an invalid sequence in the first place (defense in depth).

---

### `data_generator.py`

This is **Part 1** — the semantic pair generator. It runs in four stages, all orchestrated by `generate_training_data()`.

#### Stage 1: Generate synthetic SR problems

```python
def generate_synthetic_problem(num_samples=200, num_features=4, noise_std=0.1):
    X = np.random.randn(num_samples, num_features)
    coefficients = np.random.randn(num_features)
    intercept = np.random.randn()
    y = X @ coefficients + intercept
    y += np.random.randn(num_samples) * noise_std    # Gaussian noise for diversity

    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)   # standardize
    y = (y - y.mean()) / (y.std() + 1e-8)
    return X, y
```

Each of the 50 problems (`NUM_SYNTHETIC_PROBLEMS`) is a random linear regression `y = Xβ + ε`, standardized to mean 0 / std 1 per the paper's specification. The point is *not* to solve these problems well — it's to use them as a fitness landscape that pushes stdGP toward generating a wide spread of varied, plausible expression trees.

#### Stage 2: Run stdGP to collect a pool of functions

```python
def run_stdgp_for_functions(X, y, pop_size=2000, generations=100, verbose=False):
    pset = create_pset(num_features=X.shape[1])
    toolbox = setup_datagen_toolbox(pset)        # <- double tournament selection
    toolbox.register("evaluate", evaluate_individual, toolbox=toolbox, X=X, y=y)

    pop = toolbox.population(n=pop_size)
    # ... standard generational GP loop (selection, crossover @ 90%, mutation @ 10%) ...

    # Every individual ever seen, across every generation, is collected by string-dedup:
    unique_functions = {}
    for ind in pop:
        key = str(ind)
        if key not in unique_functions:
            unique_functions[key] = toolbox.clone(ind)

    return list(unique_functions.values()), pset
```

The crucial detail: this does **not** just return the final population — it returns every *unique* tree seen across all 100 generations (≈100K+ trees per problem, based on a smoke test). Run across 50 problems, this is where the millions of source functions come from.

#### Stage 3: Compute semantics

```python
def compute_all_semantics(functions, pset, num_samples=100):
    toolbox = setup_deap(pset)
    X_semantic = np.random.randn(num_samples, config.NUM_FEATURES)   # fixed random probe inputs

    semantics, valid_indices = [], []
    for i, func in enumerate(functions):
        sem = evaluate_semantics(func, toolbox, X_semantic)
        if np.all(np.isfinite(sem)) and np.std(sem) > 1e-10:    # drop constants/degenerate trees
            semantics.append(sem)
            valid_indices.append(i)
    return np.array(semantics, dtype=np.float32), valid_indices
```

Every function is evaluated on the *same* 100 random standardized inputs, producing a 100-dimensional semantics vector. Functions whose output is constant (`std ≈ 0`, e.g. `x0 - x0`) or numerically degenerate (NaN/Inf) are filtered out — they'd be useless/misleading for the similarity search.

#### Stage 4: k-NN semantic similarity search (FAISS)

```python
def find_semantic_pairs(functions, semantics, valid_indices, k=3, sd_max=100.0):
    n, dim = semantics.shape
    if n > 100_000:
        # IVF index for large n: clusters semantics space, searches only nearby clusters
        nlist = min(int(np.sqrt(n)), 4096)
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist)
        index.train(semantics); index.add(semantics)
        index.nprobe = min(nlist, 64)
    else:
        index = faiss.IndexFlatL2(dim)    # exact search for small n
        index.add(semantics)

    distances, indices = index.search(semantics, k + 1)   # +1 because nearest neighbor of f is f itself

    pairs = []
    for i in range(n):
        for j_pos in range(k + 1):
            j = indices[i, j_pos]
            if j == i: continue
            sd = np.sqrt(distances[i, j_pos])              # L2 index returns squared distances
            if sd > 0 and sd < sd_max:                       # filter: paper requires 0 < SD < 100
                pairs.append((valid_indices[i], valid_indices[j], sd))
            if len(pairs) >= config.TARGET_NUM_PAIRS:
                return pairs
    return pairs
```

For *every* function `fᵢ`, this finds its `k=3` semantically nearest neighbors and emits `(fᵢ, fⱼ, SD)` triples — exactly the algorithm in Eq. (1) of the paper. The `IndexIVFFlat` branch matters in practice: with millions of functions, exact brute-force L2 search (`IndexFlatL2`) becomes too slow, so above 100K vectors FAISS first clusters the semantic space (`nlist` clusters) and only searches the `nprobe=64` nearest clusters per query — a standard approximate nearest neighbor speedup.

#### Putting it together: `generate_training_data`

```python
def generate_training_data(output_dir, num_problems=50, verbose=True):
    all_functions, all_token_seqs = [], []
    for prob_idx in range(num_problems):
        X, y = generate_synthetic_problem(noise_std=random.uniform(0.05, 0.3))
        functions, pset = run_stdgp_for_functions(X, y, ...)
        for func in functions:
            tokens = tree_to_tokens(func)
            if len(tokens) <= config.TRANSFORMER_MAX_SEQ_LEN - 2:   # must fit with SOS/EOS room
                all_functions.append(func)
                all_token_seqs.append(tokens)

    semantics, valid_indices = compute_all_semantics(all_functions, pset)
    pairs = find_semantic_pairs(all_functions, semantics, valid_indices)

    training_data = [
        {"input_tokens": all_token_seqs[i], "output_tokens": all_token_seqs[j], "sd": sd}
        for i, j, sd in pairs
    ]

    # Saved as BOTH pickle (fast to load) and CSV (human-readable/portable)
    pickle.dump(training_data, open(f"{output_dir}/training_pairs.pkl", "wb"))
    # CSV: space-joined token strings, one row per pair
    ...
    return training_data
```

Output is a list of dicts — this exact schema (`input_tokens`, `output_tokens`, `sd`) is what `train_transformer.py` consumes directly.

> **Performance note (measured on an M4 Pro):** ~38 seconds per synthetic problem at full config (pop=2000, gen=100) → ~32 minutes for all 50 problems, plus a few more minutes for semantics computation and FAISS search. Total: **~35-40 minutes** for the entire Part 1 pipeline.

---

### `transformer_model.py`

This is **Part 2** — the Keras/TensorFlow encoder-decoder transformer. It's a fairly direct implementation of Vaswani et al.'s "Attention Is All You Need" architecture, scaled down per the paper's spec (2 encoder layers, 2 decoder layers, 8 heads, d_model=128), with one addition: **semantic-distance conditioning**.

#### Positional Encoding

Standard sinusoidal positional encoding, precomputed once (not learned):

```python
class PositionalEncoding(layers.Layer):
    def __init__(self, max_len, d_model, **kwargs):
        super().__init__(**kwargs)
        positions = np.arange(max_len)[:, np.newaxis]
        dims = np.arange(d_model)[np.newaxis, :]
        angles = positions / np.power(10000, (2 * (dims // 2)) / d_model)
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        self.pe = tf.constant(angles[np.newaxis, :, :].astype(np.float32))

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pe[:, :seq_len, :]
```

#### Encoder / Decoder blocks

Textbook transformer blocks — multi-head self-attention + feed-forward, each wrapped in residual connection + layer norm:

```python
class TransformerEncoderBlock(layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.ffn = keras.Sequential([layers.Dense(dff, activation="relu"), layers.Dense(d_model)])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        ...

    def call(self, x, padding_mask=None, training=False):
        attn_output = self.mha(x, x, x, attention_mask=padding_mask, training=training)
        x = self.layernorm1(x + self.dropout1(attn_output, training=training))
        ffn_output = self.ffn(x)
        x = self.layernorm2(x + self.dropout2(ffn_output, training=training))
        return x
```

The decoder block additionally has a second `MultiHeadAttention` that performs cross-attention over the encoder's output, plus a causal "look-ahead" mask so the decoder can't attend to future tokens during teacher-forced training.

#### Semantic distance (SD) conditioning — the one architectural deviation from vanilla transformers

This is the implementation of the paper's key idea ("we provide not only `fᵢ` as input, but also the semantic distance ... as an additional input to the encoder and decoder layer"):

```python
self.sd_projection = layers.Dense(d_model)   # scalar SD -> d_model-dim vector

def encode(self, enc_input, sd_input, training=False):
    x = self.enc_embedding(enc_input)
    x = x * np.sqrt(float(self.d_model))      # standard transformer embedding scale
    x = self.enc_pos_encoding(x)

    sd_expanded = tf.expand_dims(tf.expand_dims(sd_input, -1), -1)  # (batch,) -> (batch, 1, 1)
    sd_embed = self.sd_projection(sd_expanded)                       # -> (batch, 1, d_model)
    sd_embed = tf.broadcast_to(sd_embed, tf.shape(x))                 # broadcast over seq_len
    x = x + sd_embed                                                   # added to every position
    ...
```

The scalar `SD` (a single float per example, e.g. `0.1`) is projected through a learned `Dense(d_model)` layer and **added elementwise to every token's embedding**, in both the encoder and decoder. This is how the model learns "for this SD value, generate tokens that keep the output this far from the input" — and it's exactly why, at inference time, you can hand the model an arbitrary `SD_d` and get offspring at roughly that semantic step size.

#### Full model assembly

```python
class TSGPTransformer(keras.Model):
    def call(self, inputs, training=False):
        enc_input, dec_input, sd_input = inputs        # 3 inputs: parent tokens, partial output tokens, desired SD
        enc_output = self.encode(enc_input, sd_input, training=training)
        dec_output = self.decode(dec_input, enc_output, enc_input, sd_input, training=training)
        return self.final_layer(dec_output)             # (batch, seq_len, VOCAB_SIZE) logits
```

Note the model takes **3 inputs**, not the usual 2 (encoder input + decoder input) — the `sd_input` scalar is threaded through both halves.

```python
def create_model():
    model = TSGPTransformer()
    # build the model by calling it once on dummy data (Keras subclassed models need this)
    model([zeros(1, 100), zeros(1, 100), zeros(1)], training=False)
    return model
```

Total params: **~934K** — tiny by transformer standards, which is exactly the point: this isn't meant to be a general-purpose model, just enough capacity to learn local structural transformations between semantically-similar expression trees.

---

### `syntax_control.py`

Implements **grammar-constrained decoding**: at every step of auto-regressive sampling, illegal next-tokens are masked out (probability forced to ~0) so the model is *structurally incapable* of producing a malformed tree. This is the implementation of Wittenberg et al.'s "syntax control" referenced in the paper.

The core idea is tracking an **arity stack** — same concept as validating balanced parentheses, generalized to n-ary trees:

```python
class SyntaxController:
    def reset(self):
        self.arity_stack = []        # one entry per "open" function call, value = args still needed
        self.tokens_generated = 0
        self.complete = False
```

- Sampling a function token (e.g. `add`, arity 2) **pushes 2** onto the stack — "this node needs 2 more children before it's complete."
- Sampling a terminal (e.g. `x0`, arity 0) **decrements the top of the stack by 1**. If that hits 0, the node is satisfied, so it pops — and *that* might satisfy its own parent, cascading pops up the stack.
- The tree is complete exactly when the stack is empty after at least one token.

```python
def update(self, token_id):
    if token_id == EOS_ID:
        self.complete = True
        return
    self.tokens_generated += 1
    if token_id in FUNCTION_IDS:
        self.arity_stack.append(ARITY[token_id])
    elif token_id in TERMINAL_IDS:
        if self.arity_stack:
            self.arity_stack[-1] -= 1
            while self.arity_stack and self.arity_stack[-1] == 0:
                self.arity_stack.pop()
                if self.arity_stack:
                    self.arity_stack[-1] -= 1
    if not self.arity_stack and self.tokens_generated > 0:
        self.complete = True
```

**Masking logic** — before sampling each token, `get_valid_mask()` returns an additive mask (`0` for valid tokens, `-1e9` for invalid ones) that gets added to the model's logits before the softmax:

```python
def get_valid_mask(self):
    mask = np.full(VOCAB_SIZE, -1e9)

    if self.complete:
        mask[EOS_ID] = 0.0                          # only EOS valid once tree is done
        return mask
    if self.tokens_generated == 0:
        # first token: any function OR terminal is a legal tree root
        for fid in FUNCTION_IDS: mask[fid] = 0.0
        for tid in TERMINAL_IDS: mask[tid] = 0.0
        return mask

    remaining_slots = sum(self.arity_stack)
    if remaining_slots == 0:
        mask[EOS_ID] = 0.0                           # stack empty -> must stop
        return mask

    tokens_left = self.max_tokens - self.tokens_generated - 1
    at_max_depth = len(self.arity_stack) >= self.max_depth
    must_use_terminal = (tokens_left <= remaining_slots) or at_max_depth

    if must_use_terminal:
        for tid in TERMINAL_IDS: mask[tid] = 0.0      # forced terminal: running out of budget or depth
    else:
        for fid in FUNCTION_IDS: mask[fid] = 0.0       # free choice: either continues fine
        for tid in TERMINAL_IDS: mask[tid] = 0.0
    return mask
```

Two safety conditions force a terminal token even when a function token would otherwise be "grammatically legal":
1. **`tokens_left <= remaining_slots`** — if there isn't enough room left in the max sequence length to satisfy all currently-open arity slots even with all-terminal children, you *must* close out with terminals now.
2. **`at_max_depth`** — enforces `GP_MAX_TREE_DEPTH` (17), preventing runaway deep trees.

This guarantees every sequence the transformer samples is convertible back to a valid `gp.PrimitiveTree` via `tokenizer.tokens_to_tree` — no rejection sampling or retry loop needed.

---

### `train_transformer.py`

The training driver for the transformer model. Loads the pairs produced by `data_generator.py`, encodes them, and runs a manual (non-`model.fit`) training loop with checkpointing and resume support.

#### Data preparation

```python
def prepare_numpy_data(training_data):
    input_tokens = [d["input_tokens"] for d in training_data]
    output_tokens = [d["output_tokens"] for d in training_data]
    sd_values = np.array([d["sd"] for d in training_data], dtype=np.float32)

    enc_input = batch_encode_encoder(input_tokens)        # (N, 100) -- parent function, padded
    dec_full = batch_encode_decoder(output_tokens)        # (N, 100) -- <SOS> + offspring + <EOS>, padded
    dec_input = dec_full[:, :-1]                           # teacher forcing input: everything but last token
    dec_target = dec_full[:, 1:]                            # teacher forcing target: everything but first token
    return enc_input, dec_input, sd_values, dec_target
```

This is the standard **teacher forcing** shift: if the true decoder sequence is `<SOS> add x0 x1 <EOS>`, the model is trained to predict `add x0 x1 <EOS>` (the target) given `<SOS> add x0 x1` (the input) — i.e. predict each token from everything before it.

#### Masked loss

```python
def masked_loss(y_true, y_pred):
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")
    loss = loss_fn(y_true, y_pred)
    mask = tf.cast(tf.not_equal(y_true, PAD_ID), tf.float32)   # ignore padded positions
    return tf.reduce_sum(loss * mask) / (tf.reduce_sum(mask) + 1e-8)
```

Since sequences are padded to a fixed length, loss must be masked — otherwise the model would be rewarded for confidently predicting `<PAD>` over and over in the unused tail of every sequence, which teaches it nothing useful.

#### Resume-from-checkpoint logic

Training 8 epochs over 5 million pairs is a long-running job, so the training loop can recover from an interrupted run:

```python
def find_latest_checkpoint(checkpoint_dir):
    # scans checkpoints/tsgp_epoch_*.weights.h5, returns (path, epoch_number) of the highest epoch found

def prompt_resume(checkpoint_dir):
    latest_file, latest_epoch = find_latest_checkpoint(checkpoint_dir)
    if latest_file is None:
        return 0, None                          # nothing to resume, start fresh silently
    # lists all available checkpoints, then prompts interactively:
    #   "Resume from epoch 5? [Y/n/epoch_number]:"
    #   Y/Enter -> resume from latest
    #   n       -> start over from epoch 1
    #   <int>   -> resume from that specific epoch's checkpoint
```

```python
def train_model(training_data, checkpoint_dir="checkpoints", epochs=8, batch_size=256, verbose=True):
    start_epoch, resume_path = prompt_resume(checkpoint_dir)
    base_model = create_model()
    if resume_path is not None:
        base_model.load_weights(resume_path)
        print(f"Resuming training from epoch {start_epoch + 1}")
    ...
    for epoch in range(start_epoch + 1, epochs + 1):
        # tqdm progress bar over batches, running average loss in the postfix
        for start in tqdm(range(0, n, batch_size), desc=f"Epoch {epoch}/{epochs}"):
            loss = wrapper.train_on_batch([batch_enc, batch_dec, batch_sd], batch_target)
            ...
        base_model.save_weights(f"{checkpoint_dir}/tsgp_epoch_{epoch}.weights.h5")   # every epoch

    base_model.save_weights(f"{checkpoint_dir}/tsgp_final.weights.h5")               # final, convenience copy
```

Key behaviors:
- **Weights are saved after every single epoch** (`tsgp_epoch_{N}.weights.h5`), not just at the end — so an interrupted run never loses more than one epoch of progress.
- On the *next* invocation of `train_model`, if any `tsgp_epoch_*.weights.h5` files exist, the user is interactively prompted to resume rather than silently restarting (or silently resuming) — this avoids both wasted compute and a confusing "wait, why is loss not improving" if you forgot you already trained for a few epochs.
- `optimizer = keras.optimizers.legacy.Adam(...)` — deliberately the **legacy** Adam implementation. On Apple Silicon, TF's newer (`v2.11+`) optimizer implementation runs significantly slower because it isn't well-optimized for the Metal backend; the legacy optimizer avoids that regression. (See [Apple Silicon GPU Setup](#apple-silicon-gpu-setup) below for why this matters.)
- `wrapper.compile(optimizer=optimizer, loss=masked_loss)` — note that loss is passed explicitly to `compile`, not computed via an overridden `compute_loss`. An earlier version of this code overrode `Model.compute_loss` directly, but that path made `train_on_batch` return a list instead of a scalar; passing `loss=` through `compile` is what makes `train_on_batch` return a clean float.

---

### `tsgp_search.py`

This is **Part 3** — the actual TSGP variation operator and the GP evolutionary loop that uses it. Also contains `run_stdgp_baseline`, an unmodified standard GP loop used as a comparison baseline (per the paper's experimental design).

#### `TSGPSearchOperator` — the transformer as a variation operator

```python
class TSGPSearchOperator:
    def __init__(self, model, pset, sd_desired=None, temperature=1.0):
        self.model = model
        self.pset = pset
        self.sd_desired = sd_desired or config.TSGP_SD_DESIRED    # 0.1, per paper

    def sample_offspring(self, parent_individual):
        parent_tokens = tree_to_tokens(parent_individual)
        enc_input = np.array([encode(parent_tokens)], dtype=np.int32)
        sd_input = np.array([self.sd_desired], dtype=np.float32)

        syntax_ctrl = SyntaxController()
        dec_tokens = [SOS_ID]

        for _ in range(config.TRANSFORMER_MAX_SEQ_LEN - 1):
            dec_padded = pad(dec_tokens, to=100)
            logits = self.model([enc_input, dec_padded, sd_input], training=False)

            next_logits = logits[0, len(dec_tokens) - 1, :] / self.temperature
            next_logits = apply_syntax_mask(next_logits, syntax_ctrl)   # zero out invalid tokens

            probs = softmax(next_logits)
            token_id = np.random.choice(VOCAB_SIZE, p=probs)             # sample, not argmax

            if token_id == EOS_ID or syntax_ctrl.is_complete():
                break
            syntax_ctrl.update(token_id)
            dec_tokens.append(token_id)
            if syntax_ctrl.is_complete():
                break

        output_tokens = decode(dec_tokens)
        return tokens_to_tree(output_tokens, self.pset)
```

This is the token-by-token auto-regressive sampling loop from Section 3.2 of the paper:
1. The parent tree is tokenized and run through the **encoder** once (conceptually — in practice the full model is re-run each decoding step, since this is a simple non-cached implementation rather than using a KV-cache).
2. At each decoding step, the model produces logits for the *next* token given everything decoded so far. `syntax_control.apply_syntax_mask` zeroes out any token that would make the sequence un-treeable.
3. The next token is **sampled** from the resulting (masked, temperature-scaled) softmax distribution — not greedily argmax'd — which is what gives TSGP its exploration/stochasticity, analogous to mutation/crossover randomness in standard GP.
4. Decoding stops at `<EOS>` or when the syntax controller declares the tree structurally complete (arity stack empty), whichever comes first.
5. The resulting token sequence is decoded back into a `gp.PrimitiveTree`.

#### `run_tsgp` — the evolutionary loop

```python
def run_tsgp(model, X_train, y_train, X_test, y_test, pop_size=100, generations=50, verbose=True):
    pset = create_pset()
    toolbox = setup_deap(pset)                       # standard tournament selection (size 5)
    toolbox.register("evaluate", evaluate_individual, toolbox=toolbox, X=X_train, y=y_train)
    search_op = TSGPSearchOperator(model, pset)

    pop = toolbox.population(n=pop_size)
    for ind in pop: ind.fitness.values = toolbox.evaluate(ind)

    for gen in range(1, generations + 1):
        selected = toolbox.select(pop, pop_size)       # tournament selection, as usual

        offspring = []
        for parent in selected:
            child_tree = search_op.sample_offspring(parent)     # <-- transformer instead of crossover/mutation
            if child_tree is not None and len(child_tree) > 0:
                child = creator.Individual(child_tree)
                child.fitness.values = toolbox.evaluate(child)
                if child.fitness.values[0] < 1e6:                # valid, finite fitness
                    offspring.append(child)
                    continue
            # fallback: if sampling/evaluation failed, the parent survives unchanged
            clone = toolbox.clone(parent)
            clone.fitness.values = parent.fitness.values
            offspring.append(clone)

        pop[:] = offspring
        # track best train RMSE / size per generation for analysis plots
    ...
```

The structure is **identical to a standard generational GP loop** — selection, then variation, then replacement — except the variation step calls `search_op.sample_offspring(parent)` instead of `toolbox.mate` / `toolbox.mutate`. This is the literal embodiment of the paper's framing: "TSGP replaces the standard variation operators of GP with a semantic-aware transformer."

The fallback-to-parent-clone logic handles the (rare) case where the transformer produces a degenerate tree (e.g. one that fails to compile, or whose fitness blows up past the `1e6` sentinel) — rather than letting a broken individual corrupt the population, that slot is simply filled by a clone of the parent.

#### `run_stdgp_baseline`

A conventional GP loop (subtree crossover @ 90%, subtree mutation @ 10%) used purely as the `stdGP` comparison point from the paper's Table 2 — structurally the same as `run_tsgp` but with `toolbox.mate`/`toolbox.mutate` in place of the transformer call.

---

### `run_experiments.py`

The benchmark harness: loads a trained model, runs `run_tsgp` and `run_stdgp_baseline` across the five PMLB black-box datasets used in the paper, for `NUM_RUNS=30` independent runs each, and aggregates results.

```python
DATASETS = ["ERA", "ESL", "Galaxy", "LEV", "pollen"]

def load_and_prepare_dataset(name):
    X, y = pmlb.fetch_data(name, return_X_y=True)
    X = X[:, :config.NUM_FEATURES]                                  # keep only first 4 features

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)  # 50/50 split

    X_train = StandardScaler().fit_transform(X_train)               # standardize features (fit on train only)
    X_test = scaler_X.transform(X_test)
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel() # standardize target too
    y_test = scaler_y.transform(y_test.reshape(-1, 1)).ravel()
    return X_train, X_test, y_train, y_test
```

This matches the paper's experimental setup exactly: 50/50 train/test split, standardized features and target (mean 0, std 1).

```python
def run_all_experiments(model_weights_path, output_dir="results", num_runs=30, verbose=True):
    model = create_model()
    model.load_weights(model_weights_path)

    all_results = {}
    for dataset_name in DATASETS:
        X_train, X_test, y_train, y_test = load_and_prepare_dataset(dataset_name)

        tsgp_results, stdgp_results = [], []
        for run in range(num_runs):
            _, tsgp_stats = run_tsgp(model, X_train, y_train, X_test, y_test, verbose=verbose)
            tsgp_results.append(tsgp_stats)
            _, stdgp_stats = run_stdgp_baseline(X_train, y_train, X_test, y_test, verbose=verbose)
            stdgp_results.append(stdgp_stats)

        # median/mean/std of test RMSE and final solution size, across the 30 runs
        all_results[dataset_name] = { "tsgp": {...}, "stdgp": {...} }

    json.dump(all_results, open(f"{output_dir}/experiment_results.json", "w"), indent=2)
    return all_results
```

Run from the command line:

```bash
python -m tsgp.run_experiments --weights checkpoints/tsgp_final.weights.h5 --output results --runs 30
```

The output JSON (`results/experiment_results.json`) contains, per dataset, the full distribution of test RMSEs and solution sizes across all 30 runs for both TSGP and stdGP — this is the raw data needed to reproduce the paper's Table 2 / Table 3 style comparisons (median test RMSE, median solution size) and run significance tests (e.g. Wilcoxon rank-sum) on top, if desired.

---

## End-to-End Data Flow

Putting all the modules together, here's what actually happens when you run the full pipeline:

```
config.py
   │  (constants used everywhere below)
   ▼
┌─────────────────────────────── PART 1 ───────────────────────────────┐
│ data_generator.generate_training_data()                                │
│   1. generate_synthetic_problem() × 50  → (X, y) standardized regression problems
│   2. run_stdgp_for_functions()  → ~100K+ unique DEAP trees per problem  │
│        (uses primitives.setup_datagen_toolbox — double tournament)     │
│   3. tokenizer.tree_to_tokens()  → token sequences, filtered to ≤98 tok│
│   4. compute_all_semantics()  → s(f) ∈ R^100 per function              │
│        (uses primitives.evaluate_semantics)                            │
│   5. find_semantic_pairs()  → FAISS k-NN (k=3), SD filter 0<SD<100     │
│   6. → training_pairs.pkl / .csv : [{input_tokens, output_tokens, sd}] │
└──────────────────────────────────┬─────────────────────────────────────┘
                                    ▼
┌─────────────────────────────── PART 2 ───────────────────────────────┐
│ train_transformer.train_from_data("data/training/training_pairs.pkl")  │
│   1. load pickle → encode via tokenizer.batch_encode_encoder/_decoder  │
│   2. transformer_model.create_model()  → TSGPTransformer (934K params) │
│        (encoder + decoder, SD-conditioned, syntax-agnostic at train time)│
│   3. teacher-forced training loop, masked_loss, AdamW-equivalent       │
│        legacy Adam, lr=1e-3, 8 epochs, batch=256                       │
│   4. checkpoints/tsgp_epoch_{1..8}.weights.h5 + tsgp_final.weights.h5  │
└──────────────────────────────────┬─────────────────────────────────────┘
                                    ▼
┌─────────────────────────────── PART 3 ───────────────────────────────┐
│ run_experiments.run_all_experiments(weights_path)                      │
│   for each PMLB dataset (ERA, ESL, Galaxy, LEV, pollen):                │
│     load_and_prepare_dataset()  → 50/50 split, standardized            │
│     for run in 1..30:                                                   │
│       tsgp_search.run_tsgp(model, X_train, y_train, X_test, y_test)     │
│         → DEAP population, 50 generations                              │
│         → each generation: tournament select, then                     │
│           TSGPSearchOperator.sample_offspring() per parent              │
│             → tokenize parent → transformer forward pass (per token)   │
│             → syntax_control masks invalid tokens at every step        │
│             → sample → decode tokens → tokens_to_tree()                │
│       tsgp_search.run_stdgp_baseline(...)   (comparison baseline)       │
│   → results/experiment_results.json (median/mean/std RMSE & size)      │
└───────────────────────────────────────────────────────────────────────┘
```

---

## How to Run Everything

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate training data (Part 1)

```bash
python -m tsgp.data_generator
```

Produces `data/training/training_pairs.pkl` and `data/training/training_pairs.csv`. Takes roughly **35-40 minutes** at the paper's full configuration (50 problems × pop 2000 × 100 generations). Both `data/` and `checkpoints/` are gitignored — these artifacts are not meant to be committed.

### 3. Train the transformer (Part 2)

```bash
python -m tsgp.train_transformer
```

Trains for 8 epochs, saving a checkpoint after every epoch to `checkpoints/`. If interrupted, simply re-run the same command — it will detect existing checkpoints and interactively ask whether/where to resume from.

### 4. Run the benchmark experiments (Part 3)

```bash
python -m tsgp.run_experiments --weights checkpoints/tsgp_final.weights.h5 --output results --runs 30
```

Runs TSGP and the stdGP baseline 30 times each, across all 5 PMLB datasets, writing aggregated results to `results/experiment_results.json`.

---

## Apple Silicon GPU Setup

This project targets `tensorflow==2.15.*` + `tensorflow-metal==1.1.*`, which is the last known-good pairing for Apple Silicon GPU acceleration:

- **TF ≥ 2.16 does not currently expose a working GPU device on macOS** via `tensorflow-metal` (the plugin hasn't kept pace with newer TF releases as of this writing) — `tf.config.list_physical_devices('GPU')` returns `[]`.
- **TF 2.15 + tensorflow-metal 1.1** correctly registers the Metal GPU device.
- `jax`/`jaxlib`, if installed in the same environment, can pull in a `ml_dtypes` version that conflicts with TF 2.15's dependency pin — uninstall them if you hit `ValueError: JAX requires ml_dtypes version 0.5 or newer`.
- The `keras.optimizers.legacy.Adam` optimizer is used deliberately in `train_transformer.py` — the newer (`v2.11+`) Adam implementation in `tf.keras.optimizers.Adam` is known to run significantly slower on M1/M2/M4 GPUs.

Verify GPU is active:

```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
# Expect: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

---

## Design Decisions and Gotchas

- **Why both pickle and CSV for training pairs?** Pickle is what `train_transformer.py` actually loads (fast, preserves Python types). CSV is provided purely for human inspection / portability (e.g. opening in Excel, loading into pandas, or uploading somewhere that doesn't support pickle deserialization).
- **Sampling, not greedy decoding, during TSGP search.** `TSGPSearchOperator.sample_offspring` samples from the (masked) softmax distribution rather than taking the argmax at each step. This preserves the stochastic exploration that GP relies on — a deterministic decoder would make every offspring of the same parent identical given a fixed `SD_d`, collapsing genetic diversity.
- **Fallback to parent clone on failed offspring.** If the transformer produces a tree that fails to compile or has a degenerate/exploded fitness, `run_tsgp` substitutes a clone of the parent rather than discarding the population slot — this keeps population size constant without needing a retry/rejection loop, at the cost of occasionally not making progress that generation.
- **Why double tournament selection only in data generation, not in TSGP search?** The paper specifies double tournament selection (`fitness_size=5, parsimony_size=1.4`) specifically for generating the *training pool* of functions, to keep that pool diverse without being dominated by bloated trees. The actual TSGP/stdGP search loops use plain tournament selection (size 5), matching Table 1's `Selection` row, which applies to the benchmark experiments, not data generation.
- **No KV-caching during inference.** `sample_offspring` re-runs the full encoder+decoder forward pass at every decoding step rather than caching key/value projections — this is simpler to reason about and the sequences are short (≤100 tokens) and the model is small (934K params), so the performance cost is acceptable for a research implementation. A production-grade version would cache.
- **Numerical safety nets everywhere.** `protdiv`, `evaluate_individual`, and `evaluate_semantics` all guard against NaN/Inf/overflow by clipping to `±1e6` or substituting a fixed sentinel. This is necessary because GP trees built from `add/sub/mul/protdiv` over standardized data can still occasionally produce extreme values (e.g. deeply nested multiplications), and an unguarded NaN would silently corrupt the FAISS index or the fitness-based selection.
