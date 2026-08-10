NUM_FEATURES = 4

PRIMITIVE_SET_FUNCTIONS = ["add", "sub", "mul", "protdiv"]
TERMINAL_VARIABLES = [f"x{i}" for i in range(NUM_FEATURES)]
ERC_MIN = -0.5
ERC_MAX = 0.5
ERC_STEP = 0.1
ERC_VALUES = [round(ERC_MIN + i * ERC_STEP, 1)
              for i in range(int((ERC_MAX - ERC_MIN) / ERC_STEP) + 1)]

GP_POP_SIZE = 100
GP_GENERATIONS = 50
GP_TOURNAMENT_SIZE = 5
GP_INIT_DEPTH_MIN = 2
GP_INIT_DEPTH_MAX = 5
GP_MAX_TREE_DEPTH = 17
GP_CROSSOVER_PROB = 0.9
GP_MUTATION_PROB = 0.1
GP_MUTATION_DEPTH_MIN = 0
GP_MUTATION_DEPTH_MAX = 2
GP_INTERNAL_NODE_BIAS = 0.1
NUM_RUNS = 30

NUM_SYNTHETIC_PROBLEMS = 50
DATAGEN_GP_POP_SIZE = 2000
# Sect. 4.1: data generation uses "all other GP search parameters ... as
# defined in Table 1", and Table 1 fixes 50 generations. The original 100 was
# a deviation: twice the run length means twice the bloat, which pushes the
# pool's function sizes away from the sizes the GP search actually operates on.
DATAGEN_GP_GENERATIONS = 50
# Number of random standardised inputs used to approximate semantics. The paper
# does not state one. It matters more than it looks: SD is a Euclidean norm
# over this many dimensions, so it scales as sqrt(NUM_SEMANTIC_SAMPLES), and
# the SD < 100 filter therefore decides how much of the function space enters
# training at all.
NUM_SEMANTIC_SAMPLES = 100
KNN_K = 3
SD_MAX_THRESHOLD = 100.0
TARGET_NUM_PAIRS = 5_000_000

TRANSFORMER_NUM_HEADS = 8
TRANSFORMER_HIDDEN_DIM = 128
TRANSFORMER_NUM_ENCODER_LAYERS = 2
TRANSFORMER_NUM_DECODER_LAYERS = 2
TRANSFORMER_MAX_SEQ_LEN = 100
TRANSFORMER_DROPOUT = 0.1
TRANSFORMER_LEARNING_RATE = 1e-3
# Paper Sect. 4.1 specifies AdamW at lr 1e-3 but does not state a weight decay;
# this is the Keras AdamW default, which is what their Keras implementation
# would have used.
TRANSFORMER_WEIGHT_DECAY = 0.004
TRANSFORMER_EPOCHS = 8
TRANSFORMER_BATCH_SIZE = 256

# --- semantic-distance conditioning ---
# Sect. 3.2 states SD is supplied to the encoder and decoder so the desired
# step size can be controlled. Feeding the RAW scalar makes that nominal only:
# 75% of training SDs are below 0.667 while the tail reaches 100, so a linear
# projection sees almost no variation across the bulk and the learned weight
# stays tiny (measured: ||0.1*W|| = 0.032 against ||bias|| = 1.065, and
# SD_d = 0.0 vs 0.1 produce byte-identical output).
# Normalising with log1p + standardisation gives the projection a
# well-conditioned input. Off by default so existing checkpoints behave as
# trained; enable per-run with --sd-normalize.
TRANSFORMER_SD_NORMALIZE = False
TRANSFORMER_SD_LOG_MEAN = 0.464287   # mean of log1p(SD) over the 5M pairs
TRANSFORMER_SD_LOG_STD = 0.737271    # std  of log1p(SD) over the 5M pairs

# How SD is encoded before it reaches the layers. Sect. 3.1 says only that SD is
# supplied "as an additional input to the encoder and decoder layer" and never
# specifies an encoding, so this is a free parameter, not a deviation.
#
#   "linear" — the original: Dense(d_model) on the raw scalar. This is rank one,
#     so the whole SD signal lives on a single direction scaled linearly by SD.
#     Measured magnitude at the operating point SD_d = 0.1 is 0.032, against
#     token embeddings scaled by sqrt(128) = 11.3. That is 0.3% of the signal:
#     the conditioning is numerically incapable of mattering, which is why
#     achieved distance tracks the requested distance only weakly (1.15 / 1.55 /
#     6.44 for SD_d of 0.1 / 1 / 10).
#   "binned" — bucket log1p(SD) into TRANSFORMER_SD_NUM_BINS bins and look up a
#     learned embedding, scaled like the token embeddings. Gives the model
#     d_model degrees of freedom per bin instead of one shared direction, and a
#     magnitude that can actually compete with the token stream.
TRANSFORMER_SD_ENCODING = "binned"
TRANSFORMER_SD_NUM_BINS = 64
# Bins are laid out on log10 over [SD_BIN_MIN, SD_BIN_MAX], not on log1p.
# log1p is nearly linear below 1 and the training SDs are heavily massed there
# (p25 0.049, p50 0.164, p75 0.667), so log1p bins would spend 56 of 64 bins on
# the sparse tail above 1 and squeeze the whole interquartile range into 8.
# log10 over five decades gives ~13 bins per decade, so the low end -- which is
# where the operator is actually queried, SD_d = 0.1 -- gets real resolution,
# and asking for a step smaller than the training median becomes expressible.
TRANSFORMER_SD_BIN_MIN = 1e-3        # below the 1st percentile (0.0014)
TRANSFORMER_SD_BIN_MAX = 100.0       # matches SD_MAX_THRESHOLD

TSGP_SD_DESIRED = 0.1

# Semantic step control: sample this many offspring per parent and keep the one
# semantically nearest the parent. 1 is the paper's operator exactly.
#
# Measured motivation (tsgp.step_floor): a single sample cannot honour SD_d =
# 0.1. Driving the request from 0.1 down to 0.0001 leaves the achieved distance
# flat at ~0.7, so the paper's operating point is below the operator's floor,
# and this holds for both SD encodings at matched training -- it is a property
# of how coarsely tokens map onto semantics, not of the conditioning. Taking the
# nearest of k draws goes underneath it: 0.79 / 0.42 / 0.24 / 0.15 / 0.10 for
# k = 1 / 2 / 4 / 8 / 16.
#
# This is the step-size control Sect. 5 lists as future work, so any k > 1 is a
# documented extension of the paper rather than a reproduction of it. Every
# result file records the k it was produced with.
TSGP_STEP_K = 1

# Annealed step-size control (Sect. 5's future work, NOT the paper's operator).
#
# With TSGP_STEP_ANNEAL on, the k candidates are scored by |d - target| rather
# than by d, where target decays geometrically across the run. Selecting the
# *smallest* d instead was tested and made things worse (ESL paired: 0.4135 at
# k=1 vs 0.4246 at k=8) because it makes offspring near-copies from generation
# one and starves exploration.
#
# The schedule is expressed as a fraction of ||y_train||, so it transfers
# across data sets. It is set to imitate the profile stdGP produces implicitly:
# measured on ESL (||y|| = 15.6) stdGP's parent-offspring distance falls from
# ~28 at generation 1 to ~1.0 at generation 50, i.e. from ~1.8x||y|| down to
# ~0.06x||y||. TSGP has no such mechanism and sits flat at 16-34 throughout.
TSGP_STEP_ANNEAL = False
TSGP_STEP_FRAC_START = 2.0
TSGP_STEP_FRAC_END = 0.1
# Sampling temperature for the transformer variation operator. The paper does
# not state one, so 1.0 (no scaling) is the neutral default; override per-run
# with --temperature rather than changing this.
TSGP_TEMPERATURE = 1.0
