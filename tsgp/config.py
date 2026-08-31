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
DATAGEN_GP_GENERATIONS = 100
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

TSGP_SD_DESIRED = 0.1
# Sampling temperature for the transformer variation operator. The paper does
# not state one, so 1.0 (no scaling) is the neutral default; override per-run
# with --temperature rather than changing this.
TSGP_TEMPERATURE = 1.0
