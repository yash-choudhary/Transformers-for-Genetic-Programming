import numpy as np
from deap import gp

from . import config


PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"

SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN]
FUNCTION_TOKENS = config.PRIMITIVE_SET_FUNCTIONS
VARIABLE_TOKENS = config.TERMINAL_VARIABLES
ERC_TOKENS = [str(v) for v in config.ERC_VALUES]

VOCAB = SPECIAL_TOKENS + FUNCTION_TOKENS + VARIABLE_TOKENS + ERC_TOKENS

TOKEN_TO_ID = {token: idx for idx, token in enumerate(VOCAB)}
ID_TO_TOKEN = {idx: token for idx, token in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)

PAD_ID = TOKEN_TO_ID[PAD_TOKEN]
SOS_ID = TOKEN_TO_ID[SOS_TOKEN]
EOS_ID = TOKEN_TO_ID[EOS_TOKEN]

FUNCTION_IDS = {TOKEN_TO_ID[t] for t in FUNCTION_TOKENS}
TERMINAL_IDS = {TOKEN_TO_ID[t] for t in VARIABLE_TOKENS + ERC_TOKENS}

ARITY = {}
for t in FUNCTION_TOKENS:
    ARITY[TOKEN_TO_ID[t]] = 2
for t in VARIABLE_TOKENS + ERC_TOKENS:
    ARITY[TOKEN_TO_ID[t]] = 0


def _snap_erc(value):
    val = round(float(value), 1)
    val = max(config.ERC_MIN, min(config.ERC_MAX, val))
    val = round(round(val / config.ERC_STEP) * config.ERC_STEP, 1)
    return str(val)


def tree_to_tokens(individual):
    tokens = []
    for node in individual:
        if isinstance(node, gp.Primitive):
            tokens.append(node.name)
        elif isinstance(node, gp.Terminal):
            if type(node).__name__ == "ERC" or node.name == "ERC":
                tokens.append(_snap_erc(node.value))
            else:
                tokens.append(node.value)
    return tokens


def encode(tokens, max_len=None):
    if max_len is None:
        max_len = config.TRANSFORMER_MAX_SEQ_LEN
    ids = [TOKEN_TO_ID.get(t, PAD_ID) for t in tokens]
    ids = ids[:max_len]
    padding = [PAD_ID] * (max_len - len(ids))
    return ids + padding


def encode_with_sos_eos(tokens, max_len=None):
    if max_len is None:
        max_len = config.TRANSFORMER_MAX_SEQ_LEN
    ids = [SOS_ID] + [TOKEN_TO_ID.get(t, PAD_ID) for t in tokens] + [EOS_ID]
    ids = ids[:max_len]
    padding = [PAD_ID] * (max_len - len(ids))
    return ids + padding


def decode(ids):
    tokens = []
    for i in ids:
        if i == EOS_ID:
            break
        if i == PAD_ID or i == SOS_ID:
            continue
        tokens.append(ID_TO_TOKEN.get(i, PAD_TOKEN))
    return tokens


def tokens_to_tree(tokens, pset):
    if not tokens:
        return None
    expr = []
    for token in tokens:
        if token in [p.name for p in pset.primitives[pset.ret]]:
            prim = next(p for p in pset.primitives[pset.ret] if p.name == token)
            expr.append(prim)
        elif token in [t.value for t in pset.terminals[pset.ret]
                       if isinstance(t, gp.Terminal) and not isinstance(t, type)]:
            term = next(t for t in pset.terminals[pset.ret]
                        if isinstance(t, gp.Terminal) and not isinstance(t, type)
                        and t.value == token)
            expr.append(term)
        else:
            try:
                val = float(token)
                val = round(val, 1)
                ephemeral = gp.Terminal(val, False, pset.ret)
                ephemeral.name = str(val)
                expr.append(ephemeral)
            except (ValueError, TypeError):
                return None
    try:
        tree = gp.PrimitiveTree(expr)
        _ = gp.compile(tree, pset)
        return tree
    except Exception:
        return None


def batch_encode_encoder(token_lists, max_len=None):
    return np.array([encode(tl, max_len) for tl in token_lists], dtype=np.int32)


def batch_encode_decoder(token_lists, max_len=None):
    return np.array([encode_with_sos_eos(tl, max_len) for tl in token_lists],
                    dtype=np.int32)
