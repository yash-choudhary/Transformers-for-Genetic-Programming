import numpy as np

from . import config
from .tokenizer import (VOCAB_SIZE, PAD_ID, SOS_ID, EOS_ID,
                        FUNCTION_IDS, TERMINAL_IDS, ARITY, ID_TO_TOKEN)


class SyntaxController:
    """Enforces valid prefix-notation tree structure during auto-regressive sampling.

    Tracks an arity stack: each function token pushes its arity onto the stack,
    each terminal decrements the top. A tree is complete when the stack is empty.
    """

    def __init__(self, max_depth=config.GP_MAX_TREE_DEPTH,
                 max_tokens=config.TRANSFORMER_MAX_SEQ_LEN):
        self.max_depth = max_depth
        self.max_tokens = max_tokens
        self.reset()

    def reset(self):
        self.arity_stack = []
        self.tokens_generated = 0
        self.current_depth = 0
        self.complete = False

    def _compute_depth(self):
        return len(self.arity_stack)

    def get_valid_mask(self):
        mask = np.full(VOCAB_SIZE, -1e9)

        if self.complete:
            mask[EOS_ID] = 0.0
            return mask

        if self.tokens_generated == 0:
            for fid in FUNCTION_IDS:
                mask[fid] = 0.0
            for tid in TERMINAL_IDS:
                mask[tid] = 0.0
            return mask

        remaining_slots = sum(self.arity_stack) if self.arity_stack else 0
        tokens_left = self.max_tokens - self.tokens_generated - 1

        if remaining_slots == 0:
            mask[EOS_ID] = 0.0
            return mask

        at_max_depth = self._compute_depth() >= self.max_depth
        must_use_terminal = (tokens_left <= remaining_slots) or at_max_depth

        if must_use_terminal:
            for tid in TERMINAL_IDS:
                mask[tid] = 0.0
        else:
            for fid in FUNCTION_IDS:
                mask[fid] = 0.0
            for tid in TERMINAL_IDS:
                mask[tid] = 0.0

        return mask

    def update(self, token_id):
        if token_id == EOS_ID:
            self.complete = True
            return

        self.tokens_generated += 1

        if token_id in FUNCTION_IDS:
            arity = ARITY[token_id]
            self.arity_stack.append(arity)
            self.current_depth = self._compute_depth()
        elif token_id in TERMINAL_IDS:
            if self.arity_stack:
                self.arity_stack[-1] -= 1
                while self.arity_stack and self.arity_stack[-1] == 0:
                    self.arity_stack.pop()
                    if self.arity_stack:
                        self.arity_stack[-1] -= 1

        if not self.arity_stack and self.tokens_generated > 0:
            self.complete = True

    def is_complete(self):
        return self.complete


def apply_syntax_mask(logits, syntax_controller):
    mask = syntax_controller.get_valid_mask()
    return logits + mask
