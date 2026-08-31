import glob
import os
import pickle
import re
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

from . import config
from .tokenizer import (batch_encode_encoder, batch_encode_decoder, PAD_ID)
from .transformer_model import create_model


def setup_gpu():
    """Enable GPU memory growth and report which devices TF sees."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU(s) found: {[g.name for g in gpus]}")
    else:
        print(
            "WARNING: No GPU detected — training on CPU.\n"
            "  Windows fix: TF >= 2.11 dropped native GPU support.\n"
            "  Install TF 2.10 (last native Windows GPU version) — see requirements-windows-gpu.txt"
        )
    return gpus


def load_training_data(data_path):
    with open(data_path, "rb") as f:
        training_data = pickle.load(f)
    return training_data


class TSGPTrainingModel(keras.Model):
    def __init__(self, tsgp_model, **kwargs):
        super().__init__(**kwargs)
        self.tsgp_model = tsgp_model

    def call(self, inputs, training=False):
        return self.tsgp_model(inputs, training=training)


def _save_checkpoint(model, path):
    """Save weights as a numpy array — cross-platform, cross-version safe."""
    np.save(path, np.array(model.get_weights(), dtype=object), allow_pickle=True)


def _load_checkpoint(model, path):
    """Load weights from .npy (new) or legacy .weights.h5 (old Mac format)."""
    if path.endswith(".npy"):
        data = np.load(path, allow_pickle=True)
        model.set_weights(list(data))
    else:
        # Legacy HDF5 format saved by tf.keras on Mac
        model.load_weights(path)


def find_latest_checkpoint(checkpoint_dir):
    """Return (path, epoch_number) for the highest-epoch checkpoint found."""
    found = []
    for pattern, regex in [
        (os.path.join(checkpoint_dir, "tsgp_epoch_*.npy"),
         r"tsgp_epoch_(\d+)\.npy$"),
        (os.path.join(checkpoint_dir, "tsgp_epoch_*.weights.h5"),
         r"tsgp_epoch_(\d+)\.weights\.h5$"),
    ]:
        for f in glob.glob(pattern):
            m = re.search(regex, f)
            if m:
                found.append((int(m.group(1)), f))
    if not found:
        return None, 0
    found.sort(key=lambda x: x[0])
    return found[-1][1], found[-1][0]


def prompt_resume(checkpoint_dir):
    """Interactive prompt to choose a resume checkpoint. Returns (start_epoch, path)."""
    latest_file, latest_epoch = find_latest_checkpoint(checkpoint_dir)
    if latest_file is None:
        return 0, None

    print(f"\nFound existing checkpoint: {latest_file} (epoch {latest_epoch})")

    all_found = []
    for pattern, regex in [
        (os.path.join(checkpoint_dir, "tsgp_epoch_*.npy"),
         r"tsgp_epoch_(\d+)\.npy$"),
        (os.path.join(checkpoint_dir, "tsgp_epoch_*.weights.h5"),
         r"tsgp_epoch_(\d+)\.weights\.h5$"),
    ]:
        for f in glob.glob(pattern):
            m = re.search(regex, f)
            if m:
                all_found.append((int(m.group(1)), f))
    all_found.sort(key=lambda x: x[0])

    print("Available checkpoints:")
    for _, f in all_found:
        print(f"  {os.path.basename(f)}")

    while True:
        choice = input(
            f"\nResume from epoch {latest_epoch}? [Y/n/epoch_number]: "
        ).strip()
        if choice == "" or choice.lower() == "y":
            return latest_epoch, latest_file
        if choice.lower() == "n":
            return 0, None
        try:
            epoch_num = int(choice)
            npy = os.path.join(checkpoint_dir, f"tsgp_epoch_{epoch_num}.npy")
            h5  = os.path.join(checkpoint_dir, f"tsgp_epoch_{epoch_num}.weights.h5")
            if os.path.exists(npy):
                return epoch_num, npy
            if os.path.exists(h5):
                return epoch_num, h5
            print(f"  Checkpoint for epoch {epoch_num} not found.")
        except ValueError:
            print("  Enter Y, n, or an epoch number.")


def make_optimizer():
    """AdamW at the paper's fixed learning rate (Sect. 4.1).

    Keras moved AdamW between namespaces across the TF versions this project
    targets -- 2.10 (the last native-Windows GPU build) exposes it under
    `optimizers.experimental`, 2.11+ promotes it to `optimizers` -- so resolve
    it rather than pinning one path.
    """
    lr = config.TRANSFORMER_LEARNING_RATE
    wd = config.TRANSFORMER_WEIGHT_DECAY

    candidates = []
    if hasattr(keras.optimizers, "AdamW"):
        candidates.append(keras.optimizers.AdamW)
    experimental = getattr(keras.optimizers, "experimental", None)
    if experimental is not None and hasattr(experimental, "AdamW"):
        candidates.append(experimental.AdamW)

    for cls in candidates:
        # TF 2.10's experimental optimizers default to jit_compile=True, which
        # routes the update through XLA. XLA then wants libdevice and ptxas
        # from a full CUDA toolkit, which the conda cudatoolkit *runtime*
        # does not ship -- so training dies at the first step. The model is
        # 934K parameters; XLA on the update step is not worth a toolchain
        # dependency, so switch it off wherever the argument exists.
        try:
            return cls(learning_rate=lr, weight_decay=wd, jit_compile=False)
        except TypeError:
            pass
        try:
            return cls(learning_rate=lr, weight_decay=wd)
        except TypeError:
            continue

    raise RuntimeError(
        "No AdamW implementation found in this Keras build; the paper's "
        "Sect. 4.1 configuration cannot be reproduced with plain Adam."
    )


def masked_loss(y_true, y_pred):
    loss_fn = keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none")
    loss = loss_fn(y_true, y_pred)
    mask = tf.cast(tf.not_equal(y_true, PAD_ID), tf.float32)
    return tf.reduce_sum(loss * mask) / (tf.reduce_sum(mask) + 1e-8)


def train_model(training_data, checkpoint_dir="checkpoints",
                epochs=None, batch_size=None, fresh=False, normalize_sd=None,
                verbose=True):
    if epochs is None:
        epochs = config.TRANSFORMER_EPOCHS
    if batch_size is None:
        batch_size = config.TRANSFORMER_BATCH_SIZE

    os.makedirs(checkpoint_dir, exist_ok=True)

    if fresh:
        start_epoch, resume_path = 0, None
        if verbose:
            print("Starting fresh (--fresh); ignoring any existing checkpoint.")
    elif not sys.stdin or not sys.stdin.isatty():
        # Unattended run: input() would block forever, so resume from the
        # latest checkpoint automatically instead of asking.
        resume_path, start_epoch = find_latest_checkpoint(checkpoint_dir)
        if verbose:
            print(f"Non-interactive session; "
                  + (f"resuming from {resume_path} (epoch {start_epoch})."
                     if resume_path else "no checkpoint found, starting fresh."))
    else:
        start_epoch, resume_path = prompt_resume(checkpoint_dir)

    base_model = create_model(normalize_sd=normalize_sd)
    if verbose:
        print(f"SD conditioning: "
              f"{'log1p + standardised' if base_model.normalize_sd else 'raw scalar'}")

    if resume_path is not None:
        _load_checkpoint(base_model, resume_path)
        if verbose:
            print(f"Loaded weights from {resume_path}")
            print(f"Resuming training from epoch {start_epoch + 1}")

    wrapper = TSGPTrainingModel(base_model)

    optimizer = make_optimizer()
    if verbose:
        print(f"Optimizer: {type(optimizer).__name__} "
              f"(lr={config.TRANSFORMER_LEARNING_RATE}, "
              f"weight_decay={config.TRANSFORMER_WEIGHT_DECAY})")

    wrapper.compile(optimizer=optimizer, loss=masked_loss)

    input_tokens = [d["input_tokens"] for d in training_data]
    output_tokens = [d["output_tokens"] for d in training_data]
    sd_values = np.array([d["sd"] for d in training_data], dtype=np.float32)

    enc_input = batch_encode_encoder(input_tokens)
    dec_full = batch_encode_decoder(output_tokens)
    dec_input = dec_full[:, :-1]
    dec_target = dec_full[:, 1:]

    if verbose:
        base_model.summary()

    for epoch in range(start_epoch + 1, epochs + 1):
        n = len(training_data)
        indices = np.random.permutation(n)
        total_loss = 0.0
        num_batches = 0

        batch_iter = range(0, n, batch_size)
        if verbose:
            batch_iter = tqdm(
                batch_iter,
                desc=f"Epoch {epoch}/{epochs}",
                unit="batch",
                total=(n + batch_size - 1) // batch_size,
            )

        for start in batch_iter:
            end = min(start + batch_size, n)
            idx = indices[start:end]

            batch_enc = enc_input[idx]
            batch_dec = dec_input[idx]
            batch_sd = sd_values[idx]
            batch_target = dec_target[idx]

            loss = wrapper.train_on_batch(
                [batch_enc, batch_dec, batch_sd],
                batch_target
            )
            total_loss += float(loss)
            num_batches += 1

            if verbose:
                batch_iter.set_postfix(loss=f"{total_loss / num_batches:.4f}")

        avg_loss = total_loss / max(num_batches, 1)
        if verbose:
            print(f"Epoch {epoch}/{epochs} — Loss: {avg_loss:.4f}")

        _save_checkpoint(base_model,
            os.path.join(checkpoint_dir, f"tsgp_epoch_{epoch}.npy"))

    _save_checkpoint(base_model,
        os.path.join(checkpoint_dir, "tsgp_final.npy"))
    return base_model


def train_from_data(data_path, checkpoint_dir="checkpoints", fresh=False,
                    batch_size=None, epochs=None, normalize_sd=None,
                    max_sd=None, verbose=True):
    setup_gpu()
    if verbose:
        print(f"Loading training data from {data_path} ...")
    training_data = load_training_data(data_path)

    if max_sd is not None:
        # DEVIATION from Sect. 4.1, which keeps every pair with
        # SD != 0 and SD < 100. Restricting the range is a diagnostic: the
        # operator is always queried at SD_d = 0.1 but trained across the whole
        # range, and the SD conditioning is measurably inert, so the model
        # samples from its marginal rather than the requested distance.
        before = len(training_data)
        training_data = [d for d in training_data if d["sd"] <= max_sd]
        if verbose:
            print(f"DEVIATION: filtered to SD <= {max_sd} -> "
                  f"{len(training_data):,} of {before:,} pairs "
                  f"({len(training_data)/before*100:.1f}%)")

    if verbose:
        bs = batch_size or config.TRANSFORMER_BATCH_SIZE
        ep = epochs or config.TRANSFORMER_EPOCHS
        steps = (len(training_data) + bs - 1) // bs
        print(f"Loaded {len(training_data):,} training pairs")
        print(f"Batch size {bs} -> {steps:,} updates/epoch, "
              f"{steps * ep:,} over {ep} epochs")
        print("Starting training...")

    model = train_model(training_data, checkpoint_dir=checkpoint_dir,
                        epochs=epochs, batch_size=batch_size, fresh=fresh,
                        normalize_sd=normalize_sd, verbose=verbose)
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train the TSGP transformer (paper Sect. 4.1).")
    parser.add_argument("--data", default="data/training/training_pairs.pkl",
                        help="Path to training_pairs.pkl")
    parser.add_argument("--checkpoints", default="checkpoints",
                        help="Directory for per-epoch weights. Point this at a "
                             "new directory to keep an existing model intact.")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore existing checkpoints and train from "
                             "scratch (no resume prompt)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help=f"Override the batch size (default "
                             f"{config.TRANSFORMER_BATCH_SIZE}). The paper "
                             f"does not state one; Keras defaults to 32, "
                             f"which is 8x more updates per epoch.")
    parser.add_argument("--epochs", type=int, default=None,
                        help=f"Override the epoch count (default "
                             f"{config.TRANSFORMER_EPOCHS})")
    parser.add_argument("--sd-normalize", action="store_true",
                        help="Feed SD as log1p + standardised instead of the "
                             "raw scalar. Must match at inference time.")
    parser.add_argument("--max-sd", type=float, default=None,
                        help="DEVIATION from Sect. 4.1: train only on pairs "
                             "with SD <= this. Diagnostic use only.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    train_from_data(args.data, checkpoint_dir=args.checkpoints,
                    fresh=args.fresh, batch_size=args.batch_size,
                    epochs=args.epochs,
                    normalize_sd=True if args.sd_normalize else None,
                    max_sd=args.max_sd, verbose=not args.quiet)
