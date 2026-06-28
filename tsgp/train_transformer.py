import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

from . import config
from .tokenizer import (batch_encode_encoder, batch_encode_decoder, PAD_ID)
from .transformer_model import create_model


def load_training_data(data_path):
    with open(data_path, "rb") as f:
        training_data = pickle.load(f)
    return training_data


def prepare_numpy_data(training_data):
    input_tokens = [d["input_tokens"] for d in training_data]
    output_tokens = [d["output_tokens"] for d in training_data]
    sd_values = np.array([d["sd"] for d in training_data], dtype=np.float32)

    enc_input = batch_encode_encoder(input_tokens)
    dec_full = batch_encode_decoder(output_tokens)
    dec_input = dec_full[:, :-1]
    dec_target = dec_full[:, 1:]

    return enc_input, dec_input, sd_values, dec_target


class TSGPTrainingModel(keras.Model):
    def __init__(self, tsgp_model, **kwargs):
        super().__init__(**kwargs)
        self.tsgp_model = tsgp_model

    def call(self, inputs, training=False):
        return self.tsgp_model(inputs, training=training)


def train_model(training_data, checkpoint_dir="checkpoints",
                epochs=None, batch_size=None, verbose=True):
    if epochs is None:
        epochs = config.TRANSFORMER_EPOCHS
    if batch_size is None:
        batch_size = config.TRANSFORMER_BATCH_SIZE

    os.makedirs(checkpoint_dir, exist_ok=True)

    base_model = create_model()
    wrapper = TSGPTrainingModel(base_model)

    optimizer = keras.optimizers.legacy.Adam(
        learning_rate=config.TRANSFORMER_LEARNING_RATE)

    def masked_loss(y_true, y_pred):
        loss_fn = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction="none")
        loss = loss_fn(y_true, y_pred)
        mask = tf.cast(tf.not_equal(y_true, PAD_ID), tf.float32)
        return tf.reduce_sum(loss * mask) / (tf.reduce_sum(mask) + 1e-8)

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

    for epoch in range(1, epochs + 1):
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

        base_model.save_weights(
            os.path.join(checkpoint_dir, f"tsgp_epoch_{epoch}.weights.h5"))

    base_model.save_weights(
        os.path.join(checkpoint_dir, "tsgp_final.weights.h5"))
    return base_model


def train_from_data(data_path, checkpoint_dir="checkpoints", verbose=True):
    if verbose:
        print("Loading training data...")
    training_data = load_training_data(data_path)
    if verbose:
        print(f"Loaded {len(training_data)} training pairs")
        print("Starting training...")

    model = train_model(training_data, checkpoint_dir=checkpoint_dir,
                        verbose=verbose)
    return model


if __name__ == "__main__":
    train_from_data("data/training/training_pairs.pkl")
