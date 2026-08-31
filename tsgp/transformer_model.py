import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

from . import config
from .tokenizer import VOCAB_SIZE, PAD_ID


class PositionalEncoding(layers.Layer):
    def __init__(self, max_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model
        positions = np.arange(max_len)[:, np.newaxis]
        dims = np.arange(d_model)[np.newaxis, :]
        angles = positions / np.power(10000, (2 * (dims // 2)) / d_model)
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        self.pe = tf.constant(angles[np.newaxis, :, :].astype(np.float32))

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pe[:, :seq_len, :]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"max_len": self.max_len, "d_model": self.d_model})
        return cfg


class TransformerEncoderBlock(layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads)
        self.ffn = keras.Sequential([
            layers.Dense(dff, activation="relu"),
            layers.Dense(d_model),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, x, padding_mask=None, training=False):
        attn_output = self.mha(x, x, x, attention_mask=padding_mask,
                               training=training)
        attn_output = self.dropout1(attn_output, training=training)
        x = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(x)
        ffn_output = self.dropout2(ffn_output, training=training)
        x = self.layernorm2(x + ffn_output)
        return x


class TransformerDecoderBlock(layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.mha1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads)
        self.mha2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads)
        self.ffn = keras.Sequential([
            layers.Dense(dff, activation="relu"),
            layers.Dense(d_model),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
        self.dropout3 = layers.Dropout(dropout_rate)

    def call(self, x, enc_output, look_ahead_mask=None,
             padding_mask=None, training=False):
        attn1 = self.mha1(x, x, x, attention_mask=look_ahead_mask,
                          training=training)
        attn1 = self.dropout1(attn1, training=training)
        x = self.layernorm1(x + attn1)

        attn2 = self.mha2(x, enc_output, enc_output,
                          attention_mask=padding_mask, training=training)
        attn2 = self.dropout2(attn2, training=training)
        x = self.layernorm2(x + attn2)

        ffn_output = self.ffn(x)
        ffn_output = self.dropout3(ffn_output, training=training)
        x = self.layernorm3(x + ffn_output)
        return x


class TSGPTransformer(keras.Model):
    def __init__(self,
                 vocab_size=VOCAB_SIZE,
                 d_model=config.TRANSFORMER_HIDDEN_DIM,
                 num_heads=config.TRANSFORMER_NUM_HEADS,
                 num_encoder_layers=config.TRANSFORMER_NUM_ENCODER_LAYERS,
                 num_decoder_layers=config.TRANSFORMER_NUM_DECODER_LAYERS,
                 dff=None,
                 max_seq_len=config.TRANSFORMER_MAX_SEQ_LEN,
                 dropout_rate=config.TRANSFORMER_DROPOUT,
                 normalize_sd=None,
                 **kwargs):
        super().__init__(**kwargs)
        if dff is None:
            dff = d_model * 4
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        # Input transform only -- adds no weights, so checkpoints stay
        # interchangeable between the two variants. It must nevertheless match
        # how the model was trained, hence it is explicit rather than implicit.
        self.normalize_sd = (config.TRANSFORMER_SD_NORMALIZE
                             if normalize_sd is None else normalize_sd)

        self.enc_embedding = layers.Embedding(vocab_size, d_model)
        self.dec_embedding = layers.Embedding(vocab_size, d_model)
        self.enc_pos_encoding = PositionalEncoding(max_seq_len, d_model)
        self.dec_pos_encoding = PositionalEncoding(max_seq_len, d_model)

        self.sd_projection = layers.Dense(d_model)

        self.enc_dropout = layers.Dropout(dropout_rate)
        self.dec_dropout = layers.Dropout(dropout_rate)

        self.encoder_layers_list = [
            TransformerEncoderBlock(d_model, num_heads, dff, dropout_rate)
            for _ in range(num_encoder_layers)
        ]
        self.decoder_layers_list = [
            TransformerDecoderBlock(d_model, num_heads, dff, dropout_rate)
            for _ in range(num_decoder_layers)
        ]

        self.final_layer = layers.Dense(vocab_size)

    def _create_padding_mask(self, seq):
        mask = tf.cast(tf.not_equal(seq, PAD_ID), tf.float32)
        return tf.expand_dims(tf.expand_dims(mask, 1), 1)

    def _prep_sd(self, sd_input):
        """Optionally map raw SD onto a standardised log scale."""
        if not self.normalize_sd:
            return sd_input
        z = tf.math.log1p(tf.maximum(sd_input, 0.0))
        return (z - config.TRANSFORMER_SD_LOG_MEAN) / config.TRANSFORMER_SD_LOG_STD

    def encode(self, enc_input, sd_input, training=False):
        enc_padding_mask = self._create_padding_mask(enc_input)

        x = self.enc_embedding(enc_input)
        x = x * np.sqrt(float(self.d_model))
        x = self.enc_pos_encoding(x)

        sd_expanded = tf.expand_dims(tf.expand_dims(self._prep_sd(sd_input), -1), -1)
        sd_embed = self.sd_projection(sd_expanded)
        sd_embed = tf.broadcast_to(sd_embed, tf.shape(x))
        x = x + sd_embed

        x = self.enc_dropout(x, training=training)
        for enc_layer in self.encoder_layers_list:
            x = enc_layer(x, padding_mask=enc_padding_mask, training=training)
        return x

    def decode(self, dec_input, enc_output, enc_input, sd_input,
               training=False):
        seq_len = tf.shape(dec_input)[1]
        look_ahead = tf.linalg.band_part(
            tf.ones((seq_len, seq_len)), -1, 0)
        look_ahead = tf.reshape(look_ahead, [1, 1, seq_len, seq_len])
        dec_padding = tf.cast(tf.not_equal(dec_input, PAD_ID), tf.float32)
        dec_padding = tf.expand_dims(tf.expand_dims(dec_padding, 1), 1)
        combined_mask = look_ahead * dec_padding

        enc_padding_mask = self._create_padding_mask(enc_input)

        x = self.dec_embedding(dec_input)
        x = x * np.sqrt(float(self.d_model))
        x = self.dec_pos_encoding(x)

        sd_expanded = tf.expand_dims(tf.expand_dims(self._prep_sd(sd_input), -1), -1)
        sd_embed = self.sd_projection(sd_expanded)
        sd_embed = tf.broadcast_to(sd_embed, tf.shape(x))
        x = x + sd_embed

        x = self.dec_dropout(x, training=training)
        for dec_layer in self.decoder_layers_list:
            x = dec_layer(x, enc_output,
                          look_ahead_mask=combined_mask,
                          padding_mask=enc_padding_mask,
                          training=training)
        return x

    def call(self, inputs, training=False):
        enc_input, dec_input, sd_input = inputs

        enc_output = self.encode(enc_input, sd_input, training=training)
        dec_output = self.decode(dec_input, enc_output, enc_input, sd_input,
                                training=training)
        logits = self.final_layer(dec_output)
        return logits


def create_model(normalize_sd=None):
    model = TSGPTransformer(normalize_sd=normalize_sd)
    enc_input = np.zeros((1, config.TRANSFORMER_MAX_SEQ_LEN), dtype=np.int32)
    dec_input = np.zeros((1, config.TRANSFORMER_MAX_SEQ_LEN), dtype=np.int32)
    sd_input = np.zeros((1,), dtype=np.float32)
    model([enc_input, dec_input, sd_input], training=False)
    return model
