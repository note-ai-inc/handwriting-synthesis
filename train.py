import os
import numpy as np
import tensorflow as tf

import drawing
from data_frame import DataFrame
from tf_base_model import TFBaseModel
from rnn_ops import rnn_free_run
from rnn_cell import LSTMAttentionCell
from tf_utils import time_distributed_dense_layer, shape


np.random.seed(42)  # Set Numpy seed for reproducibility
tf.set_random_seed(42) # Set TensorFlow seed for reproducibility

class StyleEncoder(tf.keras.layers.Layer):
    def __init__(self, hidden_size=128, output_size=256):
        super(StyleEncoder, self).__init__()
        self.bi_rnn = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(hidden_size, return_sequences=True)
        )
        self.attention = tf.keras.layers.Dense(1, activation='tanh')
        self.projection = tf.keras.layers.Dense(output_size)

    def call(self, stroke_sequence):
        x = self.bi_rnn(stroke_sequence)
        attn_scores = self.attention(x)  # [batch, seq_len, 1]
        attn_weights = tf.nn.softmax(attn_scores, axis=1)
        x = tf.reduce_sum(x * attn_weights, axis=1)  # [batch, 2*hidden_size]
        return self.projection(x)

class StyleAdaptiveLSTMAttentionCell(LSTMAttentionCell):
    def __init__(self, style_emb, **kwargs):
        super(StyleAdaptiveLSTMAttentionCell, self).__init__(**kwargs)
        self.style_emb = style_emb

    def __call__(self, inputs, state, scope=None):
        style_injected = tf.concat([inputs, self.style_emb], axis=1)
        return super(StyleAdaptiveLSTMAttentionCell, self).__call__(style_injected, state, scope)

class StyleSynthesisModel(TFBaseModel):
    def __init__(self, lstm_size=400, output_mixture_components=20,
                 attention_mixture_components=10, style_embedding_size=256, **kwargs):
        self.lstm_size = lstm_size
        self.output_mixture_components = output_mixture_components
        self.attention_mixture_components = attention_mixture_components
        self.style_embedding_size = style_embedding_size
        self.output_units = self.output_mixture_components * 6 + 1
        super(StyleSynthesisModel, self).__init__(**kwargs)

    def calculate_loss(self):
        self.x = tf.placeholder(tf.float32, [None, None, 3], name="x")
        self.y = tf.placeholder(tf.float32, [None, None, 3], name="y")
        self.x_len = tf.placeholder(tf.int32, [None], name="x_len")

        self.c = tf.placeholder(tf.int32, [None, None], name="c")
        self.c_len = tf.placeholder(tf.int32, [None], name="c_len")

        self.ref_x = tf.placeholder(tf.float32, [None, None, 3], name="ref_x")
        self.ref_x_len = tf.placeholder(tf.int32, [None], name="ref_x_len")

        self.sample_tsteps = tf.placeholder(tf.int32, [], name="sample_tsteps")
        self.num_samples = tf.placeholder(tf.int32, [], name="num_samples")
        self.prime = tf.placeholder_with_default(False, shape=[], name="prime")

        self.bias = tf.placeholder_with_default(
            tf.zeros([1], dtype=tf.float32), shape=[None], name="bias"
        )

        self.x_prime = tf.placeholder(tf.float32, [None, None, 3], name="x_prime")
        self.x_prime_len = tf.placeholder(tf.int32, [None], name="x_prime_len")

        style_encoder = StyleEncoder(hidden_size=128, output_size=self.style_embedding_size)
        style_emb = style_encoder(self.ref_x)

        tf.summary.histogram("style_embedding", style_emb)

        cell = StyleAdaptiveLSTMAttentionCell(
            lstm_size=self.lstm_size,
            num_attn_mixture_components=self.attention_mixture_components,
            attention_values=tf.one_hot(self.c, len(drawing.alphabet)),
            attention_values_lengths=self.c_len,
            num_output_mixture_components=self.output_mixture_components,
            bias=self.bias,
            style_emb=style_emb
        )

        initial_state = cell.zero_state(tf.shape(self.x)[0], dtype=tf.float32)
        outputs, _ = tf.nn.dynamic_rnn(
            cell=cell,
            inputs=self.x,
            sequence_length=self.x_len,
            initial_state=initial_state,
            dtype=tf.float32,
            scope="rnn_style"
        )

        params = time_distributed_dense_layer(outputs, self.output_units, scope='rnn_style/gmm')
        pis, mus, sigmas, rhos, es = self.parse_parameters(params)

        self.loss = self.mdn_loss(self.y, self.x_len, pis, mus, sigmas, rhos, es)

        tf.summary.scalar("loss", self.loss)

        self.sampled_sequence = tf.cond(
            self.prime,
            lambda: self.primed_sample(cell),
            lambda: self.unprimed_sample(cell)
        )

        return self.loss

    def parse_parameters(self, z, eps=1e-8, sigma_eps=1e-4):
        pis, sigmas, rhos, mus, es = tf.split(
            z,
            [self.output_mixture_components, 2 * self.output_mixture_components,
             self.output_mixture_components, 2 * self.output_mixture_components, 1],
            axis=-1
        )
        pis = pis * (1 + tf.expand_dims(self.bias, 1))
        sigmas = sigmas - tf.expand_dims(self.bias, 1)

        pis = tf.nn.softmax(pis, axis=-1)
        sigmas = tf.clip_by_value(tf.exp(sigmas), sigma_eps, np.inf)
        rhos = tf.clip_by_value(tf.tanh(rhos), -1 + eps, 1 - eps)
        es = tf.clip_by_value(tf.nn.sigmoid(es), eps, 1.0 - eps)
        return pis, mus, sigmas, rhos, es

    def mdn_loss(self, y, lengths, pis, mus, sigmas, rhos, es, eps=1e-8):
        sigma1, sigma2 = tf.split(sigmas, 2, axis=2)
        y1, y2, y3 = tf.split(y, 3, axis=2)
        mu1, mu2 = tf.split(mus, 2, axis=2)

        norm = 1.0 / (2.0 * np.pi * sigma1 * sigma2 * tf.sqrt(1.0 - tf.square(rhos)))
        Z = (tf.square((y1 - mu1) / sigma1) +
             tf.square((y2 - mu2) / sigma2) -
             2.0 * rhos * (y1 - mu1) * (y2 - mu2) / (sigma1 * sigma2))
        exp = -Z / (2.0 * (1.0 - tf.square(rhos)))
        gmm_likelihood = tf.reduce_sum(pis * norm * tf.exp(exp), axis=2)
        gmm_likelihood = tf.clip_by_value(gmm_likelihood, eps, np.inf)

        bernoulli_likelihood = y3 * es + (1.0 - y3) * (1.0 - es)
        bernoulli_likelihood = tf.clip_by_value(bernoulli_likelihood, eps, 1.0)
        bernoulli_likelihood = tf.squeeze(bernoulli_likelihood, axis=2)

        ll = tf.log(gmm_likelihood) + tf.log(bernoulli_likelihood)
        nll = -ll
        mask = tf.sequence_mask(lengths, maxlen=tf.shape(y)[1], dtype=tf.float32)
        nll = nll * mask
        total_frames = tf.reduce_sum(mask)

        return tf.reduce_sum(nll) / (total_frames + 1e-8)

    def unprimed_sample(self, cell):
        batch_size = self.num_samples
        initial_state = cell.zero_state(batch_size, dtype=tf.float32)
        _, outputs, _ = rnn_free_run(
            cell=cell,
            sequence_length=self.sample_tsteps,
            initial_state=initial_state,
            initial_input=None,
            scope="rnn_style"
        )
        return outputs

    def primed_sample(self, cell):
        batch_size = self.num_samples
        initial_state = cell.zero_state(batch_size, dtype=tf.float32)
        _, primed_state = tf.nn.dynamic_rnn(
            inputs=self.x_prime,
            cell=cell,
            sequence_length=self.x_prime_len,
            initial_state=initial_state,
            dtype=tf.float32,
            scope='rnn_style'
        )
        _, outputs, _ = rnn_free_run(
            cell=cell,
            sequence_length=self.sample_tsteps,
            initial_state=primed_state,
            scope="rnn_style"
        )
        return outputs

################################################################################
# Example DataReader that yields (x, x_len, c, c_len, ref_x, ref_x_len, y)
################################################################################
class StyleDataReader(object):
    def __init__(self, data_dir):
        data_cols = ["x", "x_len", "c", "c_len"]
        arrays = []
        for dc in data_cols:
            arr = np.load(os.path.join(data_dir, f"{dc}.npy"), allow_pickle=True)
            arrays.append(arr)
        self.df = DataFrame(columns=data_cols, data=arrays)
        self.train_df, self.val_df = self.df.train_test_split(train_size=0.95, random_state=42)
        print("Train size:", len(self.train_df))
        print("Val size:", len(self.val_df))

    def train_batch_generator(self, batch_size):
        return self._batch_generator(self.train_df, batch_size=batch_size, shuffle=True)

    def val_batch_generator(self, batch_size):
        return self._batch_generator(self.val_df, batch_size=batch_size, shuffle=True)

    def _batch_generator(self, df, batch_size, shuffle=True):
        gen = df.batch_generator(
            batch_size=batch_size,
            shuffle=shuffle,
            num_epochs=10000
        )
        for batch in gen:
            # slice x->y for teacher forcing
            batch["x_len"] = batch["x_len"] - 1
            max_x_len = np.max(batch["x_len"])
            max_c_len = np.max(batch["c_len"])

            # slice for teacher forcing
            batch["y"] = batch["x"][:, 1 : max_x_len + 1, :]
            batch["x"] = batch["x"][:, :max_x_len, :]
            batch["c"] = batch["c"][:, :max_c_len]

            # create some ref_x
            # e.g. simply use the first 20 frames (or random slice) as "reference"
            ref_len = np.minimum(50, max_x_len)
            batch["ref_x"] = batch["x"][:, :ref_len, :]
            batch["ref_x_len"] = np.full(batch["x"].shape[0], ref_len, dtype=np.int32)

            yield batch


################################################################################
# Visualization routine for sampling
################################################################################
def visualize_after_training(model, reader, out_dir="style_viz"):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # We'll just do a single batch of size 1:
    gen = reader.val_batch_generator(batch_size=1)
    sample_batch = next(gen)

    # We'll use sample_batch["ref_x"] as the style reference
    ref_strokes = sample_batch["ref_x"]  # shape [1, ref_len, 3]
    ref_len = sample_batch["ref_x_len"]

    # We'll pick some text to generate:
    VIS_TEXTS = [
        "Hello, world!",
        "This is a test sentence.",
        "A quick brown fox jumps over the lazy dog.",
        "Meta-learning for handwriting synthesis.",
        "Generating new text in adapted style."
    ]

    for i, sentence in enumerate(VIS_TEXTS):
        c_encoded = drawing.encode_ascii(sentence)
        c_len = np.array([len(c_encoded)], dtype=np.int32)
        # pad or shape
        c_batch = np.zeros((1, len(c_encoded)), dtype=np.int32)
        c_batch[0, :len(c_encoded)] = c_encoded

        feed_dict = {
            model.num_samples: 1,
            model.sample_tsteps: 600,  # or some large number
            model.prime: False,

            model.ref_x: ref_strokes,
            model.ref_x_len: ref_len,

            model.c: c_batch,
            model.c_len: c_len,
        }

        sampled = model.session.run(model.sampled_sequence, feed_dict=feed_dict)
        sampled = sampled[0]  # remove batch dim, shape [T, 3]

        figname = os.path.join(out_dir, f"sample_{i}.png")
        drawing.draw(sampled, ascii_seq=sentence, save_file=figname, align_strokes=False)
        print("Saved sample figure:", figname)


################################################################################
# Main
################################################################################
def main():
    data_dir = "data/processed"  # adapt to your location
    reader = StyleDataReader(data_dir)

    model = StyleSynthesisModel(
        reader=reader,
        log_dir="logs_style_synthesis",
        checkpoint_dir="checkpoints_style_synthesis",
        prediction_dir="predictions_style_synthesis",
        # TFBaseModel params
        save_all_checkpoints=True,
        learning_rates=[1e-4],
        batch_sizes=[32],
        patiences=[2000],
        beta1_decays=[0.9],
        validation_batch_size=32,
        optimizer='rms',
        num_training_steps=30000,
        regularization_constant=0.0,
        keep_prob=1.0,
        enable_parameter_averaging=False,
        min_steps_to_checkpoint=300,
        log_interval=50,
        grad_clip=50,
        # style/rnn params
        lstm_size=400,
        output_mixture_components=20,
        attention_mixture_components=10,
        style_embedding_size=256
    )

    # Train
    model.fit()

    # Visualization after training:
    visualize_after_training(model, reader)


if __name__ == "__main__":
    main()


