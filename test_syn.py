import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import drawing
from train import StyleSynthesisModel, StyleDataReader

import random

# Set seeds for full determinism
random.seed(42)
np.random.seed(42)
tf.set_random_seed(42)

def test_model():
    data_dir = "data/processeed"
    reader = StyleDataReader(data_dir)

    model = StyleSynthesisModel(
        reader=reader,
        log_dir="logs_style_synthesis",
        checkpoint_dir="checkpoints_style_synthesis",
        prediction_dir="predictions_style_synthesis",
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
        min_steps_to_checkpoint=2000,
        log_interval=50,
        grad_clip=10,
        lstm_size=400,
        output_mixture_components=20,
        attention_mixture_components=10,
        style_embedding_size=256
    )

    checkpoint_path = "final_checkpoints/model-10350"
    model.saver.restore(model.session, checkpoint_path)

    model.sampling_mode = "deterministic"
    model.force_deterministic_sampling = True
    model.temperature = 0.5

    style_id = 3
    style_strokes = np.load(f"styles/style-{style_id}-strokes.npy")
    style_chars = np.load(f"styles/style-{style_id}-chars.npy").tostring().decode('utf-8')
    # style_strokes = np.load(f"my-style-downsampled-700.npy")
    # style_chars = np.load(f"/media/hassan/New Volume1/remo_work/Ink_Research _work/nomi_work_today/handwriting-synthesis/data/my_handwriting/sample-32-chars.npy").tostring().decode('utf-8')
    print("style_strokes  ; ", style_strokes.shape)
    print("style_chars  ; ", style_chars)

    test_sentences = [

        "The quick brown fox",
        # "thought that vengeance",
        "The quick brown fox",
        "The quick brown fox",
    ]

    num_samples = len(test_sentences)
    max_tsteps = 40 * max([len(s) for s in test_sentences])
    biases = [0.75 for _ in test_sentences]

    x_prime = np.zeros([num_samples, 1200, 3])
    x_prime_len = np.zeros([num_samples])
    chars = np.zeros([num_samples, 120])
    chars_len = np.zeros([num_samples])

    for i, sentence in enumerate(test_sentences):
        full_char_seq = drawing.encode_ascii(style_chars + ' ' + sentence)
        chars[i, :len(full_char_seq)] = full_char_seq
        chars_len[i] = len(full_char_seq)

        x_prime[i, :len(style_strokes), :] = style_strokes
        x_prime_len[i] = len(style_strokes)

    [samples] = model.session.run(
        [model.sampled_sequence],
        feed_dict={
            model.prime: True,
            model.x_prime: x_prime,
            model.x_prime_len: x_prime_len,
            model.ref_x: x_prime,
            model.ref_x_len: x_prime_len,
            model.num_samples: num_samples,
            model.sample_tsteps: max_tsteps,
            model.c: chars,
            model.c_len: chars_len,
            model.bias: biases
        }
    )

    os.makedirs("test_outputs", exist_ok=True)
    print("((((Sameple  : ))))", samples.shape)
    for idx, sample in enumerate(samples):
        sample = sample[~np.all(sample == 0.0, axis=1)]
        ref_coords = drawing.offsets_to_coords(style_strokes)
        ref_coords = drawing.denoise(ref_coords)
        ref_coords[:, :2] = drawing.align(ref_coords[:, :2])

        gen_coords = drawing.offsets_to_coords(sample)
        gen_coords = drawing.denoise(gen_coords)
        gen_coords[:, :2] = drawing.align(gen_coords[:, :2])

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        axes[0].set_title("Reference Style")
        stroke = []
        for x, y, eos in ref_coords:
            stroke.append((x, y))
            if eos == 1 and stroke:
                xs, ys = zip(*stroke)
                axes[0].plot(xs, ys, 'k')
                stroke = []
        if stroke:
            xs, ys = zip(*stroke)
            axes[0].plot(xs, ys, 'k')

        axes[1].set_title(f"Generated: \"{test_sentences[idx]}\"")
        stroke = []
        for x, y, eos in gen_coords:
            stroke.append((x, y))
            if eos == 1 and stroke:
                xs, ys = zip(*stroke)
                axes[1].plot(xs, ys, 'k')
                stroke = []
        if stroke:
            xs, ys = zip(*stroke)
            axes[1].plot(xs, ys, 'k')

        for ax in axes:
            ax.set_aspect('equal')
            ax.axis('off')

        plt.tight_layout()
        out_file = f"test_outputs/side_by_side_{idx}.png"
        plt.savefig(out_file, dpi=150)
        plt.close()
        print(f"[✓] Saved side-by-side figure to {out_file}")

if __name__ == "__main__":
    test_model()