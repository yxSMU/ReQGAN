# train_pqwgan_tf_tc_prob28_uniform.py
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"
import argparse
import numpy as np
import tensorflow as tf
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

import tensorcircuit as tc

tc.set_backend("tensorflow")

from utils.dataset_gpu import load_mnist_32_intensity, select_from_dataset
from models.pqwgan_patch1_32 import PQWGAN_CC


def setup_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("[Warn] No GPU found. Running on CPU.")
        return
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"[OK] Using GPU(s): {gpus}")


def denorm_to_torch_32(x_tf_flat: tf.Tensor) -> torch.Tensor:
    """ """
    x = tf.reshape(x_tf_flat, (-1, 1, 32, 32))
    x = (x + 1.0) / 2.0
    x = tf.clip_by_value(x, 0.0, 1.0)
    return torch.from_numpy(x.numpy())


def compute_gradient_penalty_tf(critic, real_x, fake_x):
    """
    real_x, fake_x: [B,256]
    critic(x): [B,1]
    """
    bs = tf.shape(real_x)[0]
    alpha = tf.random.uniform([bs, 1], 0.0, 1.0, dtype=tf.float32)
    inter = alpha * real_x + (1.0 - alpha) * fake_x

    with tf.GradientTape() as tape:
        tape.watch(inter)
        v = critic(inter, training=True)

    grads = tape.gradient(v, inter)  # [B,784]
    grads = tf.reshape(grads, [bs, -1])
    slopes = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1) + 1e-12)
    gp = tf.reduce_mean(tf.square(slopes - 1.0))
    return gp


def train(
    classes_str: str,
    n_layers: int,
    n_data_qubits: int,
    batch_size: int,
    out_dir: str,
    per_class_size: int = 1000,
    n_epochs: int = 50,
    ancilla_lowest_bit: bool = False,
):
    """ """
    setup_gpu()
    os.makedirs(out_dir, exist_ok=True)

    classes = sorted(list(set(int(d) for d in classes_str.split(","))))
    output_dim = 32 * 32

    lr_D = 0.0002
    lr_G = 0.01
    b1, b2 = 0.0, 0.9
    lambda_gp = 10.0
    n_critic = 5
    sample_interval = 10

    n_ancillas = 1
    n_qubits = n_data_qubits + n_ancillas
    latent_dim = n_qubits

    dataset = select_from_dataset(load_mnist_32_intensity(), per_class_size, classes)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )

    # Model
    gan = PQWGAN_CC(
        image_size=32,
        channels=1,
        n_data_qubits=n_data_qubits,
        n_ancillas=n_ancillas,
        n_layers=n_layers,
    )
    critic = gan.critic
    generator = gan.generator

    # Build variables
    _ = critic(tf.zeros((1, output_dim), dtype=tf.float32))
    _ = generator(
        tf.zeros((1, latent_dim), dtype=tf.float32),
        training=False,
        ancilla_lowest_bit=ancilla_lowest_bit,
    )

    opt_C = tf.keras.optimizers.Adam(learning_rate=lr_D, beta_1=b1, beta_2=b2)
    opt_G = tf.keras.optimizers.Adam(learning_rate=lr_G, beta_1=b1, beta_2=b2)

    @tf.function
    def critic_step(real_x):
        bs = tf.shape(real_x)[0]
        with tf.GradientTape() as tape:
            z = tf.random.uniform((bs, latent_dim), 0.0, 1.0, dtype=tf.float32)
            fake_x = generator(z, training=True, ancilla_lowest_bit=ancilla_lowest_bit)

            real_v = critic(real_x, training=True)
            fake_v = critic(fake_x, training=True)

            gp = compute_gradient_penalty_tf(critic, real_x, fake_x)
            d_loss = -tf.reduce_mean(real_v) + tf.reduce_mean(fake_v) + lambda_gp * gp

        grads = tape.gradient(d_loss, critic.trainable_variables)
        opt_C.apply_gradients(zip(grads, critic.trainable_variables))

        wdist = tf.reduce_mean(real_v) - tf.reduce_mean(fake_v)
        return d_loss, wdist

    @tf.function
    def generator_step(bs):
        with tf.GradientTape() as tape:
            z = tf.random.uniform((bs, latent_dim), 0.0, 1.0, dtype=tf.float32)
            fake_x = generator(z, training=True, ancilla_lowest_bit=ancilla_lowest_bit)
            fake_v = critic(fake_x, training=True)
            g_loss = -tf.reduce_mean(fake_v)

        grads = tape.gradient(g_loss, generator.trainable_variables)
        opt_G.apply_gradients(zip(grads, generator.trainable_variables))
        return g_loss

    fixed_z = tf.random.uniform((batch_size, latent_dim), 0.0, 1.0, dtype=tf.float32)

    wasserstein_distance_history = []
    saved_initial = False
    batches_done = 0

    for epoch in range(n_epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{n_epochs}")
        for i, (real_x_pt, _) in enumerate(pbar):
            real_x = tf.convert_to_tensor(real_x_pt.numpy(), dtype=tf.float32)
            bs = tf.shape(real_x)[0]

            if not saved_initial:
                fixed_images = generator(
                    fixed_z[:bs], training=False, ancilla_lowest_bit=ancilla_lowest_bit
                )
                save_image(
                    denorm_to_torch_32(fixed_images),
                    os.path.join(out_dir, f"{batches_done}.png"),
                    nrow=5,
                )
                save_image(
                    denorm_to_torch_32(real_x),
                    os.path.join(out_dir, "real_samples.png"),
                    nrow=5,
                )
                saved_initial = True

            d_loss, wdist = critic_step(real_x)
            wasserstein_distance_history.append(float(wdist.numpy()))

            g_loss_val = None
            if (epoch * len(dataloader) + i + 1) % n_critic == 0:
                g_loss_val = generator_step(bs)

                batches_done += n_critic
                np.save(
                    os.path.join(out_dir, "wasserstein_distance.npy"),
                    np.array(wasserstein_distance_history, dtype=np.float32),
                )

                if batches_done % sample_interval == 0:
                    fixed_images = generator(
                        fixed_z[:bs],
                        training=False,
                        ancilla_lowest_bit=ancilla_lowest_bit,
                    )
                    save_image(
                        denorm_to_torch_32(fixed_images),
                        os.path.join(out_dir, f"{batches_done}.png"),
                        nrow=5,
                    )
                    critic.save_weights(
                        os.path.join(out_dir, f"critic-{batches_done}.weights.h5")
                    )
                    generator.save_weights(
                        os.path.join(out_dir, f"generator-{batches_done}.weights.h5")
                    )

            if g_loss_val is None:
                pbar.set_postfix(
                    {"D": float(d_loss.numpy()), "W": float(wdist.numpy())}
                )
            else:
                pbar.set_postfix(
                    {
                        "D": float(d_loss.numpy()),
                        "G": float(g_loss_val.numpy()),
                        "W": float(wdist.numpy()),
                    }
                )

    critic.save_weights(os.path.join(out_dir, "critic-final.weights.h5"))
    generator.save_weights(os.path.join(out_dir, "generator-final.weights.h5"))

    import json

    config = {
        "image_size": 32,
        "channels": 1,
        "n_data_qubits": n_data_qubits,
        "n_ancillas": n_ancillas,
        "n_layers": n_layers,
        "latent_dim": latent_dim,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "classes": classes,
        "ancilla_lowest_bit": ancilla_lowest_bit,
        "prior": "uniform_0_1",
        "variant": "pqwgan_patch1_32",
    }
    with open(os.path.join(out_dir, "model_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    np.save(
        os.path.join(out_dir, "wasserstein_distance_final.npy"),
        np.array(wasserstein_distance_history, dtype=np.float32),
    )

    print("Done")
    print(f"Model weights saved to: {out_dir}")
    print(f"  - critic-final.weights.h5")
    print(f"  - generator-final.weights.h5")
    print(f"  - model_config.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", type=str, default="0", help="e.g. '0' or '0,1,2'")
    parser.add_argument("--layers", type=int, default=20)
    parser.add_argument(
        "--n_data_qubits",
        type=int,
        default=10,
        help="32x32: 10 data qubits + 1 ancilla = 11 total",
    )
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--out", type=str, default="./pqwgan_32x32/cc_0")
    parser.add_argument("--per_class_size", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--ancilla_lowest_bit",
        action="store_true",
        help="This implementation requires one ancilla qubit.",
    )
    args = parser.parse_args()

    train(
        classes_str=args.classes,
        n_layers=args.layers,
        n_data_qubits=args.n_data_qubits,
        batch_size=args.batch_size,
        out_dir=args.out,
        per_class_size=args.per_class_size,
        n_epochs=args.epochs,
        ancilla_lowest_bit=args.ancilla_lowest_bit,
    )
