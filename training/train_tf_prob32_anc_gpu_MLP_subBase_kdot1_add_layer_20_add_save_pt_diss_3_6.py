# train_tf_prob32_anc_gpu_MLP.py
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import argparse
import tensorflow as tf
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import tensorcircuit as tc

tc.set_backend("tensorflow")

from utils.wgan_gpu import compute_gradient_penalty
from utils.dataset_gpu import load_mnist_32_intensity, select_from_dataset

from models.size32_anc_reupload_tf_prob_gpu_MLP_subBase33_kdot1_noLayerNorm import (
    PQWGAN_CC,
)


def setup_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("[Warn] No GPU found. Running on CPU.")
        return
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"[OK] Using GPU(s): {gpus}")


def x_to_vis_images_center28(x_tf, image_size=32):
    """
    x_tf: [B, 1024] in [-1, 1]
    return: torch [B, 1, 28, 28] in [0, 1]
    """
    x = tf.reshape(x_tf, (-1, 1, image_size, image_size))
    x = (x + 1.0) / 2.0
    x = tf.clip_by_value(x, 0.0, 1.0)
    x = x[:, :, 2:30, 2:30]  # center crop 28x28
    return torch.from_numpy(x.numpy())


def _save_heatmap(img2d, title, path):
    plt.figure()
    plt.imshow(img2d, aspect="equal")
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def train(
    classes_str,
    n_layers,
    n_data_qubits,
    latent_dim,
    batch_size,
    out_dir,
    n_epochs=50,
    per_class_size=1000,
    num_workers=0,
    checkpoint_every=5,
):
    setup_gpu()
    os.makedirs(out_dir, exist_ok=True)

    classes = sorted(list(set(int(d) for d in classes_str.split(","))))
    image_size = 32
    channels = 1

    # Hyperparams
    lr_D = 2e-4
    lr_angle = 2e-4
    lr_qpqc = 1e-2
    b1, b2 = 0.0, 0.9

    lambda_gp = 10.0
    n_critic = 5
    save_every_batches = checkpoint_every

    # Dataset
    dataset = select_from_dataset(load_mnist_32_intensity(), per_class_size, classes)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
    )

    # Model
    gan = PQWGAN_CC(
        image_size=image_size,
        channels=channels,
        n_data_qubits=n_data_qubits,
        n_layers=n_layers,
        latent_dim=latent_dim,
    )
    critic = gan.critic
    generator = gan.generator

    # Optims
    opt_C = tf.keras.optimizers.Adam(lr_D, b1, b2)
    opt_G_angle = tf.keras.optimizers.Adam(lr_angle, b1, b2)
    opt_G_qpqc = tf.keras.optimizers.Adam(lr_qpqc, b1, b2)

    # Build (force variable creation)
    _ = critic(tf.zeros((1, image_size * image_size), dtype=tf.float32))
    _ = generator(tf.zeros((1, latent_dim), dtype=tf.float32), training=False)

    # Trainable var groups
    angle_vars = generator.angle_net.trainable_variables
    qpqc_vars = generator.tc_circuit_layer.trainable_variables

    N_ANGLE = len(angle_vars)

    @tf.function
    def critic_step(real_x):
        bs = tf.shape(real_x)[0]
        with tf.GradientTape() as tape:
            z = tf.random.uniform(
                (bs, latent_dim), minval=-np.pi, maxval=np.pi, dtype=tf.float32
            )
            fake_x = generator(z, training=True)

            real_v = critic(real_x, training=True)
            fake_v = critic(fake_x, training=True)

            gp = compute_gradient_penalty(critic, real_x, fake_x)
            loss = -tf.reduce_mean(real_v) + tf.reduce_mean(fake_v) + lambda_gp * gp

        grads = tape.gradient(loss, critic.trainable_variables)
        opt_C.apply_gradients(zip(grads, critic.trainable_variables))
        grad_norm = tf.sqrt(
            sum([tf.reduce_sum(tf.square(g)) for g in grads if g is not None])
        )
        wasserstein = tf.reduce_mean(real_v) - tf.reduce_mean(fake_v)
        return loss, grad_norm, wasserstein

    @tf.function
    def generator_step(bs):

        with tf.GradientTape() as tape:
            z = tf.random.uniform(
                (bs, latent_dim), minval=-np.pi, maxval=np.pi, dtype=tf.float32
            )
            fake_x = generator(z, training=True)
            fake_v = critic(fake_x, training=True)
            loss = -tf.reduce_mean(fake_v)

        all_vars = angle_vars + qpqc_vars
        grads = tape.gradient(loss, all_vars)

        opt_G_angle.apply_gradients(zip(grads[:N_ANGLE], angle_vars))
        opt_G_qpqc.apply_gradients(zip(grads[N_ANGLE:], qpqc_vars))
        grad_norm = tf.sqrt(
            sum([tf.reduce_sum(tf.square(g)) for g in grads if g is not None])
        )
        return loss, grad_norm

    global_batch = 0
    saved_real = False

    d_loss_history = []
    g_loss_history = []
    batch_indices = []
    d_grad_norms = []
    g_grad_norms = []
    wasserstein_history = []

    for epoch in range(n_epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{n_epochs}")
        for real_x_pt, _ in pbar:
            global_batch += 1
            real_x = tf.convert_to_tensor(real_x_pt.numpy(), dtype=tf.float32)
            bs = tf.shape(real_x)[0]

            if not saved_real:
                save_image(
                    x_to_vis_images_center28(real_x, image_size=image_size),
                    os.path.join(out_dir, "real.png"),
                    nrow=5,
                )
                saved_real = True

            d_loss, d_grad_norm, wasserstein = critic_step(real_x)

            if global_batch % n_critic == 0:
                g_loss, g_grad_norm = generator_step(bs)
            else:
                g_loss, g_grad_norm = None, None

            if global_batch % 20 == 0:
                d_loss_history.append(float(d_loss))
                batch_indices.append(global_batch)
                wasserstein_history.append(float(wasserstein))
                if g_loss is not None:
                    g_loss_history.append(float(g_loss))
                    d_grad_norms.append(float(d_grad_norm))
                    g_grad_norms.append(float(g_grad_norm))

                window = 10

                d_losses_arr = np.array(d_loss_history)
                g_losses_arr = np.array(g_loss_history)
                w_dist_arr = np.array(wasserstein_history)
                indices_arr = np.array(batch_indices)

                pad_len = window // 2
                d_losses_padded = np.pad(d_losses_arr, (pad_len, pad_len), mode="edge")
                g_losses_padded = np.pad(g_losses_arr, (pad_len, pad_len), mode="edge")
                w_dist_padded = np.pad(w_dist_arr, (pad_len, pad_len), mode="edge")

                d_mean = np.convolve(
                    d_losses_padded, np.ones(window) / window, mode="same"
                )[pad_len:-pad_len]
                d_std = np.array(
                    [
                        d_losses_arr[
                            max(0, i - window // 2) : min(
                                len(d_losses_arr), i + window // 2 + 1
                            )
                        ].std()
                        for i in range(len(d_losses_arr))
                    ]
                )

                g_mean = np.convolve(
                    g_losses_padded, np.ones(window) / window, mode="same"
                )[pad_len:-pad_len]
                g_std = np.array(
                    [
                        g_losses_arr[
                            max(0, i - window // 2) : min(
                                len(g_losses_arr), i + window // 2 + 1
                            )
                        ].std()
                        for i in range(len(g_losses_arr))
                    ]
                )

                w_mean = np.convolve(
                    w_dist_padded, np.ones(window) / window, mode="same"
                )[pad_len:-pad_len]
                w_std = np.array(
                    [
                        w_dist_arr[
                            max(0, i - window // 2) : min(
                                len(w_dist_arr), i + window // 2 + 1
                            )
                        ].std()
                        for i in range(len(w_dist_arr))
                    ]
                )

                fig, axes = plt.subplots(1, 3, figsize=(18, 5))

                axes[0].plot(indices_arr, g_losses_arr, "r-", alpha=0.6, linewidth=1.0)
                axes[0].fill_between(
                    indices_arr, g_mean - g_std, g_mean + g_std, color="red", alpha=0.25
                )
                axes[0].set_title("Generator Loss", fontsize=14)
                axes[0].set_xlabel("Batch")
                axes[0].set_ylabel("Loss")
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)

                axes[1].plot(indices_arr, d_losses_arr, "b-", alpha=0.6, linewidth=1.0)
                axes[1].fill_between(
                    indices_arr,
                    d_mean - d_std,
                    d_mean + d_std,
                    color="blue",
                    alpha=0.25,
                )
                axes[1].set_title("Discriminator Loss", fontsize=14)
                axes[1].set_xlabel("Batch")
                axes[1].set_ylabel("Loss")
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)

                axes[2].plot(indices_arr, w_dist_arr, "g-", alpha=0.6, linewidth=1.0)
                axes[2].fill_between(
                    indices_arr,
                    w_mean - w_std,
                    w_mean + w_std,
                    color="green",
                    alpha=0.25,
                )
                axes[2].set_title("Wasserstein Distance", fontsize=14)
                axes[2].set_xlabel("Batch")
                axes[2].set_ylabel("Distance")
                axes[2].legend()
                axes[2].grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig(
                    os.path.join(out_dir, f"loss_batch_{global_batch:06d}.png"),
                    dpi=150,
                    bbox_inches="tight",
                )
                plt.close()

                if g_loss is not None:
                    z_vis = tf.random.uniform(
                        (bs, latent_dim), minval=-np.pi, maxval=np.pi, dtype=tf.float32
                    )
                    fake_vis = generator(z_vis, training=False)

                    with tf.GradientTape() as tape:
                        tape.watch(fake_vis)
                        fake_v = critic(fake_vis, training=True)
                        g_loss_for_grad = -tf.reduce_mean(fake_v)

                    grads = tape.gradient(g_loss_for_grad, fake_vis)
                    grad_abs = tf.abs(grads)

                    with tf.GradientTape() as tape_d:
                        tape_d.watch(real_x)
                        real_v = critic(real_x, training=True)

                    grads_d = tape_d.gradient(real_v, real_x)
                    grad_abs_d = tf.abs(grads_d)

                    save_idx = min(bs, 4)
                    for j in range(save_idx):
                        img = (
                            x_to_vis_images_center28(
                                fake_vis[j : j + 1], image_size=image_size
                            )
                            .squeeze(0)
                            .squeeze(0)
                            .numpy()
                        )
                        _save_heatmap(
                            img,
                            f"Generator Image {j} @ batch {global_batch}",
                            os.path.join(
                                out_dir, f"gen_img_{global_batch:06d}_{j}.png"
                            ),
                        )

                        grad_map = (
                            grad_abs[j, : image_size * image_size]
                            .numpy()
                            .reshape(image_size, image_size)
                        )
                        _save_heatmap(
                            grad_map,
                            f"|grad| Generator {j} @ batch {global_batch}",
                            os.path.join(
                                out_dir, f"gen_grad_{global_batch:06d}_{j}.png"
                            ),
                        )

                        grad_map_d = (
                            grad_abs_d[j, : image_size * image_size]
                            .numpy()
                            .reshape(image_size, image_size)
                        )
                        _save_heatmap(
                            grad_map_d,
                            f"|grad| Discriminator {j} @ batch {global_batch}",
                            os.path.join(
                                out_dir, f"disc_grad_{global_batch:06d}_{j}.png"
                            ),
                        )
                    print(f"[Visual] Heatmaps saved at batch {global_batch}")

            if global_batch % save_every_batches == 0:
                z = tf.random.uniform(
                    (bs, latent_dim), minval=-np.pi, maxval=np.pi, dtype=tf.float32
                )
                fake = generator(z, training=False)
                save_image(
                    x_to_vis_images_center28(fake, image_size=image_size),
                    os.path.join(out_dir, f"batch_{global_batch:06d}.png"),
                    nrow=5,
                )

            if global_batch % checkpoint_every == 0:
                pt_path = os.path.join(out_dir, f"weights_batch_{global_batch:06d}.pt")
                torch.save(
                    {
                        "generator_weights": generator.get_weights(),
                        "angle_net_weights": generator.angle_net.get_weights(),
                    },
                    pt_path,
                )
                print(f"[Save] Weights saved to {pt_path}")

            if g_loss is None:
                pbar.set_postfix({"D": float(d_loss), "D_grad": float(d_grad_norm)})
            else:
                pbar.set_postfix(
                    {
                        "D": float(d_loss),
                        "G": float(g_loss),
                        "D_grad": float(d_grad_norm),
                        "G_grad": float(g_grad_norm),
                    }
                )

        generator.save_weights(os.path.join(out_dir, f"gen_epoch_{epoch}.weights.h5"))

        np.save(
            os.path.join(out_dir, "d_loss_history.npy"),
            np.array(d_loss_history, dtype=np.float32),
        )
        np.save(
            os.path.join(out_dir, "g_loss_history.npy"),
            np.array(g_loss_history, dtype=np.float32),
        )
        np.save(
            os.path.join(out_dir, "batch_indices.npy"),
            np.array(batch_indices, dtype=np.int32),
        )
        np.save(
            os.path.join(out_dir, "d_grad_norms.npy"),
            np.array(d_grad_norms, dtype=np.float32),
        )
        np.save(
            os.path.join(out_dir, "g_grad_norms.npy"),
            np.array(g_grad_norms, dtype=np.float32),
        )
        np.save(
            os.path.join(out_dir, "wasserstein_history.npy"),
            np.array(wasserstein_history, dtype=np.float32),
        )

    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", default="6")
    parser.add_argument("--layers", type=int, default=20)
    parser.add_argument("--n_data_qubits", type=int, default=10)
    parser.add_argument("--latent_dim", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--per_class_size", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--checkpoint_every", type=int, default=5)
    parser.add_argument(
        "--out",
        default="./chy3_0_diss_6/WO3_1A_out_angle_aux_subBase3_layer20_flatProb2_removeUcenter_noSmooth1.8_temp_sb1_3class",
    )
    args = parser.parse_args()

    train(
        args.classes,
        args.layers,
        args.n_data_qubits,
        args.latent_dim,
        args.batch_size,
        args.out,
        n_epochs=args.epochs,
        per_class_size=args.per_class_size,
        num_workers=args.num_workers,
        checkpoint_every=args.checkpoint_every,
    )


import tensorflow as tf

print("TF:", tf.__version__)
print("Built with CUDA:", tf.test.is_built_with_cuda())
print("GPUs:", tf.config.list_physical_devices("GPU"))

