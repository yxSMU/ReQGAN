# test_fid_kid.py
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import argparse
import tensorflow as tf
import torch
import numpy as np
from PIL import Image
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import tensorcircuit as tc

tc.set_backend("tensorflow")

from utils.dataset_gpu import load_mnist_32_intensity, select_from_dataset
from models.size32_anc_reupload_tf_prob_gpu_MLP_subBase33_kdot1 import PQWGAN_CC


def setup_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("[Warn] No GPU found. Running on CPU.")
        return
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"[OK] Using GPU(s): {gpus}")


def x_to_vis_images_center28(x_tf, image_size=32):
    x = tf.reshape(x_tf, (-1, 1, image_size, image_size))
    x = (x + 1.0) / 2.0
    x = tf.clip_by_value(x, 0.0, 1.0)
    x = x[:, :, 2:30, 2:30]
    return torch.from_numpy(x.numpy())


def calculate_fid(act1, act2):
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

    epsilon = 1e-6
    sigma1 += epsilon * np.eye(sigma1.shape[0])
    sigma2 += epsilon * np.eye(sigma2.shape[0])

    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))

    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def calculate_kid(real_features, gen_features, subset_size=1000, max_subsets=100):
    n_real = real_features.shape[0]
    n_gen = gen_features.shape[0]
    d = real_features.shape[1]

    subset_size = min(subset_size, n_real, n_gen)
    n_subsets = min(n_real // subset_size, n_gen // subset_size, max_subsets)

    kid_scores = []
    for _ in range(n_subsets):
        idx_real = np.random.choice(n_real, subset_size, replace=False)
        idx_gen = np.random.choice(n_gen, subset_size, replace=False)

        real_subset = real_features[idx_real]
        gen_subset = gen_features[idx_gen]

        kernel_real = (np.dot(real_subset, real_subset.T) / d + 1) ** 3
        kernel_gen = (np.dot(gen_subset, gen_subset.T) / d + 1) ** 3
        kernel_cross = (np.dot(real_subset, gen_subset.T) / d + 1) ** 3

        mmd = np.mean(kernel_real) + np.mean(kernel_gen) - 2 * np.mean(kernel_cross)
        kid_scores.append(mmd)

    return np.mean(kid_scores), np.std(kid_scores)


def test(
    ckpt_path,
    classes_str,
    n_layers,
    n_data_qubits,
    latent_dim,
    n_images,
    out_dir,
    n_runs=3,
    mapping="calibrated",
):
    setup_gpu()
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "real"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "fake"), exist_ok=True)

    classes = sorted(list(set(int(d) for d in classes_str.split(","))))
    image_size = 32

    gan = PQWGAN_CC(
        image_size=image_size,
        channels=1,
        n_data_qubits=n_data_qubits,
        n_layers=n_layers,
        latent_dim=latent_dim,
        mapping=mapping,
    )
    critic = gan.critic
    generator = gan.generator

    _ = critic(tf.zeros((1, image_size * image_size), dtype=tf.float32))
    _ = generator(tf.zeros((1, latent_dim), dtype=tf.float32), training=False)

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    generator.set_weights(checkpoint["generator_weights"])
    generator.angle_net.set_weights(checkpoint["angle_net_weights"])
    print(f"[Load] Weights loaded from {ckpt_path}")

    dataset = select_from_dataset(
        load_mnist_32_intensity(), n_images, classes, start_per_class=1000
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    fid_scores = []
    kid_scores = []

    for run_idx in range(n_runs):
        print(f"\n{'=' * 50}")
        print(f"Run {run_idx + 1}/{n_runs}")
        print(f"{'=' * 50}")

        real_images_list = []
        fake_images_list = []

        print(f"Generating {n_images} images...")
        for i, (real_x_pt, _) in enumerate(dataloader):
            if i >= n_images:
                break

            real_x = tf.convert_to_tensor(real_x_pt.numpy(), dtype=tf.float32)

            z = tf.random.uniform(
                (1, latent_dim), minval=-np.pi, maxval=np.pi, dtype=tf.float32
            )
            fake_x = generator(z, training=False)

            real_img = x_to_vis_images_center28(real_x, image_size=image_size)
            fake_img = x_to_vis_images_center28(fake_x, image_size=image_size)

            real_np = real_img.squeeze(0).squeeze(0).numpy()
            fake_np = fake_img.squeeze(0).squeeze(0).numpy()

            real_images_list.append(real_np)
            fake_images_list.append(fake_np)

            if run_idx == 0:
                Image.fromarray((real_np * 255).astype(np.uint8)).save(
                    os.path.join(out_dir, "real", f"real_{i:04d}.png")
                )
                Image.fromarray((fake_np * 255).astype(np.uint8)).save(
                    os.path.join(out_dir, "fake", f"fake_{i:04d}.png")
                )

        real_images = np.array(real_images_list).astype(np.float32)
        fake_images = np.array(fake_images_list).astype(np.float32)

        print(f"Real images shape: {real_images.shape}")
        print(f"Fake images shape: {fake_images.shape}")

        real_features = real_images.reshape(real_images.shape[0], -1)
        fake_features = fake_images.reshape(fake_images.shape[0], -1)

        fid = calculate_fid(real_features, fake_features)
        kid_mean, kid_std = calculate_kid(
            real_features, fake_features, subset_size=min(100, n_images)
        )

        fid_scores.append(fid)
        kid_scores.append(kid_mean)

        print(f"FID: {fid:.2f}")
        print(f"KID: {kid_mean:.4f} (+/- {kid_std:.4f})")

    fid_mean = np.mean(fid_scores)
    fid_std = np.std(fid_scores)
    kid_mean = np.mean(kid_scores)
    kid_std = np.std(kid_scores)

    print(f"\n{'=' * 50}")
    print(f"Final Results (averaged over {n_runs} runs):")
    print(f"FID: {fid_mean:.2f} (+/- {fid_std:.2f})")
    print(f"KID: {kid_mean:.4f} (+/- {kid_std:.4f})")
    print(f"{'=' * 50}")

    if run_idx == n_runs - 1:
        plt.figure(figsize=(8, 4))

        plt.subplot(1, 2, 1)
        plt.imshow(real_images[0], cmap="gray")
        plt.title("Real Image")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(fake_images[0], cmap="gray")
        plt.title("Fake Image")
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "sample_comparison.png"), dpi=150)
        plt.close()

    results = {
        "fid_mean": float(fid_mean),
        "fid_std": float(fid_std),
        "kid_mean": float(kid_mean),
        "kid_std": float(kid_std),
        "n_images": n_images,
        "n_runs": n_runs,
        "fid_scores": np.array(fid_scores),
        "kid_scores": np.array(kid_scores),
    }
    np.save(os.path.join(out_dir, "metrics.npy"), results)
    print(f"[Save] Results saved to {os.path.join(out_dir, 'metrics.npy')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        default="./chy3_0/1A_out_angle_aux_subBase3_layer10_flatProb2_removeUcenter_noSmooth1.8_temp_sb1_3class/weights_batch_003000.pt",
        help="Path to .pt checkpoint file",
    )
    parser.add_argument("--classes", default="0")
    parser.add_argument("--layers", type=int, default=10)
    parser.add_argument("--n_data_qubits", type=int, default=10)
    parser.add_argument("--latent_dim", type=int, default=10)
    parser.add_argument(
        "--n_runs", type=int, default=3, help="Number of runs to average"
    )
    parser.add_argument("--n_images", type=int, default=250)
    parser.add_argument(
        "--mapping", choices=["calibrated", "direct_max"], default="calibrated"
    )
    parser.add_argument("--out", default="./test_mnist_0/cc_10")
    args = parser.parse_args()

    test(
        ckpt_path=args.ckpt,
        classes_str=args.classes,
        n_layers=args.layers,
        n_data_qubits=args.n_data_qubits,
        latent_dim=args.latent_dim,
        n_images=args.n_images,
        n_runs=args.n_runs,
        out_dir=args.out,
        mapping=args.mapping,
    )
