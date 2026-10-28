import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

import argparse
import json
import tensorflow as tf
import torch
import numpy as np
from PIL import Image
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
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
    """
    [-1,1] -> [0,1]
    """
    x = tf.reshape(x_tf_flat, (-1, 1, 32, 32))
    x = (x + 1.0) / 2.0
    x = tf.clip_by_value(x, 0.0, 1.0)
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


def calculate_kid(real_features, gen_features, subset_size=100, max_subsets=100):
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


def sample_z_uniform(
    bs: int, latent_dim: int, z_range: tuple = (0.0, 1.0)
) -> tf.Tensor:
    """ """
    min_val, max_val = z_range
    return tf.random.uniform(
        (bs, latent_dim), minval=min_val, maxval=max_val, dtype=tf.float32
    )


def test(
    config_path: str, out_dir: str, n_images: int = 250, z_range: tuple = (0.0, 1.0)
):
    setup_gpu()
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "real"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "fake"), exist_ok=True)

    with open(config_path, "r") as f:
        config = json.load(f)

    print(f"[Config] {config}")
    print(f"[Latent] z ~ U{z_range}")

    image_size = config.get("image_size", 32)
    n_data_qubits = config.get("n_data_qubits", 10)
    n_ancillas = config.get("n_ancillas", 1)
    n_layers = config.get("n_layers", 20)
    latent_dim = config.get("latent_dim", n_data_qubits + n_ancillas)
    classes = config.get("classes", [0])
    ancilla_lowest_bit = config.get("ancilla_lowest_bit", False)

    gan = PQWGAN_CC(
        image_size=image_size,
        channels=1,
        n_data_qubits=n_data_qubits,
        n_ancillas=n_ancillas,
        n_layers=n_layers,
    )
    critic = gan.critic
    generator = gan.generator

    _ = critic(tf.zeros((1, image_size * image_size), dtype=tf.float32))
    _ = generator(
        tf.zeros((1, latent_dim), dtype=tf.float32),
        training=False,
        ancilla_lowest_bit=ancilla_lowest_bit,
    )

    generator.load_weights(
        os.path.join(os.path.dirname(config_path), "generator-final.weights.h5")
    )
    print(
        f"[Load] Generator weights loaded from {os.path.join(os.path.dirname(config_path), 'generator-final.weights.h5')}"
    )

    dataset = select_from_dataset(
        load_mnist_32_intensity(), n_images, classes, start_per_class=1000
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    data_iter = iter(dataloader)
    real_images_list = []
    fake_images_list = []

    print(f"Generating {n_images} images...")
    pbar = tqdm(range(n_images), desc="Generating")
    for i in pbar:
        try:
            real_x_pt, _ = next(data_iter)
        except StopIteration:
            break

        real_x = tf.convert_to_tensor(real_x_pt.numpy(), dtype=tf.float32)

        z = sample_z_uniform(1, latent_dim, z_range)
        fake_x = generator(z, training=False, ancilla_lowest_bit=ancilla_lowest_bit)

        real_img = denorm_to_torch_32(real_x)
        fake_img = denorm_to_torch_32(fake_x)

        real_np = real_img.squeeze(0).squeeze(0).numpy()
        fake_np = fake_img.squeeze(0).squeeze(0).numpy()

        real_images_list.append(real_np)
        fake_images_list.append(fake_np)

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

    print(f"\n{'=' * 50}")
    print(f"Results:")
    print(f"FID: {fid:.2f}")
    print(f"KID: {kid_mean:.4f} (+/- {kid_std:.4f})")
    print(f"{'=' * 50}")

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
    print(
        f"[Save] Sample comparison saved to {os.path.join(out_dir, 'sample_comparison.png')}"
    )

    results = {
        "fid": float(fid),
        "kid_mean": float(kid_mean),
        "kid_std": float(kid_std),
        "n_images": n_images,
        "z_range": z_range,
    }
    np.save(os.path.join(out_dir, "metrics.npy"), results)
    print(f"[Save] Results saved to {os.path.join(out_dir, 'metrics.npy')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="./pqwgan_32x32/cc_0/model_config.json",
        help="Path to model_config.json",
    )
    parser.add_argument(
        "--n_images", type=int, default=250, help="Number of images for testing"
    )
    parser.add_argument(
        "--z_min", type=float, default=0.0, help="Latent z minimum (default: 0.0)"
    )
    parser.add_argument(
        "--z_max", type=float, default=1.0, help="Latent z maximum (default: 1.0)"
    )
    parser.add_argument(
        "--out", type=str, default="./pqwgan_test_32x32/cc_0", help="Output directory"
    )
    args = parser.parse_args()

    test(
        config_path=args.config,
        out_dir=args.out,
        n_images=args.n_images,
        z_range=(args.z_min, args.z_max),
    )
