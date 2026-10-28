# utils/dataset.py
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset


class Flatten(object):
    def __call__(self, x):
        return x.view(-1)  # [H*W]


def load_mnist_16_intensity(file_location="./datasets"):
    """
    Returns MNIST resized to 16x16 where each sample is a length-256 vector in [-1,1].
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # [1,28,28] in [0,1]
            transforms.Resize((16, 16)),  # -> [1,16,16]
            transforms.Normalize((0.5,), (0.5,)),  # -> [-1,1]
            Flatten(),  # -> [256]
        ]
    )
    mnist_train = torchvision.datasets.MNIST(
        root=file_location,
        train=True,
        download=True,
        transform=transform,
    )
    return mnist_train


def load_mnist_28_intensity(file_location="./datasets"):
    """
    Returns MNIST where each sample is a length-784 vector in [-1,1]
    corresponding to a 28x28 intensity image.
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # [1,28,28] in [0,1]
            transforms.Normalize((0.5,), (0.5,)),  # -> [-1,1]
            Flatten(),  # -> [784]
        ]
    )
    mnist_train = torchvision.datasets.MNIST(
        root=file_location,
        train=True,
        download=True,
        transform=transform,
    )
    return mnist_train


def load_mnist_32_intensity(file_location="./datasets"):
    """
    Returns MNIST padded to 32x32 where each sample is a length-1024 vector in [-1,1].
    Perfect match for 10 data qubits output (2^10 = 1024).
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # [1,28,28] in [0,1]
            transforms.Pad((2, 2, 2, 2), fill=0),  # -> [1,32,32]
            transforms.Normalize((0.5,), (0.5,)),  # -> [-1,1]
            Flatten(),  # -> [1024]
        ]
    )
    mnist_train = torchvision.datasets.MNIST(
        root=file_location,
        train=True,
        download=True,
        transform=transform,
    )
    return mnist_train


def select_from_dataset(dataset, per_class_size, labels, start_per_class=0):
    """Select a deterministic, disjoint range within each requested class."""
    indices_by_label = [[] for _ in range(10)]
    for i in range(len(dataset)):
        current_class = dataset[i][1]
        indices_by_label[current_class].append(i)

    indices_of_desired_labels = [indices_by_label[i] for i in labels]
    chosen = [
        item
        for sublist in indices_of_desired_labels
        for item in sublist[start_per_class : start_per_class + per_class_size]
    ]
    if len(chosen) != per_class_size * len(labels):
        raise ValueError("Not enough examples for the requested class range")
    return Subset(dataset, chosen)
