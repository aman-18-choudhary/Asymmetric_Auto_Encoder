"""Dataset loading, deterministic train/validation splits and DataLoader construction.

Images are scaled to [0, 1] (``ToTensor``) and no further normalisation is applied, so the
sigmoid decoder output, the reconstruction target and the PSNR/SSIM data range (1.0) agree.
MNIST is resized to ``image_size`` so debug runs use the same architecture as CIFAR-10.
Labels are returned by the datasets but are not used for training.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from asymmetric_autoencoder.config import DataConfig
from asymmetric_autoencoder.utils.reproducibility import make_generator, seed_worker

_REGISTRY = {
    "mnist": (datasets.MNIST, 1),
    "fashion_mnist": (datasets.FashionMNIST, 1),
    "cifar10": (datasets.CIFAR10, 3),
}


def num_channels(name: str) -> int:
    """Number of image channels of a registered dataset."""
    return _REGISTRY[name][1]


def build_transform(image_size: int) -> transforms.Compose:
    """Resize (no-op when already the right size) and convert to a [0, 1] float tensor."""
    return transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])


def split_indices(n: int, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """Deterministic, disjoint train/validation index split of ``range(n)``.

    The result depends only on ``(n, val_fraction, seed)``, so every model sees the same split.
    """
    if n < 2:
        raise ValueError("need at least 2 samples to split")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    n_val = max(1, round(n * val_fraction))
    if n_val >= n:
        raise ValueError("validation split would leave no training samples")
    perm = torch.randperm(n, generator=make_generator(seed)).tolist()
    return sorted(perm[n_val:]), sorted(perm[:n_val])


@dataclass
class DataSplits:
    train: Dataset
    val: Dataset
    test: Dataset
    in_channels: int
    train_indices: list[int]
    val_indices: list[int]


def build_splits(cfg: DataConfig, download: bool = False) -> DataSplits:
    """Create train/val/test datasets. The test set is the official held-out test split.

    Set ``download=True`` (see ``scripts/download_data.py``) to fetch data; by default nothing
    is downloaded and a missing dataset raises ``RuntimeError``.
    """
    cls, channels = _REGISTRY[cfg.name]
    root = Path(cfg.root)
    tf = build_transform(cfg.image_size)
    full_train = cls(root=str(root), train=True, download=download, transform=tf)
    test = cls(root=str(root), train=False, download=download, transform=tf)
    train_idx, val_idx = split_indices(len(full_train), cfg.val_fraction, cfg.split_seed)
    return DataSplits(
        train=Subset(full_train, train_idx),
        val=Subset(full_train, val_idx),
        test=test,
        in_channels=channels,
        train_indices=train_idx,
        val_indices=val_idx,
    )


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
) -> DataLoader:
    """DataLoader whose shuffling and worker RNGs are controlled by ``seed``."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=make_generator(seed),
    )
