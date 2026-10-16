"""Build the small local assets used by the Streamlit demo (inference only).

* Images come exclusively from ``data_batch_1`` ... ``data_batch_5`` (read directly with pickle;
  no torchvision dataset class, no other files except ``batches.meta`` for class names).
* The validation indices are the exact output of ``data.datasets.split_indices`` with the saved
  ``val_fraction`` / ``split_seed``, so they are identical to the research split.
* PCA bases are fitted on TRAIN-split latents only; the validation sample is only projected.
* Outputs are written only to ``<project root>/demo/assets`` (see ``ensure_assets_dir``).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from torch import nn

from asymmetric_autoencoder.evaluation.latent_analysis import covariance_spectrum, fit_pca, project

TRAIN_FILES = tuple(f"data_batch_{i}" for i in range(1, 6))
META_FILE = "batches.meta"


def load_train_arrays(data_dir: str | Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """All 50,000 training images as uint8 (N, 3, 32, 32) in torchvision's order, labels, names."""
    data_dir = Path(data_dir)
    images, labels = [], []
    for name in TRAIN_FILES:
        with open(data_dir / name, "rb") as handle:
            batch = pickle.load(handle, encoding="bytes")
        images.append(np.asarray(batch[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32))
        labels.append(np.asarray(batch[b"labels"], dtype=np.int64))
    with open(data_dir / META_FILE, "rb") as handle:
        names = [
            n.decode() if isinstance(n, bytes) else str(n)
            for n in pickle.load(handle, encoding="bytes")[b"label_names"]
        ]
    return np.concatenate(images), np.concatenate(labels), names


def stratified_sample(
    labels: np.ndarray, candidates: np.ndarray, per_class: int, seed: int = 0
) -> np.ndarray:
    """``per_class`` indices per class drawn from ``candidates`` (deterministic, class-ordered)."""
    gen = torch.Generator().manual_seed(seed)
    chosen = []
    for c in sorted(set(labels[candidates].tolist())):
        pool = candidates[labels[candidates] == c]
        if len(pool) < per_class:
            raise ValueError(f"class {c} has only {len(pool)} candidates")
        pick = torch.randperm(len(pool), generator=gen)[:per_class].numpy()
        chosen.append(np.sort(pool[pick]))
    return np.concatenate(chosen)


@torch.no_grad()
def encode_uint8(
    model: nn.Module, images: np.ndarray, device: torch.device, batch_size: int = 500
) -> torch.Tensor:
    model = model.to(device).eval()
    out = []
    for i in range(0, len(images), batch_size):
        x = torch.from_numpy(images[i : i + batch_size]).to(device).float().div(255.0)
        out.append(model.encode(x).float().cpu())
    return torch.cat(out)


def build_pca_asset(
    model: nn.Module, train_images: np.ndarray, sample_images: np.ndarray, device: torch.device
) -> dict[str, torch.Tensor]:
    """PCA basis from TRAIN latents; the sample images are only projected onto it."""
    z_train = encode_uint8(model, train_images, device)
    basis = fit_pca(z_train)
    z_sample = encode_uint8(model, sample_images, device)
    eig = covariance_spectrum(z_train)
    return {
        "mean": basis.mean,
        "components": basis.components,
        "explained_variance_ratio": (eig / eig.sum()).float(),
        "sample_coords": project(z_sample, basis, 2).float(),
    }


PROTECTED_DIRS = ("experiments", "reports", "data")


def ensure_assets_dir(out_dir: str | Path, project_root: str | Path) -> Path:
    """Return ``<project_root>/demo/assets`` (created if needed) or raise ``ValueError``.

    ``out_dir`` is resolved (``..``, relative paths and symlinks are followed) and must equal the
    resolved ``<project_root>/demo/assets``. It is also rejected if it lies inside ``experiments/``,
    ``reports/`` or ``data/`` of the project, or if ``demo`` / ``demo/assets`` are symlinks (which
    could redirect writes elsewhere). Nothing is created unless every check passes.
    """
    root = Path(project_root).resolve()
    expected = root / "demo" / "assets"
    out = Path(out_dir).resolve()
    if out != expected:
        raise ValueError(f"demo assets may only be written to {expected}, not {out}")
    for protected in PROTECTED_DIRS:
        if out.is_relative_to(root / protected):
            raise ValueError(f"refusing to write inside {root / protected}")
    if (root / "demo").is_symlink() or (root / "demo" / "assets").is_symlink():
        raise ValueError("demo/ and demo/assets must not be symlinks")
    out.mkdir(parents=True, exist_ok=True)
    return out
