"""Seeding, device selection and environment capture for reproducible experiments."""

from __future__ import annotations

import os
import platform
import random
import sys

import numpy as np
import torch

_VALID_DEVICES = ("auto", "cpu", "cuda", "mps")


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch RNGs.

    If ``deterministic`` is True, request deterministic PyTorch algorithms. Operations without
    a deterministic implementation (some MPS/CUDA kernels) will raise instead of silently
    varying; this is a documented reproducibility limit, not a guarantee.
    """
    if not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative int, got {seed!r}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.use_deterministic_algorithms(deterministic)
    if deterministic and torch.cuda.is_available():
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """``worker_init_fn`` for DataLoader so NumPy/random are seeded per worker."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    """Dedicated generator (e.g. for DataLoader shuffling or train/val splits)."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def resolve_device(requested: str = "auto") -> torch.device:
    """Map a device request to a ``torch.device``.

    ``auto`` prefers CUDA, then MPS, then CPU. An explicit request for an unavailable
    accelerator raises ``RuntimeError`` rather than silently falling back.
    """
    if requested not in _VALID_DEVICES:
        raise ValueError(f"device must be one of {_VALID_DEVICES}, got {requested!r}")
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but not available")
    return torch.device(requested)


def environment_info() -> dict[str, str]:
    """Library and platform versions to store next to experiment results."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": str(torch.cuda.is_available()),
        "mps_available": str(torch.backends.mps.is_available()),
    }


def get_rng_state() -> dict:
    """Snapshot all RNG states in a form accepted by ``torch.load(weights_only=True)``."""
    np_state = np.random.get_state()
    state = {
        "python": random.getstate(),
        "numpy": (np_state[0], torch.from_numpy(np_state[1].copy()), *np_state[2:]),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def set_rng_state(state: dict) -> None:
    """Restore a snapshot from :func:`get_rng_state` (accelerator states only if present)."""
    random.setstate((state["python"][0], tuple(state["python"][1]), state["python"][2]))
    name, keys, pos, has_gauss, cached = state["numpy"]
    np.random.set_state((name, keys.cpu().numpy().astype(np.uint32), pos, has_gauss, cached))
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"].cpu())
