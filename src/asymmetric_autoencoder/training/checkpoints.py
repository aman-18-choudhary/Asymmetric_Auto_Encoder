"""Checkpoint saving and loading.

Checkpoints are written atomically (temp file + rename) so an interrupted save cannot corrupt
an existing file, and are loaded with ``weights_only=True`` so loading never executes pickled
code. Contents: model/optimizer/scheduler state, progress counters, history, config and RNG state.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch import nn

CHECKPOINT_VERSION = 1


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    **extra: Any,
) -> None:
    """Atomically write a checkpoint; ``extra`` entries are stored verbatim."""
    payload = {
        "version": CHECKPOINT_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        **extra,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, target)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint into ``model`` (and optimizer/scheduler if given); return the payload."""
    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    if payload.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"unsupported checkpoint version {payload.get('version')!r}")
    model.load_state_dict(payload["model"])
    if optimizer is not None:
        if payload["optimizer"] is None:
            raise ValueError("checkpoint has no optimizer state")
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        if payload["scheduler"] is None:
            raise ValueError("checkpoint has no scheduler state")
        scheduler.load_state_dict(payload["scheduler"])
    return payload
