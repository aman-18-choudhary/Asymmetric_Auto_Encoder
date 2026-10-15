"""Minimal reconstruction training loop for Phase 2 debugging.

Deliberately small: no checkpoints, scheduler or early stopping. Phase 4 replaces this with the
full training infrastructure shared by both models.
"""

from __future__ import annotations

import time
from collections.abc import Iterable

import torch
from torch import nn
from torch.nn import functional as F

from asymmetric_autoencoder.models.losses import model_loss
from asymmetric_autoencoder.utils.logging import get_logger

log = get_logger(__name__)


def _images(batch) -> torch.Tensor:
    """Datasets yield (image, label); labels are ignored (self-supervised)."""
    return batch[0] if isinstance(batch, (tuple, list)) else batch


@torch.no_grad()
def evaluate_mse(
    model: nn.Module, loader: Iterable, device: torch.device, max_batches: int | None = None
) -> float:
    """Mean per-pixel MSE over a loader (average weighted by batch size)."""
    model.eval()
    total, count = 0.0, 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = _images(batch).to(device)
        total += F.mse_loss(model(x), x, reduction="sum").item()
        count += x.numel()
    return total / max(count, 1)


def train_reconstruction(
    model: nn.Module,
    loader: Iterable,
    device: torch.device,
    epochs: int,
    lr: float,
    max_batches: int | None = None,
) -> list[float]:
    """Train with Adam on ``model_loss``; return mean training loss per epoch.

    The history is the model's own objective (plain MSE for the baseline, weighted multiscale
    MSE for the asymmetric model), so it is not comparable across models; use ``evaluate_mse``.
    """
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[float] = []
    for epoch in range(epochs):
        start, running, n = time.perf_counter(), 0.0, 0
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            x = _images(batch).to(device)
            loss = model_loss(model, x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += loss.item()
            n += 1
        history.append(running / max(n, 1))
        log.info(
            "epoch %d/%d train_mse=%.5f (%.1fs)",
            epoch + 1,
            epochs,
            history[-1],
            time.perf_counter() - start,
        )
    return history
