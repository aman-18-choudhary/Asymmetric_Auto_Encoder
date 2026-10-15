"""Reconstruction objectives.

Multiscale loss: for outputs ``x_hat_s`` at resolutions ``H_s x W_s`` the target at scale ``s`` is
the input area-averaged to ``H_s x W_s`` (``adaptive_avg_pool2d``; exact block averaging when the
ratio is an integer). The loss is ``sum_s w_s * MSE(x_hat_s, area_down(x))`` with each MSE a mean
over elements, so a scale's weight is directly its contribution and the full-resolution term
equals plain MSE when the other weights are 0.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


def downsample_target(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Area-average ``x`` to ``size``; requires integer height/width ratios."""
    h, w = x.shape[-2:]
    if h % size[0] != 0 or w % size[1] != 0:
        raise ValueError(f"cannot area-downsample {(h, w)} to {tuple(size)}: non-integer ratio")
    if (h, w) == tuple(size):
        return x
    return F.adaptive_avg_pool2d(x, size)


def multiscale_mse(
    outputs: Sequence[torch.Tensor], target: torch.Tensor, weights: Sequence[float]
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Weighted multiscale MSE. Returns (total loss, per-scale unweighted MSEs)."""
    if len(outputs) != len(weights):
        raise ValueError(f"{len(outputs)} outputs but {len(weights)} weights")
    per_scale = []
    total = target.new_zeros(())
    for out, w in zip(outputs, weights, strict=True):
        mse = F.mse_loss(out, downsample_target(target, out.shape[-2:]))
        per_scale.append(mse)
        total = total + w * mse
    return total, per_scale


def model_loss(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Training objective for ``model``.

    Models exposing ``forward_multiscale`` use ``multiscale_mse`` with ``model.scale_weights``;
    all others use plain MSE between ``model(x)`` and ``x``.
    """
    if hasattr(model, "forward_multiscale"):
        return multiscale_mse(model.forward_multiscale(x), x, model.scale_weights)[0]
    return F.mse_loss(model(x), x)
