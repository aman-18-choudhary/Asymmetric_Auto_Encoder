"""Reconstruction metrics. All images are float tensors in [0, 1] (``DATA_RANGE = 1.0``).

* MSE / MAE: mean squared / absolute per-pixel error (lower is better).
* PSNR: ``10 log10(DATA_RANGE^2 / MSE)`` in dB, computed per image then averaged (higher is
  better). It is a monotone transform of per-image MSE, not an independent measure.
* SSIM: structural similarity, 11x11 Gaussian window (sigma 1.5), K1=0.01, K2=0.03, 'valid'
  windows, averaged over channels and positions, then over images (higher is better).

None of these measure semantic quality of the latent representation.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from torch.nn import functional as F

DATA_RANGE = 1.0
_PSNR_MSE_FLOOR = 1e-10  # avoids +inf for perfect reconstructions


def mse_per_image(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).pow(2).flatten(1).mean(1)


def mae_per_image(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).abs().flatten(1).mean(1)


def psnr_per_image(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = mse_per_image(pred, target).clamp_min(_PSNR_MSE_FLOOR)
    return 10.0 * torch.log10(DATA_RANGE**2 / mse)


def _gaussian_window(size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    return g[:, None] * g[None, :]


def ssim_per_image(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    """Mean SSIM per image (shape ``(B,)``). Images smaller than the window are rejected."""
    if pred.shape != target.shape or pred.ndim != 4:
        raise ValueError("pred and target must both be (B, C, H, W) with equal shapes")
    if min(pred.shape[-2:]) < window_size:
        raise ValueError(f"images smaller than the {window_size}x{window_size} SSIM window")
    c = pred.shape[1]
    win = _gaussian_window(window_size, sigma, pred.device, pred.dtype)
    kernel = win.expand(c, 1, window_size, window_size).contiguous()

    def blur(t: torch.Tensor) -> torch.Tensor:
        return F.conv2d(t, kernel, groups=c)

    mu_x, mu_y = blur(pred), blur(target)
    var_x = blur(pred * pred) - mu_x**2
    var_y = blur(target * target) - mu_y**2
    cov = blur(pred * target) - mu_x * mu_y
    c1, c2 = (0.01 * DATA_RANGE) ** 2, (0.03 * DATA_RANGE) ** 2
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / (
        (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
    )
    return ssim_map.flatten(1).mean(1)


def batch_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-image metric tensors for one batch."""
    return {
        "mse": mse_per_image(pred, target),
        "mae": mae_per_image(pred, target),
        "psnr": psnr_per_image(pred, target),
        "ssim": ssim_per_image(pred, target),
    }


@torch.no_grad()
def evaluate_reconstruction(
    model: nn.Module,
    loader: Iterable,
    device: torch.device,
    corrupt=None,
) -> dict[str, float]:
    """Mean per-image metrics of ``model(x)`` against the *clean* ``x`` over a loader.

    ``corrupt`` (optional callable) maps clean ``x`` to the model input; the reconstruction is
    still scored against the clean image (protocol N1 in the blueprint).
    """
    model.eval()
    sums = {"mse": 0.0, "mae": 0.0, "psnr": 0.0, "ssim": 0.0}
    n = 0
    for batch in loader:
        x = (batch[0] if isinstance(batch, (tuple, list)) else batch).to(device)
        inp = corrupt(x) if corrupt is not None else x
        stats = batch_metrics(model(inp), x)
        for k in sums:
            sums[k] += stats[k].sum().item()
        n += x.shape[0]
    if n == 0:
        raise ValueError("loader produced no batches")
    out = {k: v / n for k, v in sums.items()}
    out["n_images"] = n
    return out
