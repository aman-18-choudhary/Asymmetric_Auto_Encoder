"""Input-corruption robustness evaluation (blueprint protocol N1).

Models are trained on clean images and receive corrupted inputs at test time; every metric is
computed against the *clean* image. Corruption is deterministic: each batch draws its noise from
a generator seeded with ``(seed, batch_index)``, so any two models evaluated on the same loader
(fixed order) see exactly the same corrupted images.

Kinds: ``gaussian`` (additive N(0, level^2), clipped to [0, 1]) and ``salt_pepper`` (a fraction
``level`` of pixel positions set to 0 or 1 with equal probability, all channels together).
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from asymmetric_autoencoder.evaluation.metrics import batch_metrics

KINDS = ("gaussian", "salt_pepper")


def corrupt(x: torch.Tensor, kind: str, level: float, generator: torch.Generator) -> torch.Tensor:
    """Return a corrupted copy of ``x`` (values stay in [0, 1])."""
    if kind not in KINDS:
        raise ValueError(f"unknown corruption {kind!r}; expected one of {KINDS}")
    if level < 0:
        raise ValueError("level must be >= 0")
    if level == 0:
        return x.clone()
    cpu = torch.device("cpu")
    if kind == "gaussian":
        noise = torch.randn(x.shape, generator=generator, device=cpu).to(x.device, x.dtype)
        return (x + level * noise).clamp(0.0, 1.0)
    if level > 1:
        raise ValueError("salt_pepper level is a probability and must be <= 1")
    b, _, h, w = x.shape
    hit = torch.rand((b, 1, h, w), generator=generator, device=cpu) < level
    salt = torch.rand((b, 1, h, w), generator=generator, device=cpu) < 0.5
    hit, salt = hit.to(x.device), salt.to(x.device)
    return torch.where(hit, salt.to(x.dtype).expand_as(x), x)


@torch.no_grad()
def evaluate_corruption(
    model: nn.Module,
    loader: Iterable,
    device: torch.device,
    kind: str,
    level: float,
    seed: int = 0,
) -> dict[str, float]:
    """Mean metrics of ``model(corrupt(x))`` vs clean ``x``, plus the corrupted input vs clean.

    ``input_*`` entries are the metrics of the corrupted image itself (what a model that returned
    its input unchanged would score), the reference for judging whether a model denoises.
    """
    model.eval()
    keys = ("mse", "mae", "psnr", "ssim")
    sums = {k: 0.0 for k in keys}
    in_sums = {f"input_{k}": 0.0 for k in keys}
    n = 0
    for i, batch in enumerate(loader):
        x = (batch[0] if isinstance(batch, (tuple, list)) else batch).to(device)
        gen = torch.Generator().manual_seed(seed * 1_000_003 + i)
        noisy = corrupt(x, kind, level, gen)
        out = batch_metrics(model(noisy), x)
        ref = batch_metrics(noisy, x)
        for k in keys:
            sums[k] += out[k].sum().item()
            in_sums[f"input_{k}"] += ref[k].sum().item()
        n += x.shape[0]
    if n == 0:
        raise ValueError("loader produced no batches")
    result = {k: v / n for k, v in {**sums, **in_sums}.items()}
    result["n_images"] = n
    return result
