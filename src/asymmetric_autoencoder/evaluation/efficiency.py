"""Model cost measurements: parameters, FLOPs and inference latency."""

from __future__ import annotations

import statistics
import time

import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode


def count_flops(model: nn.Module, input_shape: tuple[int, ...]) -> int:
    """Forward-pass FLOPs for one input of ``input_shape`` (multiply-add counted as 2 FLOPs).

    Uses ``torch.utils.flop_counter`` which counts convolutions and matrix multiplications
    only; element-wise ops (activations, BatchNorm, upsampling) are not included.
    """
    model.eval()
    device = next(model.parameters()).device
    with FlopCounterMode(display=False) as counter, torch.no_grad():
        model(torch.zeros(1, *input_shape, device=device))
    return int(counter.get_total_flops())


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def measure_latency(
    model: nn.Module,
    input_shape: tuple[int, ...],
    device: torch.device,
    batch_size: int,
    warmup: int = 10,
    iters: int = 50,
) -> dict[str, float]:
    """Wall-clock forward latency in milliseconds (median and IQR-based spread over ``iters``)."""
    model = model.to(device).eval()
    x = torch.rand(batch_size, *input_shape, device=device)
    for _ in range(warmup):
        model(x)
    _sync(device)
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        model(x)
        _sync(device)
        times.append((time.perf_counter() - start) * 1000.0)
    q = statistics.quantiles(times, n=4)
    return {
        "median_ms": statistics.median(times),
        "q1_ms": q[0],
        "q3_ms": q[2],
        "per_image_ms": statistics.median(times) / batch_size,
        "batch_size": batch_size,
        "device": str(device),
    }
