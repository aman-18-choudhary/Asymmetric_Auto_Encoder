"""Figures for experiment reports (matplotlib, non-interactive backend)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402


def _to_image(t: torch.Tensor) -> np.ndarray:
    a = t.detach().cpu().clamp(0, 1).numpy()
    return a[0] if a.shape[0] == 1 else a.transpose(1, 2, 0)


def reconstruction_grid(
    originals: torch.Tensor,
    reconstructions: dict[str, torch.Tensor],
    path: str | Path,
    title: str,
) -> None:
    """Rows: original then one row per model; columns: the same images for every row."""
    rows = {"original": originals, **reconstructions}
    n = originals.shape[0]
    fig, axes = plt.subplots(len(rows), n, figsize=(1.2 * n, 1.3 * len(rows)), squeeze=False)
    for r, (label, imgs) in enumerate(rows.items()):
        for c in range(n):
            ax = axes[r][c]
            ax.imshow(_to_image(imgs[c]), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=8)
    fig.suptitle(title, fontsize=10)
    _save(fig, path)


def curves(series: dict[str, list[list[float]]], path: str | Path, title: str, ylabel: str) -> None:
    """Per-model mean curve over seeds with a +/- 1 std band; ``series[name]`` = one list/seed."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, runs in series.items():
        arr = np.asarray(runs, dtype=float)
        x = np.arange(1, arr.shape[1] + 1)
        mean, std = arr.mean(0), arr.std(0, ddof=1) if arr.shape[0] > 1 else np.zeros(arr.shape[1])
        ax.plot(x, mean, label=f"{name} (n={arr.shape[0]})")
        ax.fill_between(x, mean - std, mean + std, alpha=0.2)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    ax.set_title(title)
    ax.legend()
    _save(fig, path)


def seed_bars(values: dict[str, list[float]], path: str | Path, title: str, ylabel: str) -> None:
    """Bar of the mean per model with each seed's value overlaid as a point."""
    fig, ax = plt.subplots(figsize=(5, 4))
    names = list(values)
    for i, name in enumerate(names):
        v = np.asarray(values[name], dtype=float)
        ax.bar(i, v.mean(), alpha=0.6, yerr=v.std(ddof=1) if len(v) > 1 else 0, capsize=4)
        ax.scatter([i] * len(v), v, color="black", zorder=3, s=14)
    ax.set_xticks(range(len(names)), names, fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _save(fig, path)


def _save(fig, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
