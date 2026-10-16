"""Inference-only latent-space analysis: statistics, PCA truncation and linear probes.

Nothing here trains an autoencoder. The only fitted objects are PCA bases (SVD) and a
multinomial logistic-regression probe on *frozen* features. Labels are used solely to fit and
score probes; they are never an autoencoder training target. Callers decide the split: the
analysis scripts use the train split to fit PCA/probes and the validation split to score them.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

# ---------------------------------------------------------------- extraction


@torch.no_grad()
def extract_latents(
    model: nn.Module, loader: Iterable, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a labelled loader; returns CPU float32 latents ``(N, d)`` and int labels ``(N,)``."""
    model.eval()
    zs, ys = [], []
    for images, labels in loader:
        zs.append(model.encode(images.to(device)).float().cpu())
        ys.append(labels.cpu())
    if not zs:
        raise ValueError("loader produced no batches")
    return torch.cat(zs), torch.cat(ys).long()


@torch.no_grad()
def extract_flat_images(loader: Iterable) -> tuple[torch.Tensor, torch.Tensor]:
    """Flattened raw pixels ``(N, C*H*W)`` and labels, used as a probe control."""
    xs, ys = [], []
    for images, labels in loader:
        xs.append(images.flatten(1).float())
        ys.append(labels)
    return torch.cat(xs), torch.cat(ys).long()


# ---------------------------------------------------------------- statistics


def covariance_spectrum(z: torch.Tensor) -> torch.Tensor:
    """Eigenvalues of the sample covariance (descending)."""
    if z.shape[0] < 2:
        raise ValueError("need at least 2 samples")
    s = torch.linalg.svdvals(z.double() - z.double().mean(0))
    return (s**2) / (z.shape[0] - 1)


def effective_rank(eigvals: torch.Tensor) -> float:
    """exp(entropy) of the normalised spectrum (Roy & Vetterli): 1 = rank-1, d = isotropic."""
    p = eigvals / eigvals.sum()
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


def participation_ratio(eigvals: torch.Tensor) -> float:
    """(sum lambda)^2 / sum lambda^2: another effective-dimensionality measure."""
    return float(eigvals.sum() ** 2 / (eigvals**2).sum())


def dims_for_variance(eigvals: torch.Tensor, fraction: float) -> int:
    """Smallest number of principal components explaining ``fraction`` of the variance."""
    cum = torch.cumsum(eigvals, 0) / eigvals.sum()
    return int((cum < fraction).sum().item()) + 1


def latent_statistics(z: torch.Tensor) -> dict[str, float | int | list[float]]:
    """Summary of how a latent code is used (a collapse diagnostic, not a quality score)."""
    eig = covariance_spectrum(z)
    std = z.std(0)
    return {
        "latent_dim": int(z.shape[1]),
        "n_samples": int(z.shape[0]),
        "effective_rank": effective_rank(eig),
        "participation_ratio": participation_ratio(eig),
        "dims_for_90pct_variance": dims_for_variance(eig, 0.90),
        "dims_for_95pct_variance": dims_for_variance(eig, 0.95),
        "dims_for_99pct_variance": dims_for_variance(eig, 0.99),
        "n_low_variance_dims": int((std < 0.01 * std.mean()).sum()),
        "mean_dim_std": float(std.mean()),
        "top1_variance_fraction": float(eig[0] / eig.sum()),
        "explained_variance_cumulative": (torch.cumsum(eig, 0) / eig.sum()).tolist(),
    }


# ---------------------------------------------------------------- PCA


@dataclass
class PCABasis:
    mean: torch.Tensor  # (d,)
    components: torch.Tensor  # (d, d) columns = principal axes, descending variance


def fit_pca(x_train: torch.Tensor, chunk: int = 8192) -> PCABasis:
    """PCA basis (full rank) from the covariance eigendecomposition in float64.

    Works for latents and for high-dimensional pixel vectors without a full SVD of the data.
    """
    n, d = x_train.shape
    mean = torch.zeros(d, dtype=torch.float64)
    for i in range(0, n, chunk):
        mean += x_train[i : i + chunk].double().sum(0)
    mean /= n
    cov = torch.zeros(d, d, dtype=torch.float64)
    for i in range(0, n, chunk):
        c = x_train[i : i + chunk].double() - mean
        cov += c.T @ c
    cov /= max(n - 1, 1)
    evals, evecs = torch.linalg.eigh(cov)
    order = torch.argsort(evals, descending=True)
    return PCABasis(mean.float(), evecs[:, order].float().contiguous())


def project(z: torch.Tensor, basis: PCABasis, k: int) -> torch.Tensor:
    """Coordinates of ``z`` on the first ``k`` principal axes."""
    return (z - basis.mean) @ basis.components[:, :k]


def truncate(z: torch.Tensor, basis: PCABasis, k: int) -> torch.Tensor:
    """Reconstruct ``z`` from its first ``k`` principal coordinates (k=0 gives the mean)."""
    if not 0 <= k <= basis.components.shape[1]:
        raise ValueError(f"k must be in [0, {basis.components.shape[1]}]")
    if k == 0:
        return basis.mean.expand_as(z).clone()
    v = basis.components[:, :k]
    return basis.mean + (z - basis.mean) @ v @ v.T


@torch.no_grad()
def truncation_curve(
    model: nn.Module,
    loader: Iterable,
    device: torch.device,
    basis: PCABasis,
    ks: Sequence[int],
) -> dict[int, float]:
    """Per-pixel MSE of ``decode(truncate(encode(x), k))`` against ``x`` for each ``k``.

    A post-hoc probe of how much of the latent the *trained* decoder relies on. It is not
    equivalent to training a model with a smaller latent dimension.
    """
    model.eval()
    sums = {k: 0.0 for k in ks}
    n = 0
    mean, comps = basis.mean.to(device), basis.components.to(device)
    dev_basis = PCABasis(mean, comps)
    for images, _ in loader:
        x = images.to(device)
        z = model.encode(x)
        for k in ks:
            sums[k] += F.mse_loss(
                model.decode(truncate(z, dev_basis, k)), x, reduction="sum"
            ).item()
        n += x.numel()
    return {k: v / n for k, v in sums.items()}


# ---------------------------------------------------------------- linear probe


def standardize(train: torch.Tensor, *others: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Standardise with the TRAIN mean/std (constant features are left unscaled)."""
    mean = train.mean(0)
    std = train.std(0).clamp_min(1e-6)
    return tuple((t - mean) / std for t in (train, *others))


def _fit_logreg(
    x: torch.Tensor, y: torch.Tensor, n_classes: int, l2: float, max_iter: int
) -> tuple[torch.Tensor, torch.Tensor]:
    w = torch.zeros(x.shape[1], n_classes, requires_grad=True)
    b = torch.zeros(n_classes, requires_grad=True)
    opt = torch.optim.LBFGS(
        [w, b], lr=1.0, max_iter=max_iter, line_search_fn="strong_wolfe", tolerance_grad=1e-6
    )

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(x @ w + b, y) + l2 * (w**2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach(), b.detach()


def _accuracy(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> float:
    return float(((x @ w + b).argmax(1) == y).float().mean())


def fit_linear_probe(
    z_train: torch.Tensor,
    y_train: torch.Tensor,
    z_eval: torch.Tensor,
    y_eval: torch.Tensor,
    n_classes: int = 10,
    l2_grid: Sequence[float] = (1e-4, 1e-3, 1e-2),
    holdout_fraction: float = 0.1,
    max_iter: int = 200,
    seed: int = 0,
) -> dict[str, float]:
    """Multinomial logistic regression on frozen features.

    The L2 strength is chosen on a held-out slice of the TRAIN features only; the evaluation
    split is scored once, after refitting on all training features. Features are standardised
    with train statistics.
    """
    xtr, xev = standardize(z_train, z_eval)
    n = xtr.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    n_hold = max(1, int(n * holdout_fraction))
    hold, sub = perm[:n_hold], perm[n_hold:]
    best_l2, best_acc = l2_grid[0], -1.0
    for l2 in l2_grid:
        w, b = _fit_logreg(xtr[sub], y_train[sub], n_classes, l2, max_iter)
        acc = _accuracy(xtr[hold], y_train[hold], w, b)
        if acc > best_acc:
            best_l2, best_acc = l2, acc
    w, b = _fit_logreg(xtr, y_train, n_classes, best_l2, max_iter)
    acc = _accuracy(xev, y_eval, w, b)
    n_eval = len(y_eval)
    return {
        "eval_accuracy": acc,
        "eval_accuracy_stderr": (acc * (1 - acc) / n_eval) ** 0.5,
        "train_accuracy": _accuracy(xtr, y_train, w, b),
        "heldout_accuracy_for_l2": best_acc,
        "l2": best_l2,
        "n_features": int(xtr.shape[1]),
        "n_train": int(n),
        "n_eval": int(n_eval),
        "chance": 1.0 / n_classes,
    }
