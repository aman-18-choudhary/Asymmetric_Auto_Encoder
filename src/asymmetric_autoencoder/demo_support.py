"""Read-only helpers for the local Streamlit demo (no streamlit import, CPU only).

Everything here reads existing checkpoints and saved results; nothing writes to
``experiments/results`` or ``reports/figures``. Validation images come from a small asset file
built from the training batches only (see ``demo_assets.py``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from asymmetric_autoencoder.config import load_config
from asymmetric_autoencoder.evaluation.latent_analysis import PCABasis, truncate
from asymmetric_autoencoder.evaluation.metrics import batch_metrics
from asymmetric_autoencoder.evaluation.robustness import corrupt
from asymmetric_autoencoder.models.factory import build_model
from asymmetric_autoencoder.training.checkpoints import load_checkpoint

MODEL_RUNS = {
    "baseline": "baseline_cifar10",
    "baseline_matched": "baseline_matched_cifar10",
    "asymmetric": "asymmetric_cifar10",
}
SAMPLES_FILE = "val_samples.npz"
PCA_FILE = "pca_bases.pt"
SEED = 0  # all research results come from one training run (seed 0) per model
LATENCY_LABELS = {  # saved latency keys -> table columns (median ms per forward pass)
    "cpu_batch1": "CPU latency, 1 image (ms)",
    "cpu_batch128": "CPU latency, batch of 128 (ms)",
    "mps_batch1": "MPS latency, 1 image (ms)",
    "mps_batch128": "MPS latency, batch of 128 (ms)",
}

# Figures the demo may show: metric plots only, never reconstructions of held-out images.
FIGURES = {
    "val_mse_curves.png": "Validation MSE during training (seed 0)",
    "robustness_gaussian_psnr.png": "PSNR under Gaussian noise (N1, validation)",
    "robustness_gaussian_ssim.png": "SSIM under Gaussian noise (N1, validation)",
    "robustness_salt_pepper_psnr.png": "PSNR under salt-and-pepper noise (N1, validation)",
    "robustness_salt_pepper_ssim.png": "SSIM under salt-and-pepper noise (N1, validation)",
    "latent_explained_variance.png": "Cumulative explained variance of the latents (train split)",
    "latent_truncation_mse.png": "Validation MSE after post-hoc latent truncation",
    "latent_probe_accuracy.png": "Linear-probe accuracy on frozen features (validation)",
}


# ------------------------------------------------------------------ models
def run_dir(results_dir: Path, model: str) -> Path:
    return Path(results_dir) / f"{MODEL_RUNS[model]}_seed{SEED}"


def checkpoint_path(results_dir: Path, model: str) -> Path:
    return run_dir(results_dir, model) / "checkpoints" / "best.pt"


def missing_checkpoints(results_dir: Path) -> list[Path]:
    return [p for m in MODEL_RUNS if not (p := checkpoint_path(results_dir, m)).is_file()]


def load_model(results_dir: Path, name: str) -> nn.Module:
    """Build a model from its saved config and load ``best.pt`` (CPU, eval mode, no grads)."""
    path = checkpoint_path(results_dir, name)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    cfg = load_config(run_dir(results_dir, name) / "config.yaml")
    model = build_model(cfg.model, 3, cfg.data.image_size)
    load_checkpoint(path, model, map_location="cpu")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_models(results_dir: Path) -> dict[str, nn.Module]:
    missing = missing_checkpoints(results_dir)
    if missing:
        raise FileNotFoundError("checkpoints not found: " + ", ".join(str(p) for p in missing))
    return {name: load_model(results_dir, name) for name in MODEL_RUNS}


# ------------------------------------------------------------------ samples and images
@dataclass
class Samples:
    images: np.ndarray  # (N, 3, 32, 32) uint8
    labels: np.ndarray  # (N,)
    val_indices: np.ndarray  # positions in the 50,000-image training array
    class_names: list[str]

    def __len__(self) -> int:
        return len(self.labels)

    def tensor(self, i: int) -> torch.Tensor:
        """Sample ``i`` as a float tensor in [0, 1] with shape (1, 3, 32, 32)."""
        return torch.from_numpy(self.images[i]).float().div(255.0).unsqueeze(0)


def load_samples(path: Path) -> Samples:
    with np.load(path, allow_pickle=False) as f:
        samples = Samples(
            f["images"], f["labels"], f["val_indices"], [str(c) for c in f["class_names"]]
        )
    if samples.images.ndim != 4 or samples.images.shape[1:] != (3, 32, 32):
        raise ValueError(f"unexpected sample image shape {samples.images.shape}")
    return samples


def to_display(x: torch.Tensor, factor: int = 1) -> np.ndarray:
    """(1, 3, H, W) tensor in [0, 1] -> (H*f, W*f, 3) float array (nearest-neighbour upscale)."""
    img = x[0].detach().clamp(0, 1).permute(1, 2, 0).numpy()
    return np.kron(img, np.ones((factor, factor, 1), dtype=img.dtype)) if factor > 1 else img


# ------------------------------------------------------------------ inference
@torch.no_grad()
def reconstruct(models: dict[str, nn.Module], x: torch.Tensor) -> dict[str, torch.Tensor]:
    return {name: m(x) for name, m in models.items()}


@torch.no_grad()
def multiscale_outputs(model: nn.Module, x: torch.Tensor) -> list[torch.Tensor] | None:
    """Per-scale outputs (lowest to highest resolution) if the model has them, else None."""
    fn = getattr(model, "forward_multiscale", None)
    return fn(x) if fn is not None else None


def image_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    return {k: float(v[0]) for k, v in batch_metrics(pred, target).items()}


def corrupt_image(x: torch.Tensor, kind: str, level: float, seed: int) -> torch.Tensor:
    return corrupt(x, kind, level, torch.Generator().manual_seed(seed))


# ------------------------------------------------------------------ latent assets
@dataclass
class PCAAsset:
    basis: PCABasis  # fitted on TRAIN-split latents only
    sample_coords: np.ndarray  # (N, 2) projections of the sample images on PC1, PC2


def load_pca_assets(path: Path) -> dict[str, PCAAsset]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return {
        name: PCAAsset(PCABasis(d["mean"], d["components"]), d["sample_coords"].numpy())
        for name, d in payload["models"].items()
    }


@torch.no_grad()
def truncated_reconstruction(
    model: nn.Module, x: torch.Tensor, basis: PCABasis, k: int
) -> torch.Tensor:
    """Decode the latent of ``x`` after keeping only its first ``k`` principal components."""
    return model.decode(truncate(model.encode(x), basis, k))


def default_interpolation_endpoints(labels: np.ndarray) -> tuple[int, int]:
    """Two *different* samples to start the interpolation from, preferring different classes.

    A is the first sample. B is the first sample of the class half a class-list away from A's
    class (airplane -> dog for CIFAR-10), so the default pair is visually distinct. With a single
    class it falls back to the second sample. Deterministic; both stay freely adjustable.
    """
    if len(labels) < 2:
        raise ValueError("need at least two samples to interpolate between")
    classes = sorted({int(c) for c in labels})
    if len(classes) < 2:
        return 0, 1
    target = classes[(classes.index(int(labels[0])) + len(classes) // 2) % len(classes)]
    return 0, int(np.flatnonzero(labels == target)[0])


@torch.no_grad()
def interpolate(
    model: nn.Module, x_a: torch.Tensor, x_b: torch.Tensor, alphas: list[float]
) -> list[torch.Tensor]:
    """Decode linear interpolations between two latents (alpha 0 -> a, 1 -> b)."""
    za, zb = model.encode(x_a), model.encode(x_b)
    return [model.decode((1 - a) * za + a * zb) for a in alphas]


# ------------------------------------------------------------------ saved results
def _read(path: Path) -> Any:
    return json.loads(path.read_text()) if path.is_file() else None


def load_saved_results(results_dir: Path) -> dict[str, Any]:
    """Saved JSON results per model plus the shared latent analysis (missing files -> None)."""
    results_dir = Path(results_dir)
    models = {}
    for name in MODEL_RUNS:
        d = run_dir(results_dir, name)
        models[name] = {
            key: _read(d / f"{key}.json")
            for key in ("summary", "eval_val", "eval_test", "efficiency", "robustness_val")
        }
    return {"models": models, "latent": _read(results_dir / "latent_analysis_val.json")}


def reconstruction_rows(res: dict[str, Any], split: str) -> list[dict[str, Any]]:
    rows = []
    for name, r in res["models"].items():
        ev = r.get(f"eval_{split}")
        if ev and r.get("summary"):
            rows.append(
                {
                    "model": name,
                    "parameters": r["summary"]["parameters"],
                    "MSE": ev["mse"],
                    "MAE": ev["mae"],
                    "PSNR (dB)": ev["psnr"],
                    "SSIM": ev["ssim"],
                    "images": ev["n_images"],
                }
            )
    return rows


def efficiency_rows(res: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for name, r in res["models"].items():
        e = r.get("efficiency")
        if e:
            lat = e["latency"]
            rows.append(
                {
                    "model": name,
                    "parameters": e["parameters"],
                    "FP32 size (MB)": e["size_mb_fp32"],
                    "MACs per image (millions)": e["flops"] / 2e6,
                    **{LATENCY_LABELS[k]: lat[k]["median_ms"] for k in LATENCY_LABELS if k in lat},
                }
            )
    return rows


def robustness_series(res: dict[str, Any], kind: str, metric: str = "psnr"):
    """(levels, {series: values}) including the corrupted input itself as a reference."""
    ref = next(
        (r["robustness_val"] for r in res["models"].values() if r.get("robustness_val")), None
    )
    if ref is None:
        return [], {}
    levels = sorted(ref[kind], key=float)
    series = {"corrupted input": [ref[kind][lv][f"input_{metric}"] for lv in levels]}
    for name, r in res["models"].items():
        if r.get("robustness_val"):
            series[name] = [r["robustness_val"][kind][lv][metric] for lv in levels]
    return [float(lv) for lv in levels], series


def probe_rows(res: dict[str, Any]) -> list[dict[str, Any]]:
    lat = res.get("latent")
    return (
        []
        if not lat
        else [
            {
                "representation": k,
                "features": v["n_features"],
                "val accuracy": v["eval_accuracy"],
                "std. error": v["eval_accuracy_stderr"],
            }
            for k, v in lat["probes"].items()
        ]
    )


def rank_rows(res: dict[str, Any]) -> list[dict[str, Any]]:
    lat = res.get("latent")
    return (
        []
        if not lat
        else [
            {
                "model": n,
                "effective rank": r["stats_train"]["effective_rank"],
                "participation ratio": r["stats_train"]["participation_ratio"],
                "dims for 95% variance": r["stats_train"]["dims_for_95pct_variance"],
                "random-init eff. rank": r["random_init_stats_train"]["effective_rank"],
            }
            for n, r in lat["results"].items()
        ]
    )


def truncation_series(res: dict[str, Any]) -> dict[str, dict[int, float]]:
    lat = res.get("latent")
    return (
        {}
        if not lat
        else {
            n: {int(k): v for k, v in r["truncation_val_mse"].items()}
            for n, r in lat["results"].items()
        }
    )
