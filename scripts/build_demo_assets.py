"""One-time, inference-only preparation of the local demo assets (writes only to demo/assets).

Reads the five CIFAR-10 training batches directly, selects a stratified sample of VALIDATION images
using the research split, and fits per-model PCA bases on TRAIN latents. No training occurs.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from asymmetric_autoencoder import demo_support as ds
from asymmetric_autoencoder.config import load_config
from asymmetric_autoencoder.data.datasets import split_indices
from asymmetric_autoencoder.demo_assets import (
    build_pca_asset,
    ensure_assets_dir,
    load_train_arrays,
    stratified_sample,
)
from asymmetric_autoencoder.utils.logging import get_logger, setup_logging
from asymmetric_autoencoder.utils.reproducibility import resolve_device

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/cifar-10-batches-py")
    parser.add_argument("--results-dir", default="experiments/results")
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "demo" / "assets"))
    parser.add_argument("--per-class", type=int, default=30, help="validation samples per class")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--threads", type=int, default=2, help="CPU threads (keeps the Mac cool)")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    setup_logging()
    log = get_logger("build_demo_assets")
    out_dir = ensure_assets_dir(args.out_dir, PROJECT_ROOT)
    results = Path(args.results_dir)
    missing = ds.missing_checkpoints(results)
    if missing:
        raise SystemExit(f"checkpoints not found: {[str(p) for p in missing]}")

    cfg = load_config(ds.run_dir(results, "baseline") / "config.yaml").data
    if cfg.name != "cifar10" or cfg.image_size != 32:
        raise SystemExit("the demo assets expect the frozen CIFAR-10, 32x32 configuration")
    images, labels, class_names = load_train_arrays(args.data_dir)
    train_idx, val_idx = split_indices(len(images), cfg.val_fraction, cfg.split_seed)
    train_idx, val_idx = np.asarray(train_idx), np.asarray(val_idx)
    sample_idx = stratified_sample(labels, val_idx, args.per_class)
    assert set(sample_idx.tolist()) <= set(val_idx.tolist())
    assert not set(sample_idx.tolist()) & set(train_idx.tolist())
    log.info("train=%d val=%d sample=%d", len(train_idx), len(val_idx), len(sample_idx))

    np.savez_compressed(
        out_dir / ds.SAMPLES_FILE,
        images=images[sample_idx],
        labels=labels[sample_idx],
        val_indices=sample_idx,
        class_names=np.array(class_names),
    )
    device = resolve_device(args.device)
    train_images, sample_images = images[train_idx], images[sample_idx]
    payload = {
        "meta": {
            "fit_split": "train",
            "n_train": int(len(train_idx)),
            "split_seed": int(cfg.split_seed),
            "val_fraction": float(cfg.val_fraction),
        },
        "models": {},
    }
    for name in ds.MODEL_RUNS:
        log.info("fitting PCA on train latents: %s (device %s)", name, device)
        payload["models"][name] = build_pca_asset(
            ds.load_model(results, name), train_images, sample_images, device
        )
    torch.save(payload, out_dir / ds.PCA_FILE)
    log.info("wrote %s", out_dir)


if __name__ == "__main__":
    main()
