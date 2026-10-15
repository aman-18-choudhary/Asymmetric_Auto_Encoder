"""Short baseline autoencoder run for Phase 2 verification (not a full experiment)."""

import argparse
import json

from asymmetric_autoencoder.config import load_config
from asymmetric_autoencoder.data.datasets import build_splits, make_loader
from asymmetric_autoencoder.models.baseline_autoencoder import ConvAutoencoder, count_parameters
from asymmetric_autoencoder.training.simple import evaluate_mse, train_reconstruction
from asymmetric_autoencoder.utils.logging import get_logger, setup_logging
from asymmetric_autoencoder.utils.reproducibility import (
    environment_info,
    resolve_device,
    seed_everything,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mnist_debug.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="override train.epochs")
    parser.add_argument("--max-batches", type=int, default=None, help="cap batches per epoch")
    args = parser.parse_args()

    setup_logging()
    log = get_logger("train_baseline")
    cfg = load_config(args.config)
    seed_everything(cfg.seed, cfg.deterministic)
    device = resolve_device(cfg.device)
    splits = build_splits(cfg.data)
    model = ConvAutoencoder(
        splits.in_channels, cfg.data.image_size, cfg.model.latent_dim, tuple(cfg.model.channels)
    )
    log.info("device=%s params=%d env=%s", device, count_parameters(model), environment_info())

    train_loader = make_loader(
        splits.train, cfg.data.batch_size, True, cfg.seed, cfg.data.num_workers
    )
    val_loader = make_loader(splits.val, cfg.data.batch_size, False, cfg.seed, cfg.data.num_workers)
    epochs = args.epochs or cfg.train.epochs
    history = train_reconstruction(
        model, train_loader, device, epochs, cfg.train.lr, args.max_batches
    )
    val_mse = evaluate_mse(model, val_loader, device, args.max_batches)
    log.info("final: %s", json.dumps({"train_mse": history, "val_mse": val_mse}))


if __name__ == "__main__":
    main()
