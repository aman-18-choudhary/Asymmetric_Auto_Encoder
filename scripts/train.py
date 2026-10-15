"""Train the configured autoencoder (baseline or asymmetric) with the shared Trainer."""

import argparse
from pathlib import Path

from asymmetric_autoencoder.config import load_config
from asymmetric_autoencoder.data.datasets import build_train_val, make_loader
from asymmetric_autoencoder.models.factory import build_model
from asymmetric_autoencoder.training.trainer import Trainer, run_dir_for
from asymmetric_autoencoder.utils.logging import get_logger, setup_logging
from asymmetric_autoencoder.utils.reproducibility import resolve_device, seed_everything


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value}")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mnist_debug.yaml")
    parser.add_argument("--seed", type=int, default=None, help="override seed")
    parser.add_argument("--epochs", type=_positive_int, default=None, help="override train.epochs")
    parser.add_argument(
        "--max-batches",
        type=_positive_int,
        default=None,
        help="cap batches per epoch (debugging only)",
    )
    parser.add_argument("--run-dir", default=None, help="default: <output_dir>/<name>_seed<seed>")
    parser.add_argument("--resume", action="store_true", help="continue from checkpoints/last.pt")
    args = parser.parse_args()

    overrides = {}
    if args.seed is not None:
        overrides["seed"] = args.seed
    cfg = load_config(args.config, overrides)
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
        cfg.validate()

    run_dir = Path(args.run_dir) if args.run_dir else run_dir_for(cfg)
    setup_logging(log_file=run_dir / "train.log")
    log = get_logger("train")
    seed_everything(cfg.seed, cfg.deterministic)
    device = resolve_device(cfg.device)

    splits = build_train_val(cfg.data)
    model = build_model(cfg.model, splits.in_channels, cfg.data.image_size)
    train_loader = make_loader(
        splits.train, cfg.data.batch_size, True, cfg.seed, cfg.data.num_workers
    )
    val_loader = make_loader(splits.val, cfg.data.batch_size, False, cfg.seed, cfg.data.num_workers)
    trainer = Trainer(
        model, train_loader, val_loader, cfg, device, run_dir=run_dir, max_batches=args.max_batches
    )
    trainer.fit(resume=args.resume)
    log.info("run directory: %s", run_dir)


if __name__ == "__main__":
    main()
