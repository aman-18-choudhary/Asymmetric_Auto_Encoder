"""Download a dataset into the configured data root (never committed to Git)."""

import argparse

from asymmetric_autoencoder.config import DataConfig
from asymmetric_autoencoder.data.datasets import build_test_set, build_train_val
from asymmetric_autoencoder.utils.logging import get_logger, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="mnist", choices=["mnist", "fashion_mnist", "cifar10"])
    parser.add_argument("--root", default="data")
    args = parser.parse_args()
    setup_logging()
    cfg = DataConfig(name=args.name, root=args.root)
    splits = build_train_val(cfg, download=True)
    test = build_test_set(cfg, download=True)
    get_logger("download").info(
        "%s ready under %s: train=%d val=%d test=%d",
        args.name,
        args.root,
        len(splits.train),
        len(splits.val),
        len(test),
    )


if __name__ == "__main__":
    main()
