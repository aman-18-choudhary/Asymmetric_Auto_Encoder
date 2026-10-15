"""Logging conventions: one stream handler, optional file handler, shared format."""

from __future__ import annotations

import logging
from pathlib import Path

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
ROOT_NAME = "asymmetric_autoencoder"


def setup_logging(level: int = logging.INFO, log_file: str | Path | None = None) -> logging.Logger:
    """Configure the package logger (idempotent) and return it.

    Library modules should call ``get_logger(__name__)``; only entry points call this.
    """
    logger = logging.getLogger(ROOT_NAME)
    logger.setLevel(level)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter(LOG_FORMAT)
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the package namespace."""
    if name == ROOT_NAME or name.startswith(ROOT_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT_NAME}.{name}")
