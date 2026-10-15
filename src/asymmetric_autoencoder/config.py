"""Typed experiment configuration loaded from YAML.

Unknown keys and invalid values raise errors so that typos in a config cannot silently
change an experiment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

_DEVICES = ("auto", "cpu", "cuda", "mps")
_DATASETS = ("cifar10", "mnist", "fashion_mnist")


@dataclass
class DataConfig:
    name: str = "cifar10"
    root: str = "data"
    image_size: int = 32
    batch_size: int = 128
    num_workers: int = 0
    val_fraction: float = 0.1
    split_seed: int = 0  # fixed independently of the training seed so all models share a split

    def validate(self) -> None:
        if self.name not in _DATASETS:
            raise ValueError(f"data.name must be one of {_DATASETS}, got {self.name!r}")
        if self.split_seed < 0:
            raise ValueError("data.split_seed must be >= 0")
        if self.image_size <= 0 or self.batch_size <= 0:
            raise ValueError("data.image_size and data.batch_size must be positive")
        if self.num_workers < 0:
            raise ValueError("data.num_workers must be >= 0")
        if not 0.0 < self.val_fraction < 1.0:
            raise ValueError("data.val_fraction must be in (0, 1)")


@dataclass
class ModelConfig:
    latent_dim: int = 64
    channels: list[int] = field(default_factory=lambda: [32, 64, 128])

    def validate(self) -> None:
        if self.latent_dim <= 0:
            raise ValueError("model.latent_dim must be positive")
        if not self.channels or any(c <= 0 for c in self.channels):
            raise ValueError("model.channels must be a non-empty list of positive ints")


@dataclass
class TrainConfig:
    epochs: int = 30
    lr: float = 1e-3

    def validate(self) -> None:
        if self.epochs <= 0 or self.lr <= 0:
            raise ValueError("train.epochs and train.lr must be positive")


@dataclass
class ExperimentConfig:
    experiment_name: str = "debug"
    seed: int = 0
    device: str = "auto"
    deterministic: bool = False
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def validate(self) -> None:
        if not self.experiment_name:
            raise ValueError("experiment_name must be non-empty")
        if self.seed < 0:
            raise ValueError("seed must be >= 0")
        if self.device not in _DEVICES:
            raise ValueError(f"device must be one of {_DEVICES}, got {self.device!r}")
        self.data.validate()
        self.model.validate()
        self.train.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build(cls: type, raw: dict[str, Any], path: str) -> Any:
    if not isinstance(raw, dict):
        raise ValueError(f"{path or 'config'} must be a mapping, got {type(raw).__name__}")
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown key(s) in {path or 'config'}: {sorted(unknown)}")
    kwargs = dict(raw)
    if cls is ExperimentConfig:
        for name, sub in (("data", DataConfig), ("model", ModelConfig), ("train", TrainConfig)):
            if name in kwargs:
                kwargs[name] = _build(sub, kwargs[name], name)
    return cls(**kwargs)


def config_from_dict(raw: dict[str, Any]) -> ExperimentConfig:
    """Build and validate a config from a plain dict."""
    cfg = _build(ExperimentConfig, raw, "")
    cfg.validate()
    return cfg


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> ExperimentConfig:
    """Load a YAML config; top-level ``overrides`` (e.g. ``{"seed": 1}``) win over the file."""
    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if overrides:
        raw = {**raw, **overrides}
    return config_from_dict(raw)


def save_config(cfg: ExperimentConfig, path: str | Path) -> None:
    """Write the config next to experiment results so a run is reproducible."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg.to_dict(), handle, sort_keys=False)
