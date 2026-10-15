from pathlib import Path

import pytest

from asymmetric_autoencoder.config import (
    ExperimentConfig,
    config_from_dict,
    load_config,
    save_config,
)

BASE = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"


def test_base_config_loads_with_expected_defaults():
    cfg = load_config(BASE)
    assert cfg.seed == 0
    assert cfg.data.name == "cifar10"
    assert cfg.data.image_size == 32


def test_override_wins():
    assert load_config(BASE, {"seed": 7}).seed == 7


def test_round_trip(tmp_path):
    cfg = load_config(BASE)
    out = tmp_path / "cfg.yaml"
    save_config(cfg, out)
    assert load_config(out) == cfg


def test_unknown_key_rejected():
    with pytest.raises(ValueError, match="Unknown"):
        config_from_dict({"sed": 1})
    with pytest.raises(ValueError, match="Unknown"):
        config_from_dict({"data": {"batchsize": 4}})


@pytest.mark.parametrize(
    "raw",
    [
        {"seed": -1},
        {"device": "tpu"},
        {"data": {"val_fraction": 1.5}},
        {"data": {"name": "imagenet"}},
        {"data": {"batch_size": 0}},
    ],
)
def test_invalid_values_rejected(raw):
    with pytest.raises(ValueError):
        config_from_dict(raw)


def test_defaults_valid():
    ExperimentConfig().validate()


def test_model_and_train_sections_and_split_seed():
    cfg = config_from_dict(
        {
            "data": {"split_seed": 4},
            "model": {"latent_dim": 16, "channels": [8, 16]},
            "train": {"epochs": 2, "lr": 0.01},
        }
    )
    assert cfg.data.split_seed == 4 and cfg.model.channels == [8, 16] and cfg.train.epochs == 2


@pytest.mark.parametrize(
    "raw",
    [
        {"model": {"latent_dim": 0}},
        {"model": {"channels": []}},
        {"train": {"lr": 0}},
        {"model": {"depth": 3}},
    ],
)
def test_invalid_model_train_rejected(raw):
    with pytest.raises(ValueError):
        config_from_dict(raw)


@pytest.mark.parametrize("name", ["baseline.yaml", "mnist_debug.yaml"])
def test_shipped_configs_load(name):
    load_config(BASE.parent / name)
