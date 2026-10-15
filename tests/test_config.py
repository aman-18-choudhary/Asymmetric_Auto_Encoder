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
