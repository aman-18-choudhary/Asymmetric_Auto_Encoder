import random

import numpy as np
import pytest
import torch

from asymmetric_autoencoder.utils.reproducibility import (
    environment_info,
    make_generator,
    resolve_device,
    seed_everything,
)


def _draw():
    return random.random(), float(np.random.rand()), torch.rand(3).tolist()


def test_seed_everything_is_repeatable():
    seed_everything(123)
    a = _draw()
    seed_everything(123)
    assert _draw() == a


def test_different_seeds_differ():
    seed_everything(1)
    a = _draw()
    seed_everything(2)
    assert _draw() != a


def test_invalid_seed():
    with pytest.raises(ValueError):
        seed_everything(-1)


def test_generator_repeatable():
    assert torch.equal(
        torch.rand(4, generator=make_generator(5)), torch.rand(4, generator=make_generator(5))
    )


def test_resolve_device():
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device("auto").type in {"cpu", "cuda", "mps"}
    with pytest.raises(ValueError):
        resolve_device("tpu")


def test_environment_info_keys():
    info = environment_info()
    assert {"python", "torch", "numpy", "platform"} <= set(info)
