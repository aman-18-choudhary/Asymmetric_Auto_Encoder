import pytest
import torch
from torchvision import transforms
from torchvision.datasets import FakeData

from asymmetric_autoencoder.data.datasets import build_transform, make_loader, split_indices


def test_split_is_deterministic_and_disjoint():
    a = split_indices(1000, 0.1, seed=3)
    b = split_indices(1000, 0.1, seed=3)
    assert a == b
    train, val = a
    assert len(val) == 100 and len(train) == 900
    assert set(train).isdisjoint(val)
    assert sorted(train + val) == list(range(1000))


def test_split_changes_with_seed():
    assert split_indices(1000, 0.1, 0) != split_indices(1000, 0.1, 1)


def test_cifar_sized_split_is_45k_5k():
    train, val = split_indices(50_000, 0.1, 0)
    assert (len(train), len(val)) == (45_000, 5_000)


def test_split_validation():
    with pytest.raises(ValueError):
        split_indices(1, 0.1, 0)
    with pytest.raises(ValueError):
        split_indices(10, 1.0, 0)


def test_transform_resizes_and_scales_to_unit_range():
    from PIL import Image

    img = Image.new("L", (28, 28), color=255)
    out = build_transform(32)(img)
    assert out.shape == (1, 32, 32)
    assert out.dtype == torch.float32 and 0.0 <= out.min() and out.max() <= 1.0


def test_loader_batches_and_seeded_shuffle():
    ds = FakeData(size=40, image_size=(3, 32, 32), transform=transforms.ToTensor(), random_offset=0)
    x, y = next(iter(make_loader(ds, 16, True, seed=5)))
    assert x.shape == (16, 3, 32, 32) and y.shape == (16,)
    first = [next(iter(make_loader(ds, 16, True, seed=5)))[1].tolist() for _ in range(2)]
    assert first[0] == first[1]


class _FakeCifar(FakeData):
    """In-memory stand-in with the torchvision dataset constructor signature."""

    constructed: list[bool] = []

    def __init__(self, root, train, download, transform):
        _FakeCifar.constructed.append(train)
        super().__init__(
            size=200 if train else 40,
            image_size=(3, 32, 32),
            transform=transform,
            random_offset=0 if train else 10_000,
        )


@pytest.fixture
def fake_registry(monkeypatch):
    from asymmetric_autoencoder.data import datasets

    monkeypatch.setitem(datasets._REGISTRY, "cifar10", (_FakeCifar, 3))
    _FakeCifar.constructed = []
    return datasets


def _cfg(**kw):
    from asymmetric_autoencoder.config import DataConfig

    return DataConfig(name="cifar10", val_fraction=0.1, split_seed=7, **kw)


def test_build_train_val_sizes_overlap_and_source(fake_registry):
    s = fake_registry.build_train_val(_cfg())
    assert (len(s.train), len(s.val)) == (180, 20)
    assert set(s.train_indices).isdisjoint(s.val_indices)
    assert sorted(s.train_indices + s.val_indices) == list(range(200))
    assert s.in_channels == 3
    # both subsets index the same underlying (training) dataset
    assert s.train.dataset is s.val.dataset
    assert s.train[0][0].shape == (3, 32, 32)


def test_build_train_val_is_reproducible(fake_registry):
    a = fake_registry.build_train_val(_cfg())
    b = fake_registry.build_train_val(_cfg())
    assert a.train_indices == b.train_indices and a.val_indices == b.val_indices
    assert torch.equal(a.val[0][0], b.val[0][0])


def test_train_val_does_not_construct_test_set(fake_registry):
    fake_registry.build_train_val(_cfg())
    assert _FakeCifar.constructed == [True]
    test = fake_registry.build_test_set(_cfg())
    assert _FakeCifar.constructed == [True, False]
    assert len(test) == 40


def test_test_set_images_differ_from_train(fake_registry):
    s = fake_registry.build_train_val(_cfg())
    test = fake_registry.build_test_set(_cfg())
    train_first = torch.stack([s.train[i][0] for i in range(10)])
    assert not torch.equal(train_first, torch.stack([test[i][0] for i in range(10)]))
