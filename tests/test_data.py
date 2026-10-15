import pytest
import torch
from torchvision import transforms
from torchvision.datasets import FakeData

from asymmetric_autoencoder.data.datasets import build_transform, make_loader, split_indices
from asymmetric_autoencoder.models.baseline_autoencoder import ConvAutoencoder
from asymmetric_autoencoder.training.simple import evaluate_mse, train_reconstruction


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


def test_minimal_training_loop_runs_and_reduces_loss():
    torch.manual_seed(0)
    ds = FakeData(size=32, image_size=(1, 32, 32), transform=transforms.ToTensor())
    loader = make_loader(ds, 16, False, seed=0)
    model = ConvAutoencoder(in_channels=1, latent_dim=16)
    cpu = torch.device("cpu")
    before = evaluate_mse(model, loader, cpu)
    history = train_reconstruction(model, loader, cpu, epochs=15, lr=2e-3)
    assert len(history) == 15
    assert evaluate_mse(model, loader, cpu) < before
