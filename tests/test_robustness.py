import pytest
import torch

from asymmetric_autoencoder.evaluation.metrics import evaluate_reconstruction
from asymmetric_autoencoder.evaluation.robustness import corrupt, evaluate_corruption
from asymmetric_autoencoder.models.baseline_autoencoder import ConvAutoencoder


def _gen(seed=0):
    return torch.Generator().manual_seed(seed)


def test_level_zero_is_identity_copy():
    x = torch.rand(2, 3, 8, 8)
    for kind in ("gaussian", "salt_pepper"):
        y = corrupt(x, kind, 0.0, _gen())
        assert torch.equal(x, y) and y is not x


@pytest.mark.parametrize("kind", ["gaussian", "salt_pepper"])
def test_deterministic_and_in_range(kind):
    x = torch.rand(4, 3, 16, 16)
    a, b = corrupt(x, kind, 0.2, _gen(5)), corrupt(x, kind, 0.2, _gen(5))
    assert torch.equal(a, b)
    assert not torch.equal(a, corrupt(x, kind, 0.2, _gen(6)))
    assert a.min() >= 0 and a.max() <= 1 and a.shape == x.shape


def test_gaussian_noise_scale():
    x = torch.full((64, 1, 32, 32), 0.5)  # far from the clip boundaries for sigma 0.1
    err = corrupt(x, "gaussian", 0.1, _gen()) - x
    assert err.std().item() == pytest.approx(0.1, rel=0.05)
    assert err.mean().abs().item() < 0.01


def test_salt_pepper_fraction_and_channel_consistency():
    x = torch.full((32, 3, 32, 32), 0.5)
    y = corrupt(x, "salt_pepper", 0.1, _gen())
    changed = y != x
    assert changed.float().mean().item() == pytest.approx(0.1, abs=0.01)
    assert torch.equal(changed[:, 0], changed[:, 1]) and torch.equal(changed[:, 1], changed[:, 2])
    vals = y[changed].unique()
    assert set(vals.tolist()) == {0.0, 1.0}
    salt = (y[changed] == 1.0).float().mean().item()
    assert salt == pytest.approx(0.5, abs=0.03)


def test_invalid_arguments():
    x = torch.rand(1, 1, 8, 8)
    with pytest.raises(ValueError):
        corrupt(x, "blur", 0.1, _gen())
    with pytest.raises(ValueError):
        corrupt(x, "salt_pepper", 1.5, _gen())
    with pytest.raises(ValueError):
        corrupt(x, "gaussian", -0.1, _gen())


def test_zero_level_matches_clean_evaluation_and_models_see_same_noise():
    torch.manual_seed(0)
    cpu = torch.device("cpu")
    data = [(torch.rand(4, 1, 32, 32), torch.zeros(4)) for _ in range(3)]
    m1 = ConvAutoencoder(in_channels=1, latent_dim=8)
    clean = evaluate_reconstruction(m1, data, cpu)
    zero = evaluate_corruption(m1, data, cpu, "gaussian", 0.0)
    for k in ("mse", "mae", "psnr", "ssim"):
        assert zero[k] == pytest.approx(clean[k], rel=1e-6)
    assert zero["input_mse"] == pytest.approx(0.0, abs=1e-12)

    # identical corrupted inputs for two different models => identical input_* reference metrics
    m2 = ConvAutoencoder(in_channels=1, latent_dim=16)
    r1 = evaluate_corruption(m1, data, cpu, "salt_pepper", 0.1, seed=3)
    r2 = evaluate_corruption(m2, data, cpu, "salt_pepper", 0.1, seed=3)
    assert r1["input_mse"] == r2["input_mse"] and r1["input_ssim"] == r2["input_ssim"]
    assert r1["input_mse"] > 0
