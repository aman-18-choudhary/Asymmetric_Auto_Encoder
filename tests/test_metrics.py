import math

import pytest
import torch

from asymmetric_autoencoder.evaluation.efficiency import count_flops, measure_latency
from asymmetric_autoencoder.evaluation.metrics import (
    evaluate_reconstruction,
    mae_per_image,
    mse_per_image,
    psnr_per_image,
    ssim_per_image,
)
from asymmetric_autoencoder.models.baseline_autoencoder import ConvAutoencoder


def test_mse_mae_known_values():
    a = torch.zeros(2, 1, 4, 4)
    b = torch.full((2, 1, 4, 4), 0.5)
    assert mse_per_image(a, b).tolist() == pytest.approx([0.25, 0.25])
    assert mae_per_image(a, b).tolist() == pytest.approx([0.5, 0.5])


def test_psnr_known_value_and_perfect_is_finite():
    a = torch.zeros(1, 1, 8, 8)
    b = torch.full((1, 1, 8, 8), 0.1)  # mse 0.01 -> 20 dB with data range 1
    assert psnr_per_image(a, b).item() == pytest.approx(20.0, abs=1e-4)
    assert math.isfinite(psnr_per_image(a, a).item())


def test_ssim_identity_and_degradation():
    torch.manual_seed(0)
    x = torch.rand(3, 3, 32, 32)
    assert ssim_per_image(x, x).tolist() == pytest.approx([1.0] * 3, abs=1e-5)
    noisy = (x + 0.3 * torch.randn_like(x)).clamp(0, 1)
    s = ssim_per_image(noisy, x)
    assert (s < 0.9).all() and (s > -1).all()


def test_ssim_ordering_follows_noise_level():
    torch.manual_seed(0)
    x = torch.rand(8, 1, 32, 32)
    lo = ssim_per_image((x + 0.05 * torch.randn_like(x)).clamp(0, 1), x).mean()
    hi = ssim_per_image((x + 0.3 * torch.randn_like(x)).clamp(0, 1), x).mean()
    assert lo > hi


def test_ssim_matches_reference_implementation_if_available():
    skimage = pytest.importorskip("skimage.metrics")
    import numpy as np

    torch.manual_seed(1)
    x = torch.rand(1, 1, 32, 32)
    y = (x + 0.1 * torch.randn_like(x)).clamp(0, 1)
    ref = skimage.structural_similarity(
        x[0, 0].numpy().astype(np.float64),
        y[0, 0].numpy().astype(np.float64),
        data_range=1.0,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False,
        win_size=11,
    )
    assert ssim_per_image(y, x).item() == pytest.approx(ref, abs=5e-3)


def test_ssim_rejects_small_or_mismatched_inputs():
    with pytest.raises(ValueError):
        ssim_per_image(torch.rand(1, 1, 8, 8), torch.rand(1, 1, 8, 8))
    with pytest.raises(ValueError):
        ssim_per_image(torch.rand(1, 1, 32, 32), torch.rand(1, 1, 32, 31))


def test_evaluate_reconstruction_and_corruption_scored_against_clean():
    torch.manual_seed(0)
    model = ConvAutoencoder(in_channels=1, latent_dim=8)
    data = [(torch.rand(4, 1, 32, 32), torch.zeros(4)) for _ in range(3)]
    clean = evaluate_reconstruction(model, data, torch.device("cpu"))
    assert clean["n_images"] == 12
    assert set(clean) >= {"mse", "mae", "psnr", "ssim"}
    seen = []

    def corrupt(x):
        seen.append(x.clone())
        return torch.zeros_like(x)

    evaluate_reconstruction(model, data, torch.device("cpu"), corrupt=corrupt)
    assert len(seen) == 3  # corruption applied to every batch; target stays the clean image


def test_flops_scale_with_width_and_latency_is_positive():
    small = ConvAutoencoder(in_channels=3, latent_dim=8, channels=(8, 16, 32))
    big = ConvAutoencoder(in_channels=3, latent_dim=8, channels=(16, 32, 64))
    assert 0 < count_flops(small, (3, 32, 32)) < count_flops(big, (3, 32, 32))
    lat = measure_latency(small, (3, 32, 32), torch.device("cpu"), 4, warmup=1, iters=5)
    assert lat["median_ms"] > 0 and lat["per_image_ms"] == pytest.approx(lat["median_ms"] / 4)
