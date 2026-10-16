import math
from pathlib import Path

import pytest
import torch

from asymmetric_autoencoder.evaluation import latent_analysis as la
from asymmetric_autoencoder.models.baseline_autoencoder import ConvAutoencoder


def test_effective_rank_isotropic_vs_low_rank():
    g = torch.Generator().manual_seed(0)
    iso = torch.randn(5000, 16, generator=g)
    assert la.effective_rank(la.covariance_spectrum(iso)) == pytest.approx(16, abs=0.5)
    low = torch.randn(5000, 2, generator=g) @ torch.randn(2, 16, generator=g)
    stats = la.latent_statistics(low + 1e-4 * torch.randn(5000, 16, generator=g))
    assert stats["effective_rank"] < 2.2 and stats["dims_for_99pct_variance"] == 2


def test_participation_ratio_bounds_and_collapse_flags():
    eig = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
    assert la.participation_ratio(eig) == pytest.approx(4.0)
    z = torch.randn(1000, 8)
    z[:, 3:] = 0.0  # five dead dimensions
    s = la.latent_statistics(z)
    assert s["n_low_variance_dims"] == 5 and s["effective_rank"] < 3.1


def test_pca_basis_is_orthonormal_and_sorted():
    z = torch.randn(2000, 6) * torch.tensor([5.0, 3.0, 2.0, 1.0, 0.5, 0.1])
    b = la.fit_pca(z)
    assert torch.allclose(b.components.T @ b.components, torch.eye(6), atol=1e-4)
    var = la.project(z, b, 6).var(0)
    assert (var[:-1] >= var[1:] - 1e-4).all()


def test_truncate_full_rank_is_identity_and_zero_is_mean():
    z = torch.randn(300, 8)
    b = la.fit_pca(z)
    assert torch.allclose(la.truncate(z, b, 8), z, atol=1e-4)
    assert torch.allclose(la.truncate(z, b, 0), b.mean.expand_as(z))
    err = [(la.truncate(z, b, k) - z).pow(2).mean().item() for k in (1, 2, 4, 8)]
    assert all(a >= c - 1e-9 for a, c in zip(err, err[1:], strict=False))


def test_truncation_curve_full_k_matches_plain_reconstruction():
    torch.manual_seed(0)
    cpu = torch.device("cpu")
    model = ConvAutoencoder(in_channels=1, latent_dim=8).eval()
    data = [(torch.rand(8, 1, 32, 32), torch.zeros(8)) for _ in range(3)]
    z, _ = la.extract_latents(model, data, cpu)
    assert z.shape == (24, 8)
    curve = la.truncation_curve(model, data, cpu, la.fit_pca(z), [0, 4, 8])
    with torch.no_grad():
        plain = torch.stack([(model(x) - x).pow(2).mean() for x, _ in data]).mean().item()
    assert curve[8] == pytest.approx(plain, rel=1e-4)


def test_standardize_uses_train_statistics_only():
    tr = torch.randn(500, 3) * 4 + 7
    ev = torch.randn(100, 3) * 4 + 7
    s_tr, s_ev = la.standardize(tr, ev)
    assert s_tr.mean(0).abs().max() < 1e-4 and (s_tr.std(0) - 1).abs().max() < 1e-4
    assert not torch.allclose(s_ev.mean(0), torch.zeros(3), atol=1e-6)  # not refitted on eval


def test_probe_separates_gaussians_and_is_at_chance_for_random_labels():
    g = torch.Generator().manual_seed(1)
    centers = torch.randn(4, 10, generator=g) * 3

    def make(n):
        y = torch.randint(0, 4, (n,), generator=g)
        return centers[y] + torch.randn(n, 10, generator=g), y

    (xt, yt), (xe, ye) = make(2000), make(1000)
    good = la.fit_linear_probe(xt, yt, xe, ye, n_classes=4)
    assert good["eval_accuracy"] > 0.9 and good["chance"] == 0.25
    rand_y = torch.randint(0, 4, (2000,), generator=g)
    rand_ye = torch.randint(0, 4, (1000,), generator=g)
    bad = la.fit_linear_probe(xt, rand_y, xe, rand_ye, n_classes=4)
    assert abs(bad["eval_accuracy"] - 0.25) < 0.06
    assert good["n_train"] == 2000 and good["n_eval"] == 1000
    assert math.isclose(
        good["eval_accuracy_stderr"],
        (good["eval_accuracy"] * (1 - good["eval_accuracy"]) / 1000) ** 0.5,
    )


def test_analysis_script_never_loads_the_test_split():
    text = (Path(__file__).resolve().parents[1] / "scripts" / "latent_analysis.py").read_text()
    for forbidden in ("build_test_set", "allow-test", "train=False", "eval_test"):
        assert forbidden not in text


class _LinearAE(torch.nn.Module):
    """Linear autoencoder on 1x4x4 images whose latent carries all the information."""

    def __init__(self):
        super().__init__()
        self.register_buffer("a", torch.linalg.qr(torch.randn(16, 4))[0])  # orthonormal columns

    def encode(self, x):
        return x.flatten(1) @ self.a

    def decode(self, z):
        return (z @ self.a.T).view(-1, 1, 4, 4)


def test_truncation_error_decreases_with_k_when_latent_matters():
    torch.manual_seed(0)
    cpu = torch.device("cpu")
    model = _LinearAE()
    scale = torch.tensor([4.0, 2.0, 1.0, 0.5])
    codes = torch.randn(400, 4) * scale
    imgs = (codes @ model.a.T).view(-1, 1, 4, 4)  # images lie exactly in the decoder range
    data = [(imgs[i : i + 100], torch.zeros(100)) for i in range(0, 400, 100)]
    z, _ = la.extract_latents(model, data, cpu)
    curve = la.truncation_curve(model, data, cpu, la.fit_pca(z), [0, 1, 2, 3, 4])
    vals = [curve[k] for k in (0, 1, 2, 3, 4)]
    assert vals[-1] == pytest.approx(0.0, abs=1e-8)
    assert all(a > b for a, b in zip(vals, vals[1:], strict=False))
