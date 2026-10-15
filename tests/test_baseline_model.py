import pytest
import torch
import torch.nn.functional as F

from asymmetric_autoencoder.models.baseline_autoencoder import ConvAutoencoder, count_parameters


@pytest.mark.parametrize("in_ch", [1, 3])
@pytest.mark.parametrize("latent", [8, 64])
def test_shapes(in_ch, latent):
    model = ConvAutoencoder(in_channels=in_ch, image_size=32, latent_dim=latent)
    x = torch.rand(4, in_ch, 32, 32)
    z = model.encode(x)
    assert z.shape == (4, latent)
    out = model(x)
    assert out.shape == x.shape
    assert out.min() >= 0 and out.max() <= 1


def test_parameter_count_matches_manual_formula():
    # 3 -> 32 -> 64 -> 128 convs, 2048 -> 8 linear, and the mirrored decoder.
    m = ConvAutoencoder(in_channels=3, latent_dim=8)
    conv = lambda i, o: i * o * 9 + o  # noqa: E731
    expected = (
        conv(3, 32)
        + conv(32, 64)
        + conv(64, 128)
        + 2048 * 8
        + 8
        + 8 * 2048
        + 2048
        + conv(128, 64)
        + conv(64, 32)
        + conv(32, 3)
    )
    assert count_parameters(m) == expected


def test_backward_gives_gradients_to_all_parameters():
    model = ConvAutoencoder(in_channels=1, latent_dim=8)
    x = torch.rand(4, 1, 32, 32)
    F.mse_loss(model(x), x).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert any(p.grad.abs().sum() > 0 for p in model.parameters())


def test_rejects_bad_image_size():
    with pytest.raises(ValueError):
        ConvAutoencoder(image_size=30)


def test_overfits_tiny_batch():
    torch.manual_seed(0)
    model = ConvAutoencoder(in_channels=1, latent_dim=32)
    x = torch.rand(8, 1, 32, 32)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    first = F.mse_loss(model(x), x).item()
    for _ in range(300):
        loss = F.mse_loss(model(x), x)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.3 * first
