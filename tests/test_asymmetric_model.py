import pytest
import torch
import torch.nn.functional as F

from asymmetric_autoencoder.config import ModelConfig
from asymmetric_autoencoder.models.asymmetric_autoencoder import AsymmetricAutoencoder
from asymmetric_autoencoder.models.baseline_autoencoder import ConvAutoencoder
from asymmetric_autoencoder.models.factory import build_model
from asymmetric_autoencoder.models.losses import downsample_target, model_loss, multiscale_mse
from asymmetric_autoencoder.models.summary import count_parameters, parameter_breakdown


@pytest.mark.parametrize("in_ch", [1, 3])
@pytest.mark.parametrize("latent", [8, 64])
def test_shapes_and_range(in_ch, latent):
    m = AsymmetricAutoencoder(in_channels=in_ch, latent_dim=latent).eval()
    x = torch.rand(4, in_ch, 32, 32)
    assert m.encode(x).shape == (4, latent)
    outs = m.forward_multiscale(x)
    assert [tuple(o.shape[-2:]) for o in outs] == [(8, 8), (16, 16), (32, 32)]
    assert all(o.shape[:2] == (4, in_ch) for o in outs)
    assert all(o.min() >= 0 and o.max() <= 1 for o in outs)
    assert torch.equal(m(x), outs[-1])


def test_interface_parity_with_baseline():
    a, b = ConvAutoencoder(latent_dim=16), AsymmetricAutoencoder(latent_dim=16)
    x = torch.rand(2, 3, 32, 32)
    for m in (a, b):
        z = m.encode(x)
        assert z.shape == (2, 16) and m.latent_dim == 16 and m.in_channels == 3
        assert m.decode(z).shape == x.shape == m(x).shape
        assert hasattr(m, "scale_weights")


def test_gradients_reach_every_parameter():
    m = AsymmetricAutoencoder(in_channels=1, latent_dim=8, refine_steps=2, latent_norm=True)
    x = torch.rand(4, 1, 32, 32)
    model_loss(m, x).backward()
    missing = [n for n, p in m.named_parameters() if p.grad is None]
    assert not missing
    assert all(torch.isfinite(p.grad).all() for p in m.parameters())


def test_single_scale_variant_has_no_aux_heads():
    multi = AsymmetricAutoencoder(latent_dim=8)
    single = AsymmetricAutoencoder(latent_dim=8, multiscale=False)
    assert len(multi.heads) == 3 and len(single.heads) == 1
    outs = single.forward_multiscale(torch.rand(2, 3, 32, 32))
    assert len(outs) == 1 and outs[0].shape[-1] == 32
    assert single.scale_weights == (1.0,)
    assert count_parameters(single) < count_parameters(multi)


def test_refinement_shares_weights():
    base = count_parameters(AsymmetricAutoencoder(latent_dim=8, refine_steps=0))
    k1 = count_parameters(AsymmetricAutoencoder(latent_dim=8, refine_steps=1))
    k3 = count_parameters(AsymmetricAutoencoder(latent_dim=8, refine_steps=3))
    assert k1 > base and k1 == k3  # tied weights: extra steps add compute, not parameters


def test_latent_norm_option():
    m = AsymmetricAutoencoder(latent_dim=16, latent_norm=True).eval()
    z = m.encode(torch.rand(8, 3, 32, 32))
    assert torch.allclose(z.mean(1), torch.zeros(8), atol=1e-4)


def test_invalid_arguments():
    with pytest.raises(ValueError):
        AsymmetricAutoencoder(image_size=30)
    with pytest.raises(ValueError):
        AsymmetricAutoencoder(dec_channels=(32, 16))
    with pytest.raises(ValueError):
        AsymmetricAutoencoder(scale_weights=(1.0, 1.0))
    with pytest.raises(ValueError):
        AsymmetricAutoencoder(scale_weights=(1.0, 1.0, 0.0))


def test_decoder_is_lighter_than_encoder_and_breakdown_sums():
    m = AsymmetricAutoencoder(latent_dim=64)
    parts = parameter_breakdown(m)
    assert sum(parts.values()) == count_parameters(m)
    decoder = parts["from_latent"] + parts["up_blocks"] + parts["heads"]
    encoder = parts["encoder_conv"] + parts["to_latent"]
    assert decoder < encoder / 10


def test_overfits_tiny_batch():
    """Learning check on smooth structured images.

    Pure-noise memorisation is deliberately not used: the lightweight bilinear-upsampling decoder
    memorises i.i.d. noise far more slowly than the baseline (see the Phase 3 report), which is a
    property of the design, not a training bug.
    """
    torch.manual_seed(0)
    m = AsymmetricAutoencoder(
        in_channels=1,
        latent_dim=32,
        enc_channels=(16, 32, 32, 32),
        dec_channels=(16, 16, 8, 8),
    )
    x = F.interpolate(torch.rand(8, 1, 6, 6), size=32, mode="bilinear", align_corners=False)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    first = model_loss(m, x).item()
    for _ in range(150):
        loss = model_loss(m, x)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1 * first


# ---- losses ----
def test_downsample_target_is_block_average():
    x = torch.arange(16.0).view(1, 1, 4, 4)
    out = downsample_target(x, (2, 2))
    expected = torch.tensor([[[[2.5, 4.5], [10.5, 12.5]]]])
    assert torch.allclose(out, expected)
    assert downsample_target(x, (4, 4)) is x


def test_downsample_rejects_non_integer_ratio():
    with pytest.raises(ValueError):
        downsample_target(torch.rand(1, 1, 10, 10), (4, 4))


def test_multiscale_loss_values_and_weights():
    x = torch.rand(2, 1, 8, 8)
    perfect = [downsample_target(x, (2, 2)), downsample_target(x, (4, 4)), x]
    total, per = multiscale_mse(perfect, x, (0.25, 0.5, 1.0))
    assert total.item() == pytest.approx(0.0, abs=1e-12) and len(per) == 3
    off = [o + 0.1 for o in perfect]
    total, per = multiscale_mse(off, x, (0.25, 0.5, 1.0))
    assert [p.item() for p in per] == pytest.approx([0.01] * 3, rel=1e-4)
    assert total.item() == pytest.approx(1.75 * 0.01, rel=1e-4)
    only_final, _ = multiscale_mse(off, x, (0.0, 0.0, 1.0))
    assert only_final.item() == pytest.approx(F.mse_loss(off[-1], x).item())


def test_multiscale_loss_length_mismatch():
    with pytest.raises(ValueError):
        multiscale_mse([torch.rand(1, 1, 4, 4)], torch.rand(1, 1, 4, 4), (1.0, 1.0))


def test_model_loss_baseline_is_plain_mse():
    m = ConvAutoencoder(in_channels=1, latent_dim=8)
    x = torch.rand(4, 1, 32, 32)
    assert model_loss(m, x).item() == pytest.approx(F.mse_loss(m(x), x).item())


# ---- factory / config ----
def test_factory_builds_both_models():
    assert isinstance(build_model(ModelConfig(), 3, 32), ConvAutoencoder)
    m = build_model(ModelConfig(type="asymmetric", latent_dim=16, refine_steps=2), 1, 32)
    assert isinstance(m, AsymmetricAutoencoder) and m.refine_steps == 2 and m.in_channels == 1


@pytest.mark.parametrize(
    "kw",
    [
        {"type": "vae"},
        {"type": "asymmetric", "dec_channels": [8, 8]},
        {"type": "asymmetric", "scale_weights": [1.0]},
        {"type": "asymmetric", "refine_steps": -1},
    ],
)
def test_model_config_validation(kw):
    with pytest.raises(ValueError):
        ModelConfig(**kw).validate()


def test_asymmetric_config_file_builds_model():
    from pathlib import Path

    from asymmetric_autoencoder.config import load_config

    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "asymmetric.yaml")
    m = build_model(cfg.model, 3, cfg.data.image_size)
    assert m(torch.rand(2, 3, 32, 32)).shape == (2, 3, 32, 32)
