"""Construct a model from configuration."""

from __future__ import annotations

from torch import nn

from asymmetric_autoencoder.config import ModelConfig
from asymmetric_autoencoder.models.asymmetric_autoencoder import AsymmetricAutoencoder
from asymmetric_autoencoder.models.baseline_autoencoder import ConvAutoencoder


def build_model(cfg: ModelConfig, in_channels: int, image_size: int) -> nn.Module:
    """Build the autoencoder selected by ``cfg.type``."""
    if cfg.type == "baseline":
        return ConvAutoencoder(in_channels, image_size, cfg.latent_dim, tuple(cfg.channels))
    if cfg.type == "asymmetric":
        return AsymmetricAutoencoder(
            in_channels,
            image_size,
            cfg.latent_dim,
            enc_channels=tuple(cfg.enc_channels),
            dec_channels=tuple(cfg.dec_channels),
            refine_steps=cfg.refine_steps,
            multiscale=cfg.multiscale,
            scale_weights=tuple(cfg.scale_weights) if cfg.scale_weights is not None else None,
            latent_norm=cfg.latent_norm,
        )
    raise ValueError(f"unknown model type {cfg.type!r}")
