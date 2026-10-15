"""Model A: conventional symmetric convolutional autoencoder (baseline).

Encoder: ``len(channels)`` stride-2 3x3 convolutions with ReLU, then Flatten -> Linear -> latent.
Decoder: a mirrored Linear -> reshape -> transposed convolutions, ending in a Sigmoid so outputs
lie in [0, 1] like the inputs. Trained with plain MSE.

Interface (to be shared by later models): ``forward(x) -> x_hat``, ``encode(x) -> z``,
``decode(z) -> x_hat``, attributes ``latent_dim`` and ``in_channels``.
"""

from __future__ import annotations

import torch
from torch import nn


class ConvAutoencoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        image_size: int = 32,
        latent_dim: int = 64,
        channels: tuple[int, ...] = (32, 64, 128),
    ) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        factor = 2 ** len(channels)
        if image_size % factor != 0:
            raise ValueError(f"image_size {image_size} must be divisible by {factor}")
        self.in_channels = in_channels
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.channels = tuple(channels)
        self._feat = image_size // factor
        flat = channels[-1] * self._feat * self._feat

        enc: list[nn.Module] = []
        prev = in_channels
        for c in channels:
            enc += [nn.Conv2d(prev, c, 3, stride=2, padding=1), nn.ReLU(inplace=True)]
            prev = c
        self.encoder_conv = nn.Sequential(*enc)
        self.to_latent = nn.Linear(flat, latent_dim)

        self.from_latent = nn.Linear(latent_dim, flat)
        dec: list[nn.Module] = []
        rev = list(channels[::-1]) + [in_channels]
        for i in range(len(rev) - 1):
            dec.append(
                nn.ConvTranspose2d(rev[i], rev[i + 1], 3, stride=2, padding=1, output_padding=1)
            )
            if i < len(rev) - 2:
                dec.append(nn.ReLU(inplace=True))
        dec.append(nn.Sigmoid())
        self.decoder_conv = nn.Sequential(*dec)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.to_latent(self.encoder_conv(x).flatten(1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.from_latent(z).view(z.shape[0], self.channels[-1], self._feat, self._feat)
        return self.decoder_conv(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
