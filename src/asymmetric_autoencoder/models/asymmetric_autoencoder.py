"""Model B: asymmetric autoencoder (heavy residual encoder, light multiscale decoder).

Encoder: stem conv -> [ResBlock -> stride-2 conv] x (len(enc_channels) - 1) -> optional
shared-weight refinement -> Flatten -> Linear -> latent (optionally LayerNorm).
Decoder: Linear -> reshape -> [bilinear x2 upsample + conv] stages with narrow channels.
Each stage can emit a 1x1-conv image head (sigmoid), giving reconstructions at several
resolutions that are trained against area-downsampled copies of the input
(see ``models.losses.multiscale_mse``).

"Recursive refinement" here means exactly one ResBlock whose weights are *tied* and applied
``refine_steps`` times at the bottleneck resolution. It is distinct from recurrent processing
(no hidden state across time) and from ordinary stacked layers (no new weights per step).
``refine_steps=0`` (default) disables it.

Interface matches ``ConvAutoencoder``: ``forward(x) -> x_hat``, ``encode``, ``decode``,
``latent_dim``, ``in_channels``, ``scale_weights``; plus ``forward_multiscale`` returning the
list of outputs from lowest to highest resolution (the last one is ``forward(x)``).
"""

from __future__ import annotations

import torch
from torch import nn

from asymmetric_autoencoder.models.blocks import ResBlock, UpBlock, conv_bn_act


class AsymmetricAutoencoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        image_size: int = 32,
        latent_dim: int = 64,
        enc_channels: tuple[int, ...] = (64, 128, 256, 256),
        dec_channels: tuple[int, ...] = (32, 32, 16, 16),
        refine_steps: int = 0,
        multiscale: bool = True,
        scale_weights: tuple[float, ...] | None = None,
        latent_norm: bool = False,
    ) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if refine_steps < 0:
            raise ValueError("refine_steps must be >= 0")
        n_stages = len(enc_channels) - 1
        if n_stages < 1:
            raise ValueError("enc_channels needs at least 2 entries")
        if len(dec_channels) != len(enc_channels):
            raise ValueError("dec_channels must have the same length as enc_channels")
        if image_size % (2**n_stages) != 0:
            raise ValueError(f"image_size {image_size} must be divisible by {2**n_stages}")

        n_outputs = n_stages if multiscale else 1
        if scale_weights is None:
            # lowest -> highest resolution: 0.25, 0.5, 1.0 for three scales
            scale_weights = tuple(0.5 ** (n_outputs - 1 - i) for i in range(n_outputs))
        if len(scale_weights) != n_outputs:
            raise ValueError(f"scale_weights needs {n_outputs} entries, got {len(scale_weights)}")
        if any(w < 0 for w in scale_weights) or scale_weights[-1] <= 0:
            raise ValueError("scale_weights must be >= 0 with a positive final-scale weight")

        self.in_channels = in_channels
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.refine_steps = refine_steps
        self.multiscale = multiscale
        self.scale_weights = tuple(scale_weights)
        self._feat = image_size // 2**n_stages
        self._dec0 = dec_channels[0]

        # Encoder
        layers: list[nn.Module] = [conv_bn_act(in_channels, enc_channels[0])]
        for i in range(n_stages):
            layers.append(ResBlock(enc_channels[i]))
            layers.append(conv_bn_act(enc_channels[i], enc_channels[i + 1], stride=2))
        self.encoder_conv = nn.Sequential(*layers)
        self.refine = ResBlock(enc_channels[-1]) if refine_steps > 0 else None
        self.to_latent = nn.Linear(enc_channels[-1] * self._feat**2, latent_dim)
        self.latent_norm = nn.LayerNorm(latent_dim) if latent_norm else nn.Identity()

        # Decoder
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, dec_channels[0] * self._feat**2), nn.GELU()
        )
        self.up_blocks = nn.ModuleList(
            UpBlock(dec_channels[i], dec_channels[i + 1]) for i in range(n_stages)
        )
        self.heads = nn.ModuleList(
            nn.Conv2d(dec_channels[i + 1], in_channels, 1)
            for i in range(n_stages)
            if multiscale or i == n_stages - 1
        )
        self._head_stages = [i for i in range(n_stages) if multiscale or i == n_stages - 1]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder_conv(x)
        if self.refine is not None:
            for _ in range(self.refine_steps):
                h = self.refine(h)
        return self.latent_norm(self.to_latent(h.flatten(1)))

    def decode_multiscale(self, z: torch.Tensor) -> list[torch.Tensor]:
        h = self.from_latent(z).view(z.shape[0], self._dec0, self._feat, self._feat)
        outputs = []
        head_iter = iter(self.heads)
        for i, block in enumerate(self.up_blocks):
            h = block(h)
            if i in self._head_stages:
                outputs.append(torch.sigmoid(next(head_iter)(h)))
        return outputs

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decode_multiscale(z)[-1]

    def forward_multiscale(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.decode_multiscale(self.encode(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))
