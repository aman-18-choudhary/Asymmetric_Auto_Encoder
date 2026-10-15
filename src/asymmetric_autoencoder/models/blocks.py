"""Reusable convolutional building blocks for the asymmetric autoencoder."""

from __future__ import annotations

import torch
from torch import nn


def conv_bn_act(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    """3x3 convolution -> BatchNorm -> GELU."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.GELU(),
    )


class ResBlock(nn.Module):
    """Two 3x3 convolutions with an identity skip connection (shape preserving)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.body(x))


class UpBlock(nn.Module):
    """Bilinear x2 upsampling followed by a 3x3 convolution (avoids checkerboard artifacts)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = conv_bn_act(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.up(x))
