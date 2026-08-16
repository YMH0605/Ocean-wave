"""Shared building blocks for the wave forecast models."""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock3d(nn.Module):
    """Conv3d -> BatchNorm -> ReLU, twice.

    Matches the paper's "Conv Block": BN sits between the convolution and the
    activation.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3):
        super().__init__()
        pad = kernel // 2
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel, padding=pad, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel, padding=pad, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class TimeCollapse(nn.Module):
    """Reduce (B, C, T, H, W) to (B, C, H, W).

    The source paper never states how the time axis disappears between its 3D
    feature maps and its single-frame output. Three reasonable readings are
    implemented; `conv` (a learned weighting over lead times) is the default.
    """

    def __init__(self, channels: int, time_steps: int, mode: str = "conv"):
        super().__init__()
        self.mode = mode
        if mode == "conv":
            self.reduce = nn.Conv3d(channels, channels, (time_steps, 1, 1))
        elif mode not in ("last", "mean"):
            raise ValueError(f"unknown time collapse mode {mode!r}")

    def forward(self, x):
        if self.mode == "last":
            return x[:, :, -1]
        if self.mode == "mean":
            return x.mean(dim=2)
        return self.reduce(x).squeeze(2)


def masked_output(x: torch.Tensor, ocean: torch.Tensor | None) -> torch.Tensor:
    """Zero the land points of a prediction, if a mask is supplied."""
    return x if ocean is None else x * ocean
