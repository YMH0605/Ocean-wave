"""3D CNN wave forecast model (the paper's Section 3.2).

Straight encoder of Conv3d blocks with max pooling, then 3D transposed
convolutions to restore the spatial size, then the time collapse and a 1x1
head. No skip connections -- that absence is the whole point of having this
model as the comparison against the U-Net.
"""

from __future__ import annotations

import torch.nn as nn

from .blocks import ConvBlock3d, TimeCollapse


class CNN3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        lookback: int = 48,
        base_channels: int = 32,
        depth: int = 3,
        time_collapse: str = "conv",
        out_channels: int = 1,
    ):
        super().__init__()
        chans = [base_channels * (2 ** i) for i in range(depth + 1)]

        encoder = []
        prev = in_channels
        for c in chans[:-1]:
            encoder += [ConvBlock3d(prev, c), nn.MaxPool3d(2)]
            prev = c
        encoder.append(ConvBlock3d(prev, chans[-1]))
        self.encoder = nn.Sequential(*encoder)

        decoder = []
        for i in range(depth, 0, -1):
            decoder += [
                nn.ConvTranspose3d(chans[i], chans[i - 1], kernel_size=2, stride=2,
                                   bias=False),
                nn.BatchNorm3d(chans[i - 1]),
                nn.ReLU(inplace=True),
            ]
        self.decoder = nn.Sequential(*decoder)

        self.collapse = TimeCollapse(chans[0], lookback, time_collapse)
        self.head = nn.Conv2d(chans[0], out_channels, 1)

    def forward(self, x, ocean=None):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.collapse(x)
        x = self.head(x)
        return x if ocean is None else x * ocean
