"""3D U-Net wave forecast model (the paper's Section 3.4).

Encoder blocks double the channel count and halve all three dimensions; the
decoder mirrors that with trilinear interpolation, as the paper specifies
("trillinear interpolate and Conv Blocks"). Skip connections join matching
resolutions. A final 3x3x3 convolution precedes the time collapse that turns
the (T, H, W) volume into the single (H, W) forecast frame.

With the default 48-hour lookback on the 32x48 Hatteras grid the resolutions
run 48x32x48 -> 24x16x24 -> 12x8x12 -> 6x4x6, all exact, which is why the
domain was chosen divisible by 8 in every axis.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvBlock3d, TimeCollapse


class UNet3D(nn.Module):
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
        self.depth = depth

        chans = [base_channels * (2 ** i) for i in range(depth + 1)]

        self.encoders = nn.ModuleList()
        prev = in_channels
        for c in chans[:-1]:
            self.encoders.append(ConvBlock3d(prev, c))
            prev = c
        self.pool = nn.MaxPool3d(2)

        self.bottleneck = ConvBlock3d(chans[-2], chans[-1])

        self.decoders = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            # input is upsampled bottleneck/decoder output concatenated with
            # the matching encoder feature map
            self.decoders.append(ConvBlock3d(chans[i + 1] + chans[i], chans[i]))

        self.final_conv = nn.Conv3d(chans[0], chans[0], 3, padding=1)
        self.collapse = TimeCollapse(chans[0], lookback, time_collapse)
        self.head = nn.Conv2d(chans[0], out_channels, 1)

    def forward(self, x, ocean=None):
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for dec, skip in zip(self.decoders, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear",
                              align_corners=False)
            x = dec(torch.cat([x, skip], dim=1))

        x = self.final_conv(x)
        x = self.collapse(x)
        x = self.head(x)
        return x if ocean is None else x * ocean
