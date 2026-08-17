"""ConvLSTM wave forecast model (the paper's Section 3.3).

Implements Shi et al. (2015) gating with 3x3 convolutions, stacked, followed by
a 1x1 deconvolution on the final hidden state as the paper describes. Included
as a comparison point: it models time recurrently rather than by 3D
convolution, which is the axis the paper argues its U-Net wins on.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch: int, hidden_ch: int, kernel: int = 3):
        super().__init__()
        self.hidden_ch = hidden_ch
        pad = kernel // 2
        # One convolution produces all four gate pre-activations at once.
        self.gates = nn.Conv2d(in_ch + hidden_ch, 4 * hidden_ch, kernel,
                               padding=pad)

    def forward(self, x, state):
        h, c = state
        combined = self.gates(torch.cat([x, h], dim=1))
        i, f, g, o = combined.chunk(4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        c = f * c + i * torch.tanh(g)
        h = o * torch.tanh(c)
        return h, c

    def init_state(self, batch, shape, device, dtype):
        zeros = torch.zeros(batch, self.hidden_ch, *shape,
                            device=device, dtype=dtype)
        return zeros, zeros.clone()


class ConvLSTM(nn.Module):
    def __init__(
        self,
        in_channels: int,
        lookback: int = 48,
        base_channels: int = 32,
        depth: int = 2,
        hidden_channels=None,
        kernel: int = 3,
        out_channels: int = 1,
    ):
        """`base_channels` and `depth` mean the same thing here as in the
        convolutional models, so one capacity knob drives every architecture
        and the comparison can be made at matched parameter counts. Each layer
        holds 2 x base_channels hidden units, which reproduces the previous
        hard-coded (64, 64) at the default base_channels=32.
        """
        super().__init__()
        if hidden_channels is None:
            hidden_channels = tuple([2 * base_channels] * depth)
        self.cells = nn.ModuleList()
        prev = in_channels
        for hidden in hidden_channels:
            self.cells.append(ConvLSTMCell(prev, hidden, kernel))
            prev = hidden
        self.head = nn.ConvTranspose2d(prev, out_channels, kernel_size=1)

    def forward(self, x, ocean=None):
        # x arrives as (B, C, T, H, W) to match the 3D models; step over T.
        b, _, t, h, w = x.shape
        states = [cell.init_state(b, (h, w), x.device, x.dtype)
                  for cell in self.cells]

        for step in range(t):
            inp = x[:, :, step]
            for li, cell in enumerate(self.cells):
                states[li] = cell(inp, states[li])
                inp = states[li][0]

        out = self.head(states[-1][0])
        return out if ocean is None else out * ocean
