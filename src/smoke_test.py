"""Shape/throughput smoke test for the model zoo, on synthetic tensors.

Runs without any ERA5 data so the architectures can be validated while the
download is still going.
"""

from __future__ import annotations

import time

import torch

from config import N_LAT, N_LON
from models import REGISTRY, build, count_parameters

LOOKBACK = 48
IN_CHANNELS = 8  # 7 fields + land mask


def bench(name: str, device: str, batch: int = 8) -> None:
    torch.manual_seed(0)
    model = build(name, in_channels=IN_CHANNELS, lookback=LOOKBACK).to(device)
    x = torch.randn(batch, IN_CHANNELS, LOOKBACK, N_LAT, N_LON, device=device)
    ocean = (torch.rand(batch, 1, N_LAT, N_LON, device=device) > 0.28).float()

    model.eval()
    with torch.no_grad():
        y = model(x, ocean)
    assert y.shape == (batch, 1, N_LAT, N_LON), f"{name}: got {tuple(y.shape)}"

    # One training step, to catch backward-pass problems and measure memory.
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    t0 = time.time()
    steps = 3
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = ((model(x, ocean) - torch.randn_like(y)) ** 2).mean()
        loss.backward()
        opt.step()
    if device == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) / steps

    peak = (torch.cuda.max_memory_allocated() / 1e9) if device == "cuda" else 0.0
    print(f"  {name:10s} out={tuple(y.shape)}  params={count_parameters(model):>10,}"
          f"  {dt * 1000:7.0f} ms/step  peak={peak:5.2f} GB")


def main() -> None:
    print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"device: {torch.cuda.get_device_name(0)}  "
              f"capability sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}")

    print(f"\ninput: (B, {IN_CHANNELS}, {LOOKBACK}, {N_LAT}, {N_LON})   "
          f"batch=8\n")
    for name in REGISTRY:
        bench(name, device)

    print("\nall models produce the expected (B, 1, H, W) forecast frame")


if __name__ == "__main__":
    main()
