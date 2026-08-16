"""Sliding-window Dataset over the preprocessed ERA5 cache.

A sample is a lookback window of L hourly fields used to predict one field at
lead time h:

    x        (C, L, H, W)   normalised inputs, C channels
    y        (1, H, W)      normalised target at t + h
    persist  (1, H, W)      target variable at time t, PHYSICAL units
    valid    (1, H, W)      ocean and finite -- the mask the loss respects

`persist` is carried explicitly because the persistence baseline must work even
in `paper` channel mode, where the target variable is deliberately absent from
the input channels (see CHANNEL_SETS below).

Splits are by YEAR, not random. The paper randomly splits 80/20 at the sample
level, which places hour t in train and hour t+1 in validation; with a 48-hour
lookback the two windows overlap by 47 hours and the validation score is
optimistic. Windows that straddle a split boundary are dropped here.
"""

from __future__ import annotations

import json

import numpy as np
import torch
from torch.utils.data import Dataset

from config import TEST_YEARS
from preprocess import (
    CACHE_ARRAY,
    CACHE_MASK,
    CACHE_META,
    CACHE_STATS,
    CACHE_TIMES,
    CHANNEL_INDEX,
    CHANNELS,
)

# Which variables the model is allowed to see.
#
# `paper_*` reproduces the source paper faithfully: when forecasting SWH the
# input channels are MWD, MP2, PP1D, U10, V10 -- SWH itself is NOT an input.
# That is an unusual choice (the model never sees the history of the quantity
# it predicts), and it is why the paper's 1-hour scores are not trivially
# reproducible by persistence. `full` is the conventional setup.
CHANNEL_SETS = {
    "paper_swh": ["mwd_sin", "mwd_cos", "mp2", "pp1d", "u10", "v10"],
    "paper_mp2": ["mwd_sin", "mwd_cos", "swh", "pp1d", "u10", "v10"],
    "full": list(CHANNELS),
}

# Every 5th year of the non-test record is held out for validation. Scattering
# them rather than taking a contiguous block keeps both sets representative
# across the AMO phase and the observed trend in storm frequency.
#
# With TEST_YEARS = 2021-2025 this gives train 17 / val 4 / test 5 years.
VAL_YEARS = (2004, 2009, 2014, 2019)


class ERA5Cache:
    """Thin holder for the memmap and its sidecar metadata."""

    def __init__(self, load_to_ram: bool = False):
        self.meta = json.loads(CACHE_META.read_text())
        self.shape = tuple(self.meta["shape"])
        self.stats = json.loads(CACHE_STATS.read_text())
        self.times = np.load(CACHE_TIMES)
        self.land = np.load(CACHE_MASK)
        self.ocean = ~self.land

        self.load_to_ram = load_to_ram
        self._open()

        self.years = self.times.astype("datetime64[Y]").astype(int) + 1970

    def _open(self) -> None:
        arr = np.memmap(CACHE_ARRAY, dtype=np.float32, mode="r", shape=self.shape)
        if self.load_to_ram:
            nbytes = np.prod(self.shape) * 4
            print(f"[cache] loading {nbytes / 1e9:.1f} GB into RAM")
            arr = np.asarray(arr)
        self.data = arr

    # DataLoader workers on Windows are spawned, not forked, so the cache is
    # pickled to each child. Pickling the array itself would serialise the
    # whole multi-GB record per worker (and fails outright above 2 GB), so the
    # handle is dropped and reopened on the far side. The OS page cache means
    # the workers still share one copy of the file in memory.
    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["data"] = None
        state["load_to_ram"] = False
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._open()

    def split_of(self) -> np.ndarray:
        """Per-timestep split label: 0 train, 1 val, 2 test."""
        split = np.zeros(len(self.times), dtype=np.int8)
        split[np.isin(self.years, VAL_YEARS)] = 1
        split[np.isin(self.years, TEST_YEARS)] = 2
        return split

    def denormalize(self, name: str, arr):
        s = self.stats[name]
        return arr * (s["max"] - s["min"]) + s["min"]

    def normalize(self, name: str, arr):
        s = self.stats[name]
        return (arr - s["min"]) / (s["max"] - s["min"])


class WaveWindowDataset(Dataset):
    def __init__(
        self,
        cache: ERA5Cache,
        split: str = "train",
        target: str = "swh",
        channel_set: str = "full",
        lookback: int = 48,
        lead: int = 1,
        include_mask_channel: bool = True,
        stride: int = 1,
    ):
        if target not in CHANNEL_INDEX:
            raise ValueError(f"unknown target {target!r}")
        self.cache = cache
        self.target = target
        self.lookback = lookback
        self.lead = lead
        self.include_mask_channel = include_mask_channel

        names = CHANNEL_SETS[channel_set]
        if channel_set.startswith("paper") and target in names:
            raise ValueError(
                f"channel set {channel_set!r} includes the target {target!r}; "
                "use paper_swh with target swh, or paper_mp2 with target mp2"
            )
        self.channel_names = names
        self.channel_idx = np.array([CHANNEL_INDEX[n] for n in names])
        self.target_idx = CHANNEL_INDEX[target]

        split_id = {"train": 0, "val": 1, "test": 2}[split]
        self.indices = self._valid_indices(cache, split_id, stride)

        # Precompute normalisation as (scale, offset) so __getitem__ is cheap.
        self._in_min = np.array([cache.stats[n]["min"] for n in names],
                                dtype=np.float32)[:, None, None, None]
        self._in_rng = np.array([cache.stats[n]["max"] - cache.stats[n]["min"]
                                 for n in names], dtype=np.float32)[:, None, None, None]
        ts = cache.stats[target]
        self._t_min = np.float32(ts["min"])
        self._t_rng = np.float32(ts["max"] - ts["min"])

        self._ocean = torch.from_numpy(cache.ocean.astype(np.float32))[None]

    def _valid_indices(self, cache, split_id, stride) -> np.ndarray:
        split = cache.split_of()
        n = len(split)
        hours = cache.times.astype("int64")

        anchors = np.arange(self.lookback - 1, n - self.lead, dtype=np.int64)
        lo = anchors - (self.lookback - 1)
        hi = anchors + self.lead

        # The whole span must sit in the requested split ...
        same = np.ones(len(anchors), dtype=bool)
        for offset in range(0, self.lookback):
            same &= split[anchors - offset] == split_id
        same &= split[hi] == split_id

        # ... and be strictly hourly, with no gap anywhere in the span.
        contiguous = (hours[hi] - hours[lo]) == (self.lookback - 1 + self.lead)
        keep = anchors[same & contiguous]
        return keep[::stride]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        t = int(self.indices[i])
        lo = t - self.lookback + 1

        window = self.cache.data[lo:t + 1]                  # (L, C_all, H, W)
        x = np.asarray(window[:, self.channel_idx], dtype=np.float32)
        x = np.transpose(x, (1, 0, 2, 3))                   # (C, L, H, W)
        x = (x - self._in_min) / self._in_rng
        np.nan_to_num(x, copy=False, nan=0.0)
        np.clip(x, 0.0, 1.0, out=x)

        y_phys = np.asarray(self.cache.data[t + self.lead, self.target_idx],
                            dtype=np.float32)[None]         # (1, H, W)
        persist = np.asarray(self.cache.data[t, self.target_idx],
                             dtype=np.float32)[None]

        valid = np.isfinite(y_phys) & self.cache.ocean[None]

        y = (np.nan_to_num(y_phys, nan=0.0) - self._t_min) / self._t_rng

        x = torch.from_numpy(x)
        if self.include_mask_channel:
            mask_ch = self._ocean.expand(1, self.lookback, *self._ocean.shape[1:])
            x = torch.cat([x, mask_ch], dim=0)

        return {
            "x": x,
            "y": torch.from_numpy(y),
            "y_phys": torch.from_numpy(np.nan_to_num(y_phys, nan=0.0)),
            "persist": torch.from_numpy(np.nan_to_num(persist, nan=0.0)),
            "valid": torch.from_numpy(valid),
            "t_index": t,
        }

    @property
    def n_channels(self) -> int:
        return len(self.channel_names) + (1 if self.include_mask_channel else 0)

    def describe(self) -> str:
        return (f"{len(self):,} samples | target={self.target} "
                f"| channels={self.n_channels} {self.channel_names}"
                f"{' +mask' if self.include_mask_channel else ''} "
                f"| lookback={self.lookback}h lead={self.lead}h")


def masked_mse(pred, target, valid):
    """MSE over ocean points only.

    The paper zeroes land and lets it into the loss, which spends model capacity
    on reproducing a constant and dilutes the gradient over real water. With
    ~20% of this domain being land that is not a small effect.
    """
    diff = (pred - target) ** 2
    denom = valid.sum().clamp(min=1)
    return (diff * valid).sum() / denom


def masked_l1(pred, target, valid):
    diff = (pred - target).abs()
    denom = valid.sum().clamp(min=1)
    return (diff * valid).sum() / denom
