"""Convert the yearly ERA5 netCDF files into one contiguous array for training.

Random access into 46 xarray-backed netCDF files is far too slow to feed a GPU,
so we flatten everything once into a single float32 memmap laid out as

    (time, channel, lat, lon)

and keep the values in PHYSICAL units. Normalisation happens on the fly in
dataset.py, which means the normalisation scheme can be changed without
re-running this step, and evaluation metrics can be computed in metres and
seconds without an inverse transform.

Three things here differ deliberately from the source paper:

  * Mean wave direction is stored as sin/cos components. The paper min-max
    normalises the raw 0-360 degree field, which puts a discontinuity at due
    north: 359 deg and 1 deg are adjacent in reality but map to 0.997 and 0.003.
  * A land mask is stored explicitly rather than relying on "land is zero and
    ReLU makes it invisible", which is not true -- zeros propagate through
    convolutions like any other value.
  * Normalisation statistics are computed on TRAIN YEARS ONLY. Computing them
    over the full record leaks test-set information into training.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import xarray as xr

from config import (
    INTERIM,
    N_LAT,
    N_LON,
    PROCESSED,
    RAW,
    TEST_YEARS,
    YEAR_END,
    YEAR_START,
)

# Both locations can be redirected, which lets an experiment (or the pipeline
# smoke test) run against an alternate cache without touching the real one.
ERA5_DIR = Path(os.environ.get("WAVE_ERA5_DIR", RAW / "era5"))
CACHE_DIR = Path(os.environ.get("WAVE_CACHE_DIR", PROCESSED))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Canonical channel order in the memmap. Everything downstream indexes by name
# through CHANNEL_INDEX, never by hard-coded integer.
CHANNELS = ["swh", "mp2", "pp1d", "mwd_sin", "mwd_cos", "u10", "v10"]
CHANNEL_INDEX = {name: i for i, name in enumerate(CHANNELS)}
N_CHANNELS = len(CHANNELS)

CACHE_ARRAY = CACHE_DIR / "era5_hatteras.f32.memmap"
CACHE_META = CACHE_DIR / "era5_hatteras_meta.json"
CACHE_TIMES = CACHE_DIR / "era5_hatteras_times.npy"
CACHE_MASK = CACHE_DIR / "land_mask.npy"
CACHE_STATS = CACHE_DIR / "norm_stats.json"


def _time_dim(ds: xr.Dataset) -> str:
    for candidate in ("valid_time", "time"):
        if candidate in ds.dims:
            return candidate
    raise KeyError(f"no recognised time dimension in {list(ds.dims)}")


def _spatial_dims(ds: xr.Dataset) -> tuple[str, str]:
    lat = "latitude" if "latitude" in ds.dims else "lat"
    lon = "longitude" if "longitude" in ds.dims else "lon"
    return lat, lon


def _year_files() -> list:
    files = sorted(ERA5_DIR.glob("era5_hatteras_[0-9][0-9][0-9][0-9].nc"))
    if not files:
        raise FileNotFoundError(
            f"no ERA5 year files in {ERA5_DIR}. Run download_era5.py first."
        )
    return files


def _load_year(path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (data[T,C,H,W] float32, times[T] datetime64, land_mask[H,W] bool)."""
    ds = xr.open_dataset(path)
    tdim = _time_dim(ds)
    latdim, londim = _spatial_dims(ds)

    # ERA5 ships latitude descending (north first). Flip to ascending so that
    # array row 0 is the southern edge, which is what the plotting code assumes.
    if float(ds[latdim][0]) > float(ds[latdim][-1]):
        ds = ds.isel({latdim: slice(None, None, -1)})

    n_time = ds.sizes[tdim]
    out = np.full((n_time, N_CHANNELS, N_LAT, N_LON), np.nan, dtype=np.float32)

    def grab(name):
        if name not in ds:
            raise KeyError(f"{name} missing from {path.name}; have {list(ds.data_vars)}")
        return ds[name].values.astype(np.float32)

    swh, mp2, pp1d = grab("swh"), grab("mp2"), grab("pp1d")
    mwd = grab("mwd")
    u10, v10 = grab("u10"), grab("v10")

    mwd_rad = np.deg2rad(mwd)
    out[:, CHANNEL_INDEX["swh"]] = swh
    out[:, CHANNEL_INDEX["mp2"]] = mp2
    out[:, CHANNEL_INDEX["pp1d"]] = pp1d
    out[:, CHANNEL_INDEX["mwd_sin"]] = np.sin(mwd_rad)
    out[:, CHANNEL_INDEX["mwd_cos"]] = np.cos(mwd_rad)
    out[:, CHANNEL_INDEX["u10"]] = u10
    out[:, CHANNEL_INDEX["v10"]] = v10

    # ERA5 wave fields are undefined over land; wind fields are not. Land is
    # therefore whatever SWH never resolves across the whole year.
    land = np.all(np.isnan(swh), axis=0)

    times = ds[tdim].values.astype("datetime64[h]")
    ds.close()
    return out, times, land


def build(force: bool = False) -> None:
    if CACHE_ARRAY.exists() and not force:
        print(f"[preprocess] cache exists at {CACHE_ARRAY}, use --force to rebuild")
        return

    files = _year_files()
    print(f"[preprocess] found {len(files)} year files")

    # First pass: count timesteps so the memmap can be sized exactly.
    counts, all_times = [], []
    for path in files:
        ds = xr.open_dataset(path)
        tdim = _time_dim(ds)
        counts.append(ds.sizes[tdim])
        all_times.append(ds[tdim].values.astype("datetime64[h]"))
        ds.close()
    total = sum(counts)
    print(f"[preprocess] {total:,} hourly timesteps -> "
          f"{total * N_CHANNELS * N_LAT * N_LON * 4 / 1e9:.1f} GB float32")

    mm = np.memmap(CACHE_ARRAY, dtype=np.float32, mode="w+",
                   shape=(total, N_CHANNELS, N_LAT, N_LON))

    land_accum = np.ones((N_LAT, N_LON), dtype=bool)
    cursor = 0
    for path, n in zip(files, counts):
        data, _, land = _load_year(path)
        mm[cursor:cursor + n] = data
        land_accum &= land
        cursor += n
        print(f"[preprocess]   {path.name}  {n:5d} steps  "
              f"({cursor:,}/{total:,})")
    mm.flush()

    times = np.concatenate(all_times)
    np.save(CACHE_TIMES, times)
    np.save(CACHE_MASK, land_accum)

    gaps = np.diff(times.astype("int64"))
    n_gaps = int((gaps != 1).sum())
    print(f"[preprocess] land fraction {land_accum.mean():.1%}, "
          f"{n_gaps} non-hourly time gaps")

    meta = {
        "shape": [total, N_CHANNELS, N_LAT, N_LON],
        "channels": CHANNELS,
        "dtype": "float32",
        "year_start": YEAR_START,
        "year_end": YEAR_END,
        "n_time_gaps": n_gaps,
        "land_fraction": float(land_accum.mean()),
    }
    CACHE_META.write_text(json.dumps(meta, indent=2))
    print(f"[preprocess] wrote {CACHE_ARRAY}")

    compute_stats(force=True)


def compute_stats(force: bool = False) -> dict:
    """Min/max and mean/std per channel, over ocean points in TRAIN years only."""
    if CACHE_STATS.exists() and not force:
        return json.loads(CACHE_STATS.read_text())

    meta = json.loads(CACHE_META.read_text())
    total = meta["shape"][0]
    mm = np.memmap(CACHE_ARRAY, dtype=np.float32, mode="r",
                   shape=tuple(meta["shape"]))
    times = np.load(CACHE_TIMES)
    land = np.load(CACHE_MASK)
    ocean = ~land

    years = times.astype("datetime64[Y]").astype(int) + 1970
    train_mask = ~np.isin(years, TEST_YEARS)
    train_idx = np.flatnonzero(train_mask)
    print(f"[stats] {len(train_idx):,} of {total:,} timesteps are non-test years")

    stats = {}
    for name, ci in CHANNEL_INDEX.items():
        # Chunked reduction: the full channel is several GB.
        lo, hi = np.inf, -np.inf
        s, ss, n = 0.0, 0.0, 0
        for start in range(0, len(train_idx), 20000):
            block = mm[train_idx[start:start + 20000], ci][:, ocean]
            block = block[np.isfinite(block)]
            if block.size == 0:
                continue
            lo = min(lo, float(block.min()))
            hi = max(hi, float(block.max()))
            s += float(block.sum())
            ss += float(np.square(block, dtype=np.float64).sum())
            n += block.size
        mean = s / n
        std = float(np.sqrt(max(ss / n - mean ** 2, 1e-12)))
        stats[name] = {"min": lo, "max": hi, "mean": mean, "std": std, "count": n}
        print(f"[stats] {name:9s} min={lo:9.3f} max={hi:9.3f} "
              f"mean={mean:8.3f} std={std:7.3f}")

    CACHE_STATS.write_text(json.dumps(stats, indent=2))
    print(f"[stats] wrote {CACHE_STATS}")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--stats-only", action="store_true")
    args = ap.parse_args()

    if args.stats_only:
        compute_stats(force=True)
    else:
        build(force=args.force)


if __name__ == "__main__":
    main()
