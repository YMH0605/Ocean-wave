"""End-to-end pipeline test on synthetic ERA5 files.

Exercises the real preprocess -> dataset -> train -> predict path against
generated netCDF files that have the same structure, dtypes, land mask and
coordinate conventions as the CDS output, so that bugs surface while the real
download is still running rather than after it.

    python pipeline_smoke_test.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from config import LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, N_LAT, N_LON, TABLES

# Chosen so the year-based splits in dataset.py all end up non-empty:
# 2016/2017 -> train, 2018 -> val, 2019-2021 -> test.
FAKE_YEARS = [2016, 2017, 2018, 2019, 2020, 2021]
STEPS_PER_YEAR = 1200


def synth_year(year: int, rng: np.random.Generator) -> xr.Dataset:
    """A plausible-looking ERA5 year: smooth fields, land NaN in the west."""
    times = pd.date_range(f"{year}-01-01", periods=STEPS_PER_YEAR, freq="h")
    lat = np.linspace(LAT_MAX, LAT_MIN, N_LAT)      # descending, as CDS ships
    lon = np.linspace(LON_MIN, LON_MAX, N_LON)
    t = np.arange(STEPS_PER_YEAR)

    yy, xx = np.meshgrid(lat, lon, indexing="ij")
    # Land wedge along the western/north-western edge, roughly coastal.
    land = (xx < LON_MIN + 6 + 0.35 * (yy - LAT_MIN))

    # A couple of travelling storms per year plus a smooth background.
    swh = np.empty((STEPS_PER_YEAR, N_LAT, N_LON), dtype=np.float32)
    phase = rng.uniform(0, 2 * np.pi)
    for i in t:
        centre_lon = LON_MIN + 4 + (i * 0.35) % (LON_MAX - LON_MIN)
        centre_lat = 32 + 6 * np.sin(i / 190 + phase)
        dist2 = ((xx - centre_lon) / 7.0) ** 2 + ((yy - centre_lat) / 5.0) ** 2
        storm = 6.5 * np.exp(-dist2)
        background = 1.4 + 0.6 * np.sin(i / 130 + phase) + 0.02 * (xx - LON_MIN)
        swh[i] = background + storm
    swh += rng.normal(0, 0.05, swh.shape).astype(np.float32)
    swh = np.clip(swh, 0.05, None)

    mp2 = (2.6 + 1.05 * np.sqrt(swh)).astype(np.float32)
    pp1d = (mp2 * 1.35).astype(np.float32)
    mwd = ((180 + 60 * np.sin(t / 100 + phase))[:, None, None]
           * np.ones((1, N_LAT, N_LON))).astype(np.float32) % 360
    u10 = (7 * np.sin(t / 90 + phase)[:, None, None]
           + rng.normal(0, 1.5, swh.shape)).astype(np.float32)
    v10 = (5 * np.cos(t / 110 + phase)[:, None, None]
           + rng.normal(0, 1.5, swh.shape)).astype(np.float32)

    # Wave variables are undefined over land; winds are not.
    for arr in (swh, mp2, pp1d, mwd):
        arr[:, land] = np.nan

    dims = ("valid_time", "latitude", "longitude")
    coords = {"valid_time": times, "latitude": lat, "longitude": lon}
    return xr.Dataset(
        {name: (dims, arr) for name, arr in
         [("swh", swh), ("mp2", mp2), ("pp1d", pp1d), ("mwd", mwd),
          ("u10", u10), ("v10", v10)]},
        coords=coords,
    )


def main() -> int:
    import os

    src_dir = Path(__file__).parent
    sandbox = Path(tempfile.mkdtemp(prefix="wave_smoke_"))
    print(f"[smoke] sandbox: {sandbox}")

    era5_dir = sandbox / "raw" / "era5"
    processed = sandbox / "processed"
    era5_dir.mkdir(parents=True)
    processed.mkdir(parents=True)

    rng = np.random.default_rng(0)
    for year in FAKE_YEARS:
        ds = synth_year(year, rng)
        ds.to_netcdf(era5_dir / f"era5_hatteras_{year}.nc")
        ds.close()
    print(f"[smoke] wrote {len(FAKE_YEARS)} synthetic years "
          f"({STEPS_PER_YEAR} steps each)")

    # Point the cache at the sandbox before anything imports preprocess, so
    # this process and the train/predict subprocesses agree on the paths.
    os.environ["WAVE_ERA5_DIR"] = str(era5_dir)
    os.environ["WAVE_CACHE_DIR"] = str(processed)

    import preprocess

    print("\n" + "=" * 70 + "\n[smoke] STEP 1: preprocess\n" + "=" * 70)
    preprocess.build(force=True)

    print("\n" + "=" * 70 + "\n[smoke] STEP 2: dataset\n" + "=" * 70)
    import dataset as ds_mod

    cache = ds_mod.ERA5Cache()
    for split in ("train", "val", "test"):
        d = ds_mod.WaveWindowDataset(cache=cache, split=split, target="swh",
                                     channel_set="full", lookback=48, lead=1)
        print(f"  {split:5s}: {d.describe()}")
        if len(d) == 0:
            print(f"  !! {split} split is empty")
            return 1
        sample = d[0]
        print(f"         x={tuple(sample['x'].shape)} "
              f"y={tuple(sample['y'].shape)} "
              f"valid={sample['valid'].float().mean():.1%} ocean")

    d_paper = ds_mod.WaveWindowDataset(cache=cache, split="train", target="swh",
                                       channel_set="paper_swh", lookback=48,
                                       lead=1)
    print(f"  paper mode: {d_paper.describe()}")

    print("\n" + "=" * 70 + "\n[smoke] STEP 3: train (2 epochs)\n" + "=" * 70)
    env_paths = {
        "WAVE_ERA5_DIR": str(era5_dir),
        "WAVE_CACHE_DIR": str(processed),
    }
    train_cmd = [
        sys.executable, "-u", "train.py", "--model", "unet3d",
        "--target", "swh", "--lead", "1", "--epochs", "2",
        "--batch-size", "8", "--stride", "4", "--num-workers", "0",
        "--patience", "5", "--tag", "smoke",
    ]
    result = subprocess.run(train_cmd, cwd=src_dir, env=_env(env_paths),
                            text=True)
    if result.returncode != 0:
        print("[smoke] training FAILED")
        return 1

    print("\n" + "=" * 70 + "\n[smoke] STEP 4: predict\n" + "=" * 70)
    predict_cmd = [
        sys.executable, "-u", "predict.py", "--checkpoints",
        "unet3d_swh_full_lb48_h1_smoke", "--stride", "8", "--num-workers", "0",
    ]
    result = subprocess.run(predict_cmd, cwd=src_dir, env=_env(env_paths),
                            text=True)
    if result.returncode != 0:
        print("[smoke] prediction FAILED")
        return 1

    # The smoke run writes a checkpoint and a log into the real outputs tree;
    # remove them so they cannot be mistaken for a genuine experiment.
    from train import CHECKPOINTS, LOGS
    for leftover in (CHECKPOINTS / "unet3d_swh_full_lb48_h1_smoke.pt",
                     LOGS / "unet3d_swh_full_lb48_h1_smoke.json"):
        leftover.unlink(missing_ok=True)
    for leftover in TABLES.glob("test_metrics_*_h1.csv"):
        leftover.unlink(missing_ok=True)

    shutil.rmtree(sandbox, ignore_errors=True)
    print("\n[smoke] pipeline OK end to end (sandbox and artefacts removed)")
    return 0


def _env(extra: dict) -> dict:
    import os
    env = dict(os.environ)
    env.update(extra)
    return env


if __name__ == "__main__":
    raise SystemExit(main())
