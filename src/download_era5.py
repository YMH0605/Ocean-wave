"""Download ERA5 wave + wind fields over the Cape Hatteras domain.

Deliberately downloads the CONTINUOUS record, not storm windows. The HURDAT2
EDA showed that storm-window subsetting keeps only ~3% of the hourly record and
contains zero events in Dec-Apr, so the storm catalogue is used downstream for
stratified evaluation and loss weighting, never as a data filter.

Two CDS quirks drive the design here:

  * Cost limit. A request is priced by size against a ceiling of 121,000.
    Six variables over the 32x48 grid costs 53,568 per month, so a two-month
    chunk (101,952) is the largest that passes. A whole year is 630,720 and is
    rejected outright with "cost limits exceeded".
  * Split streams. Wave parameters come from the `wave` stream and 10 m winds
    from `oper`, so any request mixing them is delivered as a zip of two netCDF
    files no matter what `download_format` says. They are merged on arrival.

So: 6 chunks per year x 46 years = 276 requests, fetched by a small thread pool
and then assembled into one file per year.

Usage
-----
    python download_era5.py --test              # one month, verifies the grid
    python download_era5.py                     # full 1979-2024
    python download_era5.py --years 2019 2024 --workers 4
    python download_era5.py --status            # what is already on disk
"""

from __future__ import annotations

import argparse
import random
import sys
import tempfile
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from config import (
    ERA5_WAVE_VARS,
    ERA5_WIND_VARS,
    GRID_RES,
    LAT_MAX,
    LAT_MIN,
    LON_MAX,
    LON_MIN,
    N_LAT,
    N_LON,
    RAW,
    YEAR_END,
    YEAR_START,
)

DATASET = "reanalysis-era5-single-levels"
ERA5_DIR = RAW / "era5"
CHUNK_DIR = ERA5_DIR / "chunks"
ERA5_DIR.mkdir(parents=True, exist_ok=True)
CHUNK_DIR.mkdir(parents=True, exist_ok=True)

ALL_VARS = list(ERA5_WAVE_VARS.values()) + list(ERA5_WIND_VARS.values())

# CDS area order is [North, West, South, East].
AREA = [LAT_MAX, LON_MIN, LAT_MIN, LON_MAX]

# Two months per request: the largest chunk under the 121,000 cost ceiling.
CHUNKS = [("01", "02"), ("03", "04"), ("05", "06"),
          ("07", "08"), ("09", "10"), ("11", "12")]

DAYS = [f"{d:02d}" for d in range(1, 32)]
HOURS = [f"{h:02d}:00" for h in range(24)]

_local = threading.local()
_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def _client():
    """cdsapi.Client is not thread-safe, so give each worker its own."""
    if not hasattr(_local, "client"):
        import cdsapi
        _local.client = cdsapi.Client(quiet=True, progress=False)
    return _local.client


def build_request(year: int, months) -> dict:
    return {
        "product_type": ["reanalysis"],
        "variable": ALL_VARS,
        "year": [str(year)],
        "month": list(months),
        "day": DAYS,
        "time": HOURS,
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": AREA,
        "grid": [GRID_RES, GRID_RES],
    }


def _consolidate(archive: Path, target: Path) -> Path:
    """Merge CDS output into a single netCDF, unzipping the split streams."""
    import xarray as xr

    if not zipfile.is_zipfile(archive):
        archive.replace(target)
        return target

    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(archive) as z:
            z.extractall(tmp)
        parts = sorted(Path(tmp).glob("*.nc"))
        if not parts:
            raise RuntimeError(f"no .nc inside {archive.name}")

        datasets = []
        for part in parts:
            ds = xr.open_dataset(part)
            # expver is bookkeeping that differs between ERA5 and ERA5T and
            # blocks a clean merge.
            datasets.append(ds.drop_vars("expver", errors="ignore").load())
            ds.close()

        merged = xr.merge(datasets, compat="override", join="exact")
        encoding = {v: {"zlib": True, "complevel": 4} for v in merged.data_vars}
        tmp_out = target.with_suffix(".tmp.nc")
        merged.to_netcdf(tmp_out, encoding=encoding)
        merged.close()

    archive.unlink(missing_ok=True)
    tmp_out.replace(target)
    return target


def chunk_path(year: int, ci: int) -> Path:
    return CHUNK_DIR / f"era5_{year}_c{ci}.nc"


def year_path(year: int) -> Path:
    return ERA5_DIR / f"era5_hatteras_{year}.nc"


# CDS caps how many jobs one user may have queued against a dataset. Hitting
# it is not an error, just backpressure, so it gets its own generous budget.
THROTTLE_MARKERS = (
    "temporarily limited",
    "has been rejected",
    "too many requests",
    "rate limit",
)
MAX_THROTTLE_WAITS = 40
THROTTLE_SLEEP = 90


def _is_throttle(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in THROTTLE_MARKERS)


def fetch_chunk(year: int, ci: int, retries: int = 3) -> Path:
    target = chunk_path(year, ci)
    if target.exists() and target.stat().st_size > 0:
        return target

    months = CHUNKS[ci]
    staging = CHUNK_DIR / f"_dl_{year}_c{ci}.bin"
    label = f"{year} {months[0]}-{months[1]}"

    attempt, throttled = 0, 0
    while True:
        try:
            t0 = time.time()
            _client().retrieve(DATASET, build_request(year, months), str(staging))
            _consolidate(staging, target)
            log(f"  [ok]   {label}  {target.stat().st_size / 1e6:5.1f} MB  "
                f"{(time.time() - t0) / 60:4.1f} min")
            return target
        except Exception as exc:
            staging.unlink(missing_ok=True)

            if _is_throttle(exc):
                throttled += 1
                if throttled > MAX_THROTTLE_WAITS:
                    log(f"  [FAIL] {label}: still throttled after "
                        f"{MAX_THROTTLE_WAITS} waits")
                    raise
                # Jitter so parallel workers do not resubmit in lockstep.
                wait = THROTTLE_SLEEP + random.uniform(0, 60)
                if throttled == 1 or throttled % 10 == 0:
                    log(f"  [queue full] {label}: waiting "
                        f"(attempt {throttled}/{MAX_THROTTLE_WAITS})")
                time.sleep(wait)
                continue

            attempt += 1
            if attempt >= retries:
                log(f"  [FAIL] {label}: {exc}")
                raise
            log(f"  [retry {attempt}/{retries}] {label}: {exc}")
            time.sleep(60 * attempt)


def assemble_year(year: int) -> Path | None:
    """Concatenate the six chunks of a year into one file."""
    import xarray as xr

    target = year_path(year)
    if target.exists() and target.stat().st_size > 0:
        return target

    parts = [chunk_path(year, ci) for ci in range(len(CHUNKS))]
    if not all(p.exists() for p in parts):
        return None

    datasets = [xr.open_dataset(p).load() for p in parts]
    tdim = "valid_time" if "valid_time" in datasets[0].dims else "time"
    combined = xr.concat(datasets, dim=tdim).sortby(tdim)
    for ds in datasets:
        ds.close()

    encoding = {v: {"zlib": True, "complevel": 4} for v in combined.data_vars}
    tmp_out = target.with_suffix(".tmp.nc")
    combined.to_netcdf(tmp_out, encoding=encoding)
    n_steps = combined.sizes[tdim]
    combined.close()
    tmp_out.replace(target)

    log(f"[assemble] {year}: {n_steps:5d} steps -> "
        f"{target.stat().st_size / 1e6:.0f} MB")
    return target


def verify(path: Path) -> None:
    """Confirm the delivered grid matches what config.py promises."""
    import xarray as xr

    ds = xr.open_dataset(path)
    lat = "latitude" if "latitude" in ds.coords else "lat"
    lon = "longitude" if "longitude" in ds.coords else "lon"
    tdim = "valid_time" if "valid_time" in ds.dims else "time"
    n_lat, n_lon = ds.sizes[lat], ds.sizes[lon]

    print(f"\n[verify] {path.name}")
    print(f"  variables : {list(ds.data_vars)}")
    print(f"  grid      : {n_lat} lat x {n_lon} lon  (expected {N_LAT} x {N_LON})")
    print(f"  lat range : {float(ds[lat].min())} .. {float(ds[lat].max())}")
    print(f"  lon range : {float(ds[lon].min())} .. {float(ds[lon].max())}")
    print(f"  timesteps : {ds.sizes[tdim]}")
    if "swh" in ds:
        land = float(ds["swh"].isel({tdim: 0}).isnull().mean())
        print(f"  land/NaN fraction in SWH: {land:.1%}")
    if (n_lat, n_lon) != (N_LAT, N_LON):
        print("  !! grid mismatch - fix config.py before bulk downloading")
    ds.close()


def status(years) -> None:
    """Report progress in chunks, since assembly only happens at the very end.

    A year whose six chunks are all present is finished as far as downloading
    goes; reporting it as incomplete just because assemble_year has not run yet
    makes a nearly-done job look like it has not started.
    """
    assembled, chunks_done, partial, total_chunks = [], 0, [], 0
    for year in years:
        total_chunks += len(CHUNKS)
        if year_path(year).exists():
            assembled.append(year)
            chunks_done += len(CHUNKS)
            continue
        n = sum(chunk_path(year, ci).exists() for ci in range(len(CHUNKS)))
        chunks_done += n
        if n == len(CHUNKS):
            partial.append((year, "downloaded"))
        elif n:
            partial.append((year, f"{n}/{len(CHUNKS)}"))

    total_mb = sum(p.stat().st_size for p in ERA5_DIR.rglob("*.nc")) / 1e6
    pct = 100 * chunks_done / total_chunks if total_chunks else 0
    print(f"[status] chunks         : {chunks_done}/{total_chunks} ({pct:.0f}%)")
    print(f"[status] years assembled: {len(assembled)}/{len(years)}"
          f"  (assembly runs after all downloads finish)")
    in_progress = [f"{y}({s})" for y, s in partial if s != "downloaded"]
    ready = [str(y) for y, s in partial if s == "downloaded"]
    if ready:
        print(f"[status] downloaded, awaiting assembly: {len(ready)} years "
              f"({ready[0]}-{ready[-1]})")
    if in_progress:
        print(f"[status] in progress    : {', '.join(in_progress)}")
    print(f"[status] on disk        : {total_mb / 1000:.2f} GB")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true",
                    help="fetch one two-month chunk and verify the grid")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--years", nargs=2, type=int, metavar=("START", "END"),
                    default=[YEAR_START, YEAR_END])
    ap.add_argument("--workers", type=int, default=2,
                    help="concurrent CDS requests; CDS caps queued jobs per "
                         "dataset, and more than 2 just triggers throttling")
    args = ap.parse_args()

    years = list(range(args.years[0], args.years[1] + 1))

    if args.status:
        status(years)
        return 0

    try:
        import cdsapi  # noqa: F401
    except ImportError:
        print("cdsapi not installed:  pip install 'cdsapi>=0.7.2'", file=sys.stderr)
        return 1

    if args.test:
        path = fetch_chunk(2020, 4)  # Sep-Oct 2020
        verify(path)
        return 0

    jobs = [(y, ci) for y in years for ci in range(len(CHUNKS))
            if not year_path(y).exists() and not chunk_path(y, ci).exists()]
    print(f"[era5] {len(jobs)} chunks to fetch across {len(years)} years "
          f"({args.workers} workers)")

    t0 = time.time()
    failures = []
    if jobs:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(fetch_chunk, y, ci): (y, ci) for y, ci in jobs}
            for i, fut in enumerate(as_completed(futures), 1):
                y, ci = futures[fut]
                try:
                    fut.result()
                except Exception:
                    failures.append((y, ci))
                if i % 10 == 0 or i == len(jobs):
                    log(f"[era5] {i}/{len(jobs)} chunks  "
                        f"({(time.time() - t0) / 60:.0f} min elapsed)")

    print("\n[era5] assembling years ...")
    assembled = [y for y in years if assemble_year(y) is not None]

    print(f"\n[era5] {len(assembled)}/{len(years)} years complete in "
          f"{(time.time() - t0) / 60:.0f} min")
    if failures:
        print(f"[era5] {len(failures)} chunks failed: {failures}")
        print("[era5] rerun the same command to retry them")
    if assembled:
        verify(year_path(assembled[0]))
    status(years)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
