"""NDBC buoy observations: download, parse, and match to the model grid.

Everything up to this point has compared model output against ERA5, which is
itself a wave model constrained by satellites. Those numbers say how well we
reproduce ERA5, not how well we reproduce the ocean. Buoys are the closest
thing to a direct measurement available here, and -- unlike the altimeters ERA5
assimilates -- they are largely independent of it.

Buoys are used for validation only. They never enter training: 13 point
locations against 1,536 grid cells is a vanishing fraction of the signal, and
using them would forfeit the one independent reference we have.

One caveat matters for reading the results. A buoy measures a point; an ERA5
cell is an average over roughly 56 x 46 km. In a storm the point can genuinely
exceed the cell mean, so a positive buoy-minus-ERA5 difference is not by itself
evidence of a model error. This is the representativeness problem, and it is
why the comparison is reported per station rather than pooled.

    python buoys.py --download        # fetch the raw records
    python buoys.py                   # download if needed, then match
"""

from __future__ import annotations

import argparse
import gzip
import io

import numpy as np
import pandas as pd
import requests

from config import GRID_RES, INTERIM, LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, RAW, TEST_YEARS

BUOY_DIR = RAW / "ndbc"
BUOY_DIR.mkdir(parents=True, exist_ok=True)
MATCHED = INTERIM / "buoy_matched.parquet"

ARCHIVE = "https://www.ndbc.noaa.gov/data/historical/stdmet/{s}h{y}.txt.gz"

# Stations inside the model domain with data across the test years. Distance to
# the coast matters for interpretation, so it is recorded here.
STATIONS = {
    "41025": ("Diamond Shoals, NC", 35.01, -75.40, "shelf"),
    "44095": ("Oregon Inlet, NC", 35.75, -75.33, "shelf"),
    "44100": ("Duck FRF 26m, NC", 36.26, -75.59, "nearshore"),
    "44056": ("Duck FRF 17m, NC", 36.20, -75.71, "nearshore"),
    "41013": ("Frying Pan Shoals, NC", 33.44, -77.74, "shelf"),
    "44014": ("Virginia Beach, VA", 36.61, -74.84, "shelf"),
    "44009": ("Delaware Bay", 38.46, -74.70, "shelf"),
    "41001": ("E of Cape Hatteras", 34.72, -72.32, "offshore"),
    "41002": ("S Hatteras", 31.76, -74.84, "offshore"),
    "41004": ("Edisto, SC", 32.50, -79.10, "shelf"),
    "41047": ("NE Bahamas", 27.47, -71.49, "offshore"),
    "41048": ("W Bermuda", 31.83, -69.65, "offshore"),
    "44008": ("Nantucket SE", 40.50, -69.25, "offshore"),
}

# NDBC uses these as fill values, per column.
FILL = {"WVHT": 99.0, "APD": 99.0, "DPD": 99.0, "MWD": 999.0,
        "WSPD": 99.0, "WDIR": 999.0}


def download(force: bool = False) -> None:
    for station in STATIONS:
        for year in TEST_YEARS:
            dest = BUOY_DIR / f"{station}_{year}.txt"
            if dest.exists() and not force:
                continue
            url = ARCHIVE.format(s=station, y=year)
            try:
                r = requests.get(url, timeout=90)
                if r.status_code != 200:
                    print(f"  {station} {year}: HTTP {r.status_code}")
                    continue
                text = gzip.decompress(r.content).decode("utf-8", "replace")
                dest.write_text(text, encoding="utf-8")
                print(f"  {station} {year}: {len(text) / 1e6:.1f} MB")
            except Exception as exc:
                print(f"  {station} {year}: {exc}")


def read_station_year(path) -> pd.DataFrame | None:
    """Parse one NDBC standard-meteorological file."""
    with open(path, "r", encoding="utf-8") as fh:
        header = fh.readline().lstrip("#").split()
        second = fh.readline()
        # Modern files carry a units line starting with '#'; older ones do not.
        skip = 2 if second.startswith("#") else 1

    df = pd.read_csv(path, sep=r"\s+", names=header, skiprows=skip,
                     na_values=["MM"], engine="python")
    if "WVHT" not in df.columns:
        return None

    ycol = "YY" if "YY" in df.columns else "#YY"
    df["time"] = pd.to_datetime(
        dict(year=df[ycol], month=df["MM"], day=df["DD"],
             hour=df["hh"], minute=df.get("mm", 0)), errors="coerce")

    keep = ["time"] + [c for c in ("WVHT", "APD", "DPD", "MWD", "WSPD", "WDIR")
                       if c in df.columns]
    df = df[keep].dropna(subset=["time"])

    for col, fill in FILL.items():
        if col in df.columns:
            df.loc[df[col] >= fill, col] = np.nan

    # NDBC reports every 10-30 min; ERA5 is hourly, so average to the hour.
    df["time"] = df["time"].dt.floor("h")
    return df.groupby("time", as_index=False).mean(numeric_only=True)


def load_observations() -> pd.DataFrame:
    frames = []
    for station in STATIONS:
        for year in TEST_YEARS:
            path = BUOY_DIR / f"{station}_{year}.txt"
            if not path.exists():
                continue
            try:
                df = read_station_year(path)
            except Exception as exc:
                print(f"  parse failed {path.name}: {exc}")
                continue
            if df is None or df.empty:
                continue
            df["station"] = station
            frames.append(df)
    if not frames:
        raise SystemExit("no buoy files; run with --download first")
    obs = pd.concat(frames, ignore_index=True)
    obs = obs.dropna(subset=["WVHT"])
    obs = obs[(obs["WVHT"] > 0) & (obs["WVHT"] < 25)]
    return obs


def grid_cell(lat: float, lon: float) -> tuple[int, int]:
    return (int(round((lat - LAT_MIN) / GRID_RES)),
            int(round((lon - LON_MIN) / GRID_RES)))


def match_to_era5(obs: pd.DataFrame) -> pd.DataFrame:
    """Attach the ERA5 SWH of the containing grid cell to every observation."""
    from dataset import ERA5Cache
    from preprocess import CHANNEL_INDEX

    cache = ERA5Cache()
    times = pd.DatetimeIndex(cache.times.astype("datetime64[ns]"))
    lookup = pd.Series(np.arange(len(times)), index=times)
    ci = CHANNEL_INDEX["swh"]

    rows = []
    for station, grp in obs.groupby("station"):
        name, lat, lon, setting = STATIONS[station]
        i, j = grid_cell(lat, lon)
        if not (0 <= i < cache.ocean.shape[0] and 0 <= j < cache.ocean.shape[1]):
            print(f"  {station}: outside the domain, skipped")
            continue
        if not cache.ocean[i, j]:
            print(f"  {station}: nearest cell is land, skipped")
            continue

        idx = lookup.reindex(grp["time"]).to_numpy()
        ok = ~np.isnan(idx)
        if ok.sum() == 0:
            continue
        idx_ok = idx[ok].astype(int)
        series = np.asarray(cache.data[idx_ok, ci, i, j], dtype=np.float32)

        sub = grp.loc[ok].copy()
        sub["era5_swh"] = series
        sub["t_index"] = idx_ok
        sub["lat"], sub["lon"] = lat, lon
        sub["cell_i"], sub["cell_j"] = i, j
        sub["name"], sub["setting"] = name, setting
        rows.append(sub)
        print(f"  {station} {name:<24} {len(sub):6,} hours matched  "
              f"(cell {i},{j})")

    matched = pd.concat(rows, ignore_index=True).dropna(subset=["era5_swh"])
    matched.to_parquet(MATCHED, index=False)
    return matched


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.download or not any(BUOY_DIR.glob("*.txt")):
        print(f"[buoys] downloading {len(STATIONS)} stations x "
              f"{len(TEST_YEARS)} years")
        download(force=args.force)

    print("\n[buoys] parsing")
    obs = load_observations()
    print(f"  {len(obs):,} hourly observations from "
          f"{obs['station'].nunique()} stations")

    print("\n[buoys] matching to ERA5 grid cells")
    matched = match_to_era5(obs)

    print(f"\n[buoys] {len(matched):,} paired hours -> {MATCHED}")
    print(f"  buoy SWH  mean {matched['WVHT'].mean():.2f} m, "
          f"max {matched['WVHT'].max():.2f} m")
    print(f"  ERA5 SWH  mean {matched['era5_swh'].mean():.2f} m, "
          f"max {matched['era5_swh'].max():.2f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
