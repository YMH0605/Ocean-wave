"""Download and parse the NHC HURDAT2 Atlantic best-track dataset.

HURDAT2 is a flat text file alternating between a header line that identifies a
storm and a fixed number of six-hourly track lines:

    AL092021,                LARRY,     53,
    20210901, 0000,  , TD, 11.4N,  22.4W,  30, 1006, ...

Parsing yields one row per track point, tagged with the storm it belongs to.

Note on scope: we deliberately keep every status code, including EX
(extratropical). Storms that transition before reaching the mid-Atlantic still
drive large waves off Cape Hatteras, and dropping them would bias the event
catalogue toward the tropical-only regime.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from config import (
    HURDAT2_DIR_URL,
    HURDAT2_RAW,
    HURDAT2_URL,
    HATTERAS_LAT,
    HATTERAS_LON,
    LAT_MAX,
    LAT_MIN,
    LON_MAX,
    LON_MIN,
    classify_intensity,
)

EARTH_RADIUS_KM = 6371.0088


def _latest_url() -> str:
    """Scrape the NHC index for the most recent Atlantic HURDAT2 file.

    Filenames look like hurdat2-1851-<last_year>-<issue_date>.txt, and the
    issue-date suffix is sometimes 6 digits and sometimes 8, so we sort on the
    end year rather than trying to parse the date.
    """
    index = requests.get(HURDAT2_DIR_URL, timeout=60)
    index.raise_for_status()
    names = re.findall(r"hurdat2-1851-(\d{4})-(\d{6,8})\.txt", index.text)
    if not names:
        raise RuntimeError("no Atlantic HURDAT2 files found in NHC index")
    end_year, issued = max(names, key=lambda t: (int(t[0]), len(t[1]), t[1]))
    return f"{HURDAT2_DIR_URL}hurdat2-1851-{end_year}-{issued}.txt"


def download(force: bool = False) -> Path:
    """Fetch HURDAT2 from NHC unless we already have it on disk."""
    if HURDAT2_RAW.exists() and not force:
        print(f"[hurdat2] already present: {HURDAT2_RAW} "
              f"({HURDAT2_RAW.stat().st_size / 1e6:.1f} MB)")
        return HURDAT2_RAW

    for url in (HURDAT2_URL, None):
        if url is None:
            url = _latest_url()
            print(f"[hurdat2] pinned version gone, using latest: {url}")
        print(f"[hurdat2] downloading {url}")
        resp = requests.get(url, timeout=120)
        if resp.status_code == 404:
            continue
        resp.raise_for_status()
        HURDAT2_RAW.write_bytes(resp.content)
        print(f"[hurdat2] saved {len(resp.content) / 1e6:.1f} MB -> {HURDAT2_RAW}")
        return HURDAT2_RAW

    raise RuntimeError("could not download HURDAT2 from NHC")


def _parse_coord(token: str) -> float:
    """'35.2N' -> 35.2, '75.5W' -> -75.5."""
    token = token.strip()
    value = float(token[:-1])
    if token[-1] in ("S", "W"):
        value = -value
    return value


def _missing_to_nan(value: int) -> float:
    return np.nan if value <= -999 else float(value)


def parse(path=None) -> pd.DataFrame:
    """Parse HURDAT2 into a tidy per-track-point DataFrame."""
    path = path or HURDAT2_RAW
    rows = []

    with open(path, "r", encoding="utf-8") as fh:
        lines = [ln.rstrip("\n") for ln in fh if ln.strip()]

    i = 0
    while i < len(lines):
        parts = [p.strip() for p in lines[i].split(",")]
        storm_id, storm_name, n_points = parts[0], parts[1], int(parts[2])
        i += 1

        for _ in range(n_points):
            f = [p.strip() for p in lines[i].split(",")]
            i += 1
            rows.append(
                {
                    "storm_id": storm_id,
                    "storm_name": storm_name.title(),
                    "time": pd.to_datetime(f[0] + f[1], format="%Y%m%d%H%M"),
                    "record_id": f[2],
                    "status": f[3],
                    "lat": _parse_coord(f[4]),
                    "lon": _parse_coord(f[5]),
                    "wind_kt": _missing_to_nan(int(f[6])),
                    "pres_mb": _missing_to_nan(int(f[7])),
                }
            )

    df = pd.DataFrame(rows)
    df["year"] = df["time"].dt.year
    df["basin"] = df["storm_id"].str[:2]
    print(f"[hurdat2] parsed {len(df):,} track points from "
          f"{df['storm_id'].nunique():,} storms "
          f"({df['year'].min()}-{df['year'].max()})")
    return df


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. Accepts scalars or numpy arrays."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def add_geometry(df: pd.DataFrame) -> pd.DataFrame:
    """Attach distance-to-Hatteras and in-domain flags to each track point."""
    df = df.copy()
    df["dist_hatteras_km"] = haversine_km(
        df["lat"].values, df["lon"].values, HATTERAS_LAT, HATTERAS_LON
    )
    df["in_domain"] = (
        df["lat"].between(LAT_MIN, LAT_MAX) & df["lon"].between(LON_MIN, LON_MAX)
    )
    return df


def storm_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse track points to one row per storm, with relevance metrics."""
    grouped = df.groupby("storm_id")
    summary = pd.DataFrame(
        {
            "storm_name": grouped["storm_name"].first(),
            "year": grouped["year"].first(),
            "time_start": grouped["time"].min(),
            "time_end": grouped["time"].max(),
            "n_points": grouped.size(),
            "peak_wind_kt": grouped["wind_kt"].max(),
            "min_pres_mb": grouped["pres_mb"].min(),
            "min_dist_hatteras_km": grouped["dist_hatteras_km"].min(),
            "n_points_in_domain": grouped["in_domain"].sum(),
            "ever_tropical": grouped["status"].apply(
                lambda s: bool(set(s) & {"TD", "TS", "HU"})
            ),
            "ever_extratropical": grouped["status"].apply(lambda s: "EX" in set(s)),
        }
    ).reset_index()

    summary["duration_hr"] = (
        summary["time_end"] - summary["time_start"]
    ).dt.total_seconds() / 3600.0
    summary["peak_category"] = summary["peak_wind_kt"].apply(classify_intensity)
    summary["month_peak"] = summary["time_start"].dt.month
    return summary


def load(force_download: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience entry point: returns (track_points, storm_summary)."""
    download(force=force_download)
    tracks = add_geometry(parse())
    return tracks, storm_summary(tracks)


if __name__ == "__main__":
    tracks, storms = load()
    print(tracks.head())
    print(storms.head())
