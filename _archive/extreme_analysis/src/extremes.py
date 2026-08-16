"""Wave-based extreme event catalogue for the Cape Hatteras domain.

The source paper defines "extreme sea conditions" as "the time window of a
tropical cyclone", taken from a storm track database. The HURDAT2 EDA showed
why that definition fails here: it covers 3% of the record, contains only five
major hurricanes in 46 years, and has no events at all between December and
April, which is exactly when Outer Banks nor'easters do their damage.

So events are detected from the SWH field itself:

    domain statistic  ->  threshold at a high percentile of TRAIN years
                      ->  runs above threshold lasting >= MIN_DURATION
                      ->  nearby runs merged
                      ->  each event labelled TC-origin or ET-origin by
                          checking HURDAT2 for a track point in the window

The catalogue is used for stratified evaluation and for optional loss
weighting. It is never used to filter the training record.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import hurdat2
from config import (
    FIGURES,
    LAT_MAX,
    LAT_MIN,
    LON_MAX,
    LON_MIN,
    TABLES,
    TEST_YEARS,
)
from dataset import ERA5Cache
from preprocess import CHANNEL_INDEX

# An event must hold above threshold for at least this long to count; two runs
# separated by less than MERGE_GAP are treated as one storm.
MIN_DURATION_HR = 12
MERGE_GAP_HR = 12
THRESHOLD_PERCENTILE = 99.0

# Criteria for calling an event tropical in origin. All three must hold for the
# same HURDAT2 track point.
#
# Requiring proximity to the event PEAK rather than merely to the event window
# matters: a weak, unrelated depression drifting past days before the waves
# build would otherwise mislabel an extratropical storm as tropical. The
# November 1980 event -- 13.0 m off Hatteras, with only a 25 kt depression near
# Florida five days earlier and Hurricane Karl 1500 km to the east two days
# later -- is the case that motivated tightening this.
TC_MATCH_PAD_DEG = 5.0        # how far outside the domain a track may sit
TC_MATCH_PAD_HR = 24          # tolerance around the event window
TC_MATCH_PEAK_HR = 72         # ... and around the peak itself
TC_MATCH_MIN_WIND_KT = 34     # at least tropical-storm strength


def domain_series(cache: ERA5Cache, stat: str = "p95") -> np.ndarray:
    """Collapse the SWH field to one number per hour.

    The domain maximum is too spiky -- a single grid cell at the open-ocean edge
    can trip it -- so the default is the 95th percentile over ocean points,
    which responds to a storm affecting a meaningful area.
    """
    ci = CHANNEL_INDEX["swh"]
    ocean = cache.ocean
    n = cache.shape[0]
    out = np.empty(n, dtype=np.float32)

    for start in range(0, n, 20000):
        block = cache.data[start:start + 20000, ci][:, ocean]
        if stat == "max":
            out[start:start + len(block)] = np.nanmax(block, axis=1)
        elif stat == "mean":
            out[start:start + len(block)] = np.nanmean(block, axis=1)
        else:
            out[start:start + len(block)] = np.nanpercentile(block, 95, axis=1)
    return out


def detect_events(series: np.ndarray, times: np.ndarray, years: np.ndarray,
                  percentile: float = THRESHOLD_PERCENTILE) -> pd.DataFrame:
    """Threshold-exceedance events, with the threshold set on train years."""
    train = ~np.isin(years, TEST_YEARS)
    threshold = float(np.nanpercentile(series[train], percentile))
    print(f"[extremes] P{percentile:g} threshold on train years: "
          f"{threshold:.2f} m")

    above = series >= threshold
    edges = np.diff(above.astype(np.int8))
    starts = list(np.flatnonzero(edges == 1) + 1)
    ends = list(np.flatnonzero(edges == -1) + 1)
    if above[0]:
        starts.insert(0, 0)
    if above[-1]:
        ends.append(len(above))

    runs = [[s, e] for s, e in zip(starts, ends)]

    merged = []
    for run in runs:
        if merged and run[0] - merged[-1][1] <= MERGE_GAP_HR:
            merged[-1][1] = run[1]
        else:
            merged.append(run)

    rows = []
    for s, e in merged:
        if e - s < MIN_DURATION_HR:
            continue
        peak = int(s + np.argmax(series[s:e]))
        rows.append({
            "start_index": s,
            "end_index": e,
            "time_start": pd.Timestamp(times[s]),
            "time_end": pd.Timestamp(times[e - 1]),
            "time_peak": pd.Timestamp(times[peak]),
            "duration_hr": int(e - s),
            "peak_swh_m": float(series[peak]),
            "mean_swh_m": float(series[s:e].mean()),
            "year": int(years[peak]),
            "month": pd.Timestamp(times[peak]).month,
        })

    df = pd.DataFrame(rows)
    df.attrs["threshold"] = threshold
    return df


def label_origin(events: pd.DataFrame, tracks: pd.DataFrame) -> pd.DataFrame:
    """Tag each event TC or ET from the HURDAT2 catalogue.

    An event is tropical only if a single track point satisfies all of:
      * inside the domain expanded by TC_MATCH_PAD_DEG,
      * within TC_MATCH_PAD_HR of the event window AND TC_MATCH_PEAK_HR of the
        wave peak,
      * at least TC_MATCH_MIN_WIND_KT, so passing depressions do not count.

    The matched storm's name, time offset and distance are recorded so every
    label can be audited rather than taken on trust.
    """
    near = tracks[
        tracks["lat"].between(LAT_MIN - TC_MATCH_PAD_DEG, LAT_MAX + TC_MATCH_PAD_DEG)
        & tracks["lon"].between(LON_MIN - TC_MATCH_PAD_DEG, LON_MAX + TC_MATCH_PAD_DEG)
        & (tracks["wind_kt"] >= TC_MATCH_MIN_WIND_KT)
    ]
    pad = pd.Timedelta(hours=TC_MATCH_PAD_HR)
    peak_pad = pd.Timedelta(hours=TC_MATCH_PEAK_HR)

    records = []
    for _, ev in events.iterrows():
        dt_peak = (near["time"] - ev["time_peak"]).abs()
        candidates = near[
            (near["time"] >= ev["time_start"] - pad)
            & (near["time"] <= ev["time_end"] + pad)
            & (dt_peak <= peak_pad)
        ]
        if candidates.empty:
            records.append({"origin": "ET", "hurdat_name": "",
                            "hurdat_status": "", "hurdat_wind_kt": np.nan,
                            "hurdat_dt_hr": np.nan,
                            "hurdat_dist_km": np.nan})
            continue

        best = candidates.loc[
            (candidates["time"] - ev["time_peak"]).abs().idxmin()]
        records.append({
            "origin": "TC",
            "hurdat_name": best["storm_name"],
            "hurdat_status": best["status"],
            "hurdat_wind_kt": float(best["wind_kt"]),
            "hurdat_dt_hr": float(
                (best["time"] - ev["time_peak"]).total_seconds() / 3600),
            "hurdat_dist_km": float(best["dist_hatteras_km"]),
        })

    labelled = pd.concat([events.reset_index(drop=True),
                          pd.DataFrame(records)], axis=1)
    labelled.attrs.update(events.attrs)  # concat does not carry attrs over
    return labelled


def make_figure(events: pd.DataFrame, series: np.ndarray,
                times: np.ndarray, threshold: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    colors = {"TC": "#d73027", "ET": "#4292c6"}

    months = np.arange(1, 13)
    tc = [((events["origin"] == "TC") & (events["month"] == m)).sum()
          for m in months]
    et = [((events["origin"] == "ET") & (events["month"] == m)).sum()
          for m in months]
    axes[0].bar(months, tc, color=colors["TC"], label="TC origin")
    axes[0].bar(months, et, bottom=tc, color=colors["ET"], label="ET origin")
    axes[0].set_xticks(months)
    axes[0].set_title("Extreme wave events by month")
    axes[0].set_xlabel("month"); axes[0].set_ylabel("events"); axes[0].legend()

    for origin, grp in events.groupby("origin"):
        axes[1].scatter(grp["duration_hr"], grp["peak_swh_m"], s=18, alpha=0.6,
                        color=colors[origin], label=f"{origin} (n={len(grp)})")
    axes[1].axhline(threshold, color="k", ls="--", lw=1, label="threshold")
    axes[1].set_xlabel("duration (h)"); axes[1].set_ylabel("peak SWH (m)")
    axes[1].set_title("Event intensity vs duration"); axes[1].legend(fontsize=8)

    per_year = events.groupby(["year", "origin"]).size().unstack(fill_value=0)
    per_year.plot(kind="bar", stacked=True, ax=axes[2],
                  color=[colors.get(c, "#999") for c in per_year.columns],
                  width=0.85, legend=True)
    axes[2].set_title("Events per year")
    axes[2].set_xlabel("year"); axes[2].set_ylabel("events")
    for i, label in enumerate(axes[2].get_xticklabels()):
        label.set_visible(i % 5 == 0)

    fig.tight_layout()
    out = FIGURES / "03_extreme_events.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[extremes] figure -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--percentile", type=float, default=THRESHOLD_PERCENTILE)
    ap.add_argument("--stat", default="p95", choices=["p95", "max", "mean"])
    args = ap.parse_args()

    cache = ERA5Cache(load_to_ram=False)
    print(f"[extremes] {cache.shape[0]:,} hourly steps")

    series = domain_series(cache, stat=args.stat)
    events = detect_events(series, cache.times, cache.years, args.percentile)
    threshold = events.attrs["threshold"]

    tracks, _ = hurdat2.load()
    events = label_origin(events, tracks)

    n_tc = (events["origin"] == "TC").sum()
    n_et = (events["origin"] == "ET").sum()
    covered = int(events["duration_hr"].sum())

    print("\n" + "=" * 72)
    print(f"Wave-based extreme event catalogue "
          f"(SWH {args.stat} >= P{args.percentile:g} = {threshold:.2f} m, "
          f">= {MIN_DURATION_HR} h)")
    print("=" * 72)
    print(f"  events            : {len(events)}  "
          f"({len(events) / len(np.unique(cache.years)):.1f}/yr)")
    print(f"  tropical origin   : {n_tc:4d}  ({n_tc / len(events):5.1%})")
    print(f"  extratropical     : {n_et:4d}  ({n_et / len(events):5.1%})")
    print(f"  hours covered     : {covered:,} "
          f"({covered / cache.shape[0]:.1%} of record)")
    print(f"  peak SWH          : {events['peak_swh_m'].max():.2f} m "
          f"({events.loc[events['peak_swh_m'].idxmax(), 'time_peak']})")

    print("\n  Monthly split (TC / ET):")
    names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
             "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for m in range(1, 13):
        tc = ((events["origin"] == "TC") & (events["month"] == m)).sum()
        et = ((events["origin"] == "ET") & (events["month"] == m)).sum()
        print(f"    {names[m - 1]}  {tc:3d} / {et:3d}   "
              + "#" * tc + "-" * et)
    print("=" * 72 + "\n")

    out = TABLES / "extreme_events.csv"
    events.to_csv(out, index=False)
    print(f"[extremes] catalogue -> {out}")

    make_figure(events, series, cache.times, threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
