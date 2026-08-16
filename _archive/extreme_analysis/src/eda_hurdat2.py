"""Exploratory analysis of the HURDAT2 storm catalogue over the Hatteras domain.

Purpose is to answer, before any ERA5 is downloaded:

  1. How many Atlantic storms are actually relevant to Cape Hatteras, 1979-2024?
  2. What fraction of the hourly record do those storms cover? (i.e. how much
     data would the paper's event-subsetting strategy have thrown away)
  3. What is the tropical vs. extratropical split at closest approach? This
     decides whether a separate nor'easter catalogue is needed.

Outputs land in outputs/tables and outputs/figures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import hurdat2
from config import (
    FIGURES,
    HATTERAS_LAT,
    HATTERAS_LON,
    INTERIM,
    LAT_MAX,
    LAT_MIN,
    LOCAL_RADIUS_KM,
    LON_MAX,
    LON_MIN,
    TABLES,
    YEAR_END,
    YEAR_START,
    classify_intensity,
)

# Track points are 6-hourly; a storm is treated as "active" for this many hours
# on either side of its qualifying track points when accumulating coverage.
EVENT_PAD_HR = 24


def qualifying_storms(tracks: pd.DataFrame) -> pd.DataFrame:
    """One row per (storm, criterion) with the timing of its relevant window."""
    era = tracks[tracks["year"].between(YEAR_START, YEAR_END)].copy()
    era["near_hatteras"] = era["dist_hatteras_km"] <= LOCAL_RADIUS_KM

    records = []
    for criterion, mask_col in (("near_hatteras", "near_hatteras"),
                                ("in_domain", "in_domain")):
        hits = era[era[mask_col]]
        for storm_id, grp in hits.groupby("storm_id"):
            full = era[era["storm_id"] == storm_id]
            closest = full.loc[full["dist_hatteras_km"].idxmin()]
            records.append(
                {
                    "criterion": criterion,
                    "storm_id": storm_id,
                    "storm_name": grp["storm_name"].iloc[0],
                    "year": grp["year"].iloc[0],
                    "window_start": grp["time"].min(),
                    "window_end": grp["time"].max(),
                    "n_qualifying_points": len(grp),
                    "min_dist_km": full["dist_hatteras_km"].min(),
                    "peak_wind_kt_overall": full["wind_kt"].max(),
                    "wind_kt_at_closest": closest["wind_kt"],
                    "status_at_closest": closest["status"],
                    "month_at_closest": closest["time"].month,
                }
            )

    out = pd.DataFrame(records)
    out["window_hr"] = (
        out["window_end"] - out["window_start"]
    ).dt.total_seconds() / 3600.0 + 2 * EVENT_PAD_HR
    out["category_at_closest"] = out["wind_kt_at_closest"].apply(classify_intensity)
    out["category_overall"] = out["peak_wind_kt_overall"].apply(classify_intensity)
    return out


def coverage_hours(events: pd.DataFrame) -> dict:
    """Union of all padded event windows, in hours, versus the full record."""
    hourly = pd.date_range(
        f"{YEAR_START}-01-01", f"{YEAR_END}-12-31 23:00", freq="h"
    )
    covered = np.zeros(len(hourly), dtype=bool)
    pad = pd.Timedelta(hours=EVENT_PAD_HR)

    for _, ev in events.iterrows():
        lo = np.searchsorted(hourly, ev["window_start"] - pad, side="left")
        hi = np.searchsorted(hourly, ev["window_end"] + pad, side="right")
        covered[lo:hi] = True

    return {
        "total_hours": len(hourly),
        "covered_hours": int(covered.sum()),
        "covered_fraction": float(covered.mean()),
        "discarded_fraction": float(1 - covered.mean()),
    }


def _coastline_patch(ax):
    """Draw coastlines if cartopy is available; otherwise skip silently."""
    try:
        import cartopy.feature as cfeature
        ax.add_feature(cfeature.LAND, facecolor="#e8e4dc", zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6, color="#555", zorder=3)
        return True
    except Exception:
        return False


def make_figures(tracks: pd.DataFrame, events: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    near = events[events["criterion"] == "near_hatteras"]
    era = tracks[tracks["year"].between(YEAR_START, YEAR_END)]
    near_ids = set(near["storm_id"])

    # ---- Figure 1: storm tracks over the model domain -------------------
    try:
        import cartopy.crs as ccrs
        proj = ccrs.PlateCarree()
        fig, ax = plt.subplots(figsize=(11, 8), subplot_kw={"projection": proj})
        ax.set_extent([LON_MIN - 8, LON_MAX + 6, LAT_MIN - 6, LAT_MAX + 4], crs=proj)
        has_map = _coastline_patch(ax)
        transform = {"transform": proj}
    except Exception:
        fig, ax = plt.subplots(figsize=(11, 8))
        ax.set_xlim(LON_MIN - 8, LON_MAX + 6)
        ax.set_ylim(LAT_MIN - 6, LAT_MAX + 4)
        has_map = False
        transform = {}

    colors = {
        "TD": "#9ecae1", "TS": "#4292c6", "Cat1": "#fdae61", "Cat2": "#f46d43",
        "Cat3": "#d73027", "Cat4": "#a50026", "Cat5": "#6a0011",
        "Unknown": "#bbbbbb",
    }
    cat_of = dict(zip(near["storm_id"], near["category_overall"]))
    for storm_id in near_ids:
        seg = era[era["storm_id"] == storm_id]
        ax.plot(seg["lon"], seg["lat"], lw=0.7, alpha=0.55,
                color=colors.get(cat_of.get(storm_id), "#999"), **transform)

    ax.plot([LON_MIN, LON_MAX, LON_MAX, LON_MIN, LON_MIN],
            [LAT_MIN, LAT_MIN, LAT_MAX, LAT_MAX, LAT_MIN],
            color="k", lw=2.0, ls="--", label="model domain (32x48 @ 0.5deg)",
            **transform)
    ax.plot(HATTERAS_LON, HATTERAS_LAT, marker="*", ms=18, color="#111",
            mec="w", mew=1.2, ls="none", label="Cape Hatteras", **transform)

    handles = [plt.Line2D([], [], color=c, lw=2, label=k)
               for k, c in colors.items() if k != "Unknown"]
    handles += [plt.Line2D([], [], color="k", lw=2, ls="--", label="model domain"),
                plt.Line2D([], [], color="#111", marker="*", ms=14, ls="none",
                           label="Cape Hatteras")]
    ax.legend(handles=handles, loc="upper left", fontsize=8, ncol=2, framealpha=0.9)
    ax.set_title(f"Atlantic storms passing within {LOCAL_RADIUS_KM:.0f} km of "
                 f"Cape Hatteras, {YEAR_START}-{YEAR_END}  (n={len(near_ids)})")
    if not has_map:
        ax.set_xlabel("longitude"); ax.set_ylabel("latitude"); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "01_storm_tracks_domain.png", dpi=150)
    plt.close(fig)

    # ---- Figure 2: per-year counts and seasonality ----------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))

    per_year = near.groupby("year").size().reindex(
        range(YEAR_START, YEAR_END + 1), fill_value=0)
    axes[0].bar(per_year.index, per_year.values, color="#4292c6")
    axes[0].plot(per_year.index, per_year.rolling(5, center=True).mean(),
                 color="#d73027", lw=2, label="5-yr mean")
    axes[0].set_title("Storms within 500 km of Hatteras, per year")
    axes[0].set_xlabel("year"); axes[0].set_ylabel("count"); axes[0].legend()

    order = ["TD", "TS", "Cat1", "Cat2", "Cat3", "Cat4", "Cat5"]
    tropical = near[near["status_at_closest"].isin(["TD", "TS", "HU"])]
    extratrop = near[~near["status_at_closest"].isin(["TD", "TS", "HU"])]
    months = np.arange(1, 13)
    axes[1].bar(months, [(tropical["month_at_closest"] == m).sum() for m in months],
                color="#d73027", label="tropical at closest approach")
    axes[1].bar(months, [(extratrop["month_at_closest"] == m).sum() for m in months],
                bottom=[(tropical["month_at_closest"] == m).sum() for m in months],
                color="#4292c6", label="extratropical / other")
    axes[1].set_xticks(months)
    axes[1].set_title("Seasonality at closest approach")
    axes[1].set_xlabel("month"); axes[1].set_ylabel("count"); axes[1].legend(fontsize=8)

    counts = near["category_at_closest"].value_counts().reindex(order, fill_value=0)
    axes[2].bar(counts.index, counts.values,
                color=[colors[c] for c in counts.index])
    axes[2].set_title("Intensity at closest approach")
    axes[2].set_ylabel("count")

    fig.tight_layout()
    fig.savefig(FIGURES / "02_storm_statistics.png", dpi=150)
    plt.close(fig)
    print(f"[eda] figures -> {FIGURES}")


def main() -> None:
    tracks, storms = hurdat2.load()
    events = qualifying_storms(tracks)

    near = events[events["criterion"] == "near_hatteras"]
    dom = events[events["criterion"] == "in_domain"]

    cov_near = coverage_hours(near)
    cov_dom = coverage_hours(dom)

    print("\n" + "=" * 72)
    print(f"HURDAT2 event catalogue, {YEAR_START}-{YEAR_END}")
    print("=" * 72)
    print(f"  storms within {LOCAL_RADIUS_KM:.0f} km of Hatteras : {len(near):4d}"
          f"   ({len(near) / (YEAR_END - YEAR_START + 1):.1f}/yr)")
    print(f"  storms entering the model domain        : {len(dom):4d}"
          f"   ({len(dom) / (YEAR_END - YEAR_START + 1):.1f}/yr)")

    print(f"\n  Record coverage (event window +/-{EVENT_PAD_HR} h):")
    for label, cov in (("near-Hatteras", cov_near), ("in-domain", cov_dom)):
        print(f"    {label:14s}  {cov['covered_hours']:7,d} / "
              f"{cov['total_hours']:,d} h  = {cov['covered_fraction']:6.2%} "
              f"covered,  {cov['discarded_fraction']:6.2%} DISCARDED "
              f"by paper-style subsetting")

    print("\n  Status at closest approach (near-Hatteras storms):")
    for status, n in near["status_at_closest"].value_counts().items():
        print(f"    {status:4s} {n:4d}  ({n / len(near):5.1%})")

    print("\n  Peak intensity at closest approach:")
    order = ["Cat5", "Cat4", "Cat3", "Cat2", "Cat1", "TS", "TD"]
    for cat in order:
        n = (near["category_at_closest"] == cat).sum()
        if n:
            print(f"    {cat:5s} {n:4d}  ({n / len(near):5.1%})")

    print("\n  Monthly distribution (near-Hatteras):")
    mnames = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for m in range(1, 13):
        n = (near["month_at_closest"] == m).sum()
        bar = "#" * int(round(n / max(1, near["month_at_closest"].value_counts().max()) * 40))
        print(f"    {mnames[m - 1]}  {n:3d}  {bar}")
    print("=" * 72 + "\n")

    events.to_csv(TABLES / "hurdat2_hatteras_events.csv", index=False)
    near.sort_values("min_dist_km").head(30).to_csv(
        TABLES / "hurdat2_closest_30.csv", index=False)
    tracks.to_parquet(INTERIM / "hurdat2_tracks.parquet", index=False)
    print(f"[eda] tables  -> {TABLES}")
    print(f"[eda] tracks  -> {INTERIM / 'hurdat2_tracks.parquet'}")

    make_figures(tracks, events)


if __name__ == "__main__":
    main()
