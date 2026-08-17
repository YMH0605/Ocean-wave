"""Three-way comparison: buoy observations, ERA5, and the forecast model.

The point is to separate two things that have been conflated so far. The model
under-predicts large waves relative to ERA5 -- but ERA5 is itself a model, and
if it also under-predicts relative to the ocean, part of that shortfall is
inherited rather than introduced. Only observations can split them:

    buoy  -  ERA5   =  error in the training target
    ERA5  -  model  =  error introduced by learning it
    buoy  -  model  =  what a user actually experiences

Sea states follow the WMO code, so the rows mean the same thing as in the wave
literature rather than being percentiles a reader has to look up.

    python validate_buoys.py
    python validate_buoys.py --lead 24
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from buoys import MATCHED, STATIONS
from config import FIGURES, TABLES, TEST_YEARS

# WMO sea state code. The source paper bins its results this way; using the
# same scale makes the tables directly comparable.
SEA_STATE = [
    ("1-3  (0-1.25 m)", 0.0, 1.25),
    ("4    (1.25-2.5 m)", 1.25, 2.5),
    ("5    (2.5-4 m)", 2.5, 4.0),
    ("6    (4-6 m)", 4.0, 6.0),
    ("7    (6-9 m)", 6.0, 9.0),
    ("8    (9-14 m)", 9.0, 14.0),
]


def metrics(pred: np.ndarray, ref: np.ndarray) -> dict:
    e = pred - ref
    ss_tot = np.sum((ref - ref.mean()) ** 2)
    big = ref > 0.5
    return {
        "n": len(ref),
        "MAE": float(np.mean(np.abs(e))),
        "RMSE": float(np.sqrt(np.mean(e ** 2))),
        "bias": float(np.mean(e)),
        "MAPE": float(np.mean(np.abs(e[big] / ref[big])) * 100) if big.any() else np.nan,
        "R2": float(1 - np.sum(e ** 2) / ss_tot) if ss_tot > 0 else np.nan,
        "R": float(np.corrcoef(pred, ref)[0, 1]) if len(ref) > 2 else np.nan,
    }


def by_sea_state(pred, ref, label) -> pd.DataFrame:
    rows = []
    for name, lo, hi in SEA_STATE:
        sel = (ref >= lo) & (ref < hi)
        if sel.sum() < 30:
            continue
        m = metrics(pred[sel], ref[sel])
        m.update(comparison=label, sea_state=name)
        rows.append(m)
    m = metrics(pred, ref)
    m.update(comparison=label, sea_state="All")
    rows.append(m)
    return pd.DataFrame(rows)


def show(df: pd.DataFrame, title: str) -> None:
    print(f"\n  {title}")
    print(f"    {'sea state':<20}{'n':>9}{'MAE':>8}{'RMSE':>8}{'bias':>9}"
          f"{'MAPE':>8}{'R2':>8}")
    for _, r in df.iterrows():
        print(f"    {r['sea_state']:<20}{int(r['n']):>9,}{r['MAE']:>8.3f}"
              f"{r['RMSE']:>8.3f}{r['bias']:>+9.3f}{r['MAPE']:>7.1f}%"
              f"{r['R2']:>8.3f}")


def load_model_at_buoys(lead: int, matched: pd.DataFrame) -> pd.DataFrame | None:
    """Attach forecast values at each buoy's grid cell, where they exist."""
    from config import OUTPUTS

    cache = OUTPUTS / f"predictions_h{lead}_all.npz"
    if not cache.exists():
        print(f"\n  (no {cache.name}; run the notebook's section 3 to build it)")
        return None

    d = np.load(cache, allow_pickle=True)
    model_names = [k for k in d.files
                   if k not in ("true", "valid", "t_index", "times", "ocean")]
    # The cache is strided, so only some hours have a forecast. Its t_index is
    # the ANCHOR time; the forecast it holds is valid at anchor + lead.
    valid_at = pd.Series(np.arange(len(d["t_index"])),
                         index=d["t_index"] + lead)

    pos = valid_at.reindex(matched["t_index"]).to_numpy()
    ok = ~np.isnan(pos)
    if ok.sum() == 0:
        print("\n  (no overlap between buoy hours and cached forecasts)")
        return None

    sub = matched.loc[ok].copy()
    rows = pos[ok].astype(int)
    for name in model_names:
        field = d[name]
        sub[name] = field[rows, sub["cell_i"].to_numpy(),
                          sub["cell_j"].to_numpy()]
    print(f"\n  {ok.sum():,} buoy hours have a +{lead} h forecast "
          f"({', '.join(model_names)})")
    return sub


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lead", type=int, default=1)
    args = ap.parse_args()

    if not MATCHED.exists():
        raise SystemExit("run buoys.py first")
    matched = pd.read_parquet(MATCHED)

    print("=" * 80)
    print(f"  BUOY VALIDATION — NDBC vs ERA5, test years "
          f"{TEST_YEARS[0]}-{TEST_YEARS[-1]}")
    print("=" * 80)
    print(f"  {len(matched):,} paired hours, "
          f"{matched['station'].nunique()} stations")

    buoy = matched["WVHT"].to_numpy()
    era5 = matched["era5_swh"].to_numpy()

    era_vs_buoy = by_sea_state(era5, buoy, "ERA5 vs buoy")
    show(era_vs_buoy, "ERA5 against the buoys  (positive bias = ERA5 too high)")

    # Per station, because the point-versus-cell-average mismatch varies with
    # how exposed and how close to shore a station is.
    print("\n  Per station (all sea states)")
    print(f"    {'stn':>6}  {'location':<24}{'setting':<11}{'n':>8}"
          f"{'MAE':>8}{'bias':>9}{'R':>7}")
    per_station = []
    for stn, grp in matched.groupby("station"):
        m = metrics(grp["era5_swh"].to_numpy(), grp["WVHT"].to_numpy())
        name, lat, lon, setting = STATIONS[stn]
        m.update(station=stn, name=name, setting=setting)
        per_station.append(m)
        print(f"    {stn:>6}  {name:<24}{setting:<11}{int(m['n']):>8,}"
              f"{m['MAE']:>8.3f}{m['bias']:>+9.3f}{m['R']:>7.3f}")
    pd.DataFrame(per_station).to_csv(TABLES / "buoy_era5_per_station.csv",
                                     index=False)

    # ---- add the model where forecasts exist ---------------------------
    with_model = load_model_at_buoys(args.lead, matched)
    frames = [era_vs_buoy]
    if with_model is not None:
        b = with_model["WVHT"].to_numpy()
        e = with_model["era5_swh"].to_numpy()
        models = [c for c in with_model.columns
                  if c in ("3D U-Net", "ConvLSTM", "3D CNN")]

        show(by_sea_state(e, b, "ERA5 vs buoy (subset)"),
             "ERA5 against the buoys, on the hours a forecast exists")

        for name in models:
            p = with_model[name].to_numpy()
            df_m = by_sea_state(p, b, f"{name} vs buoy")
            frames.append(df_m)
            show(df_m, f"{name} against the buoys")

        print("\n" + "=" * 80)
        print("  WHERE THE SHORTFALL COMES FROM  (rough seas, >= 4 m at the buoy)")
        print("=" * 80)
        rough = b >= 4.0
        print(f"    {rough.sum():,} hours\n")
        print(f"    ERA5   - buoy : {np.mean(e[rough] - b[rough]):+.3f} m"
              f"   <- error already in the training target")
        for name in models:
            p = with_model[name].to_numpy()
            print(f"    {name:<6} - buoy : {np.mean(p[rough] - b[rough]):+.3f} m"
                  f"   ({np.mean(p[rough] - e[rough]):+.3f} m of that "
                  f"introduced by the model)")

    out = TABLES / f"buoy_validation_h{args.lead}.csv"
    pd.concat(frames, ignore_index=True).to_csv(out, index=False)
    print(f"\n  -> {out}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
