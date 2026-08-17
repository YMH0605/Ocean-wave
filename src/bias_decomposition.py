"""Split the forecast shortfall into what was inherited and what was learned.

Against ERA5 alone, a model that under-predicts large waves looks like a model
problem. But ERA5 is itself a wave model, and if it also under-predicts against
the ocean, part of that shortfall was in the training target before the network
saw it. With buoys as a third reference the two separate cleanly:

    ERA5  - buoy    bias already present in the training target
    model - ERA5    bias introduced by learning it
    model - buoy    what a user of the forecast actually gets

The first term does not change with lead time -- it is a property of the
reanalysis. The second grows as the forecast horizon lengthens. Where they
cross is where effort should shift from better training data to a better model,
and that crossover is the point of this table.

    python bias_decomposition.py
    python bias_decomposition.py --model ConvLSTM
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from buoys import MATCHED
from config import OUTPUTS, TABLES

SEA_STATE = [
    ("1-3 (0-1.25 m)", 0.0, 1.25),
    ("4   (1.25-2.5 m)", 1.25, 2.5),
    ("5   (2.5-4 m)", 2.5, 4.0),
    ("6+  (>4 m)", 4.0, 99.0),
]


def paired(lead: int, matched: pd.DataFrame, model: str):
    """Buoy, ERA5 and forecast values on the hours all three exist."""
    cache = OUTPUTS / f"predictions_h{lead}_all.npz"
    if not cache.exists():
        return None
    d = np.load(cache, allow_pickle=True)
    if model not in d.files:
        return None

    # t_index is the anchor hour; the forecast it holds is valid at anchor+lead.
    valid_at = pd.Series(np.arange(len(d["t_index"])), index=d["t_index"] + lead)
    pos = valid_at.reindex(matched["t_index"]).to_numpy()
    ok = ~np.isnan(pos)
    if ok.sum() == 0:
        return None

    sub = matched.loc[ok]
    rows = pos[ok].astype(int)
    fc = d[model][rows, sub["cell_i"].to_numpy(), sub["cell_j"].to_numpy()]
    return (sub["WVHT"].to_numpy(), sub["era5_swh"].to_numpy(),
            np.asarray(fc, dtype=np.float64))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="3D U-Net")
    ap.add_argument("--leads", nargs="*", type=int, default=[1, 6, 24, 48])
    args = ap.parse_args()

    matched = pd.read_parquet(MATCHED)
    rows = []

    print("=" * 92)
    print(f"  WHERE THE SHORTFALL COMES FROM — {args.model}, "
          f"NDBC buoys, test years 2021–2025")
    print("=" * 92)
    print("    Every column is a mean difference in metres. Negative means the")
    print("    first term is lower than the second — an under-prediction.\n")
    print(f"    {'lead':>6}  {'sea state':<19}{'n':>7}"
          f"{'ERA5-buoy':>12}{'model-ERA5':>12}{'model-buoy':>12}"
          f"{'inherited':>11}")

    for lead in args.leads:
        got = paired(lead, matched, args.model)
        if got is None:
            print(f"    {lead:>5}h  (no cached forecast)")
            continue
        buoy, era5, fc = got
        for label, lo, hi in SEA_STATE:
            sel = (buoy >= lo) & (buoy < hi)
            if sel.sum() < 30:
                continue
            d_era = float(np.mean(era5[sel] - buoy[sel]))
            d_mod = float(np.mean(fc[sel] - era5[sel]))
            d_tot = float(np.mean(fc[sel] - buoy[sel]))
            # How much of the total shortfall was already in the target.
            share = abs(d_era) / abs(d_tot) if abs(d_tot) > 1e-9 else np.nan
            rows.append({"lead": lead, "sea_state": label, "n": int(sel.sum()),
                         "era5_minus_buoy": d_era, "model_minus_era5": d_mod,
                         "model_minus_buoy": d_tot, "inherited_share": share})
            print(f"    {lead:>5}h  {label:<19}{int(sel.sum()):>7,}"
                  f"{d_era:>+12.3f}{d_mod:>+12.3f}{d_tot:>+12.3f}"
                  f"{share:>10.0%}")
        print()

    df = pd.DataFrame(rows)
    out = TABLES / "bias_decomposition.csv"
    df.to_csv(out, index=False)

    rough = df[df["sea_state"].str.startswith("6+")]
    if not rough.empty:
        print("=" * 92)
        print("  ROUGH SEAS (>4 m at the buoy) — how the balance shifts with "
              "lead time")
        print("=" * 92)
        print(f"    {'lead':>6}{'ERA5-buoy':>12}{'model-ERA5':>12}"
              f"{'inherited':>12}   which term dominates")
        for _, r in rough.iterrows():
            who = ("the training target" if abs(r["era5_minus_buoy"])
                   > abs(r["model_minus_era5"]) else "the model")
            print(f"    {int(r['lead']):>5}h{r['era5_minus_buoy']:>+12.3f}"
                  f"{r['model_minus_era5']:>+12.3f}"
                  f"{r['inherited_share']:>11.0%}   {who}")

    print(f"\n  -> {out}")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
