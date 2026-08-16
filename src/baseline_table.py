"""Reference table: how well do you do by not modelling anything at all?

Computes persistence and climatology error directly from the cache, for every
lead time, split and SWH stratum. No GPU, no training, no Dataset machinery --
just array shifts.

This is the table the source paper is missing. Its headline result, MAE 0.14 m
for a +1 h SWH forecast, cannot be interpreted without knowing what persistence
scores on the same data; every model in this project is reported against these
numbers.

    python baseline_table.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import TABLES, TEST_YEARS
from dataset import VAL_YEARS, ERA5Cache
from preprocess import CHANNEL_INDEX

LEADS = (1, 3, 6, 12, 24, 48)
PERCENTILES = (50, 90, 99)


def split_mask(cache: ERA5Cache, split: str) -> np.ndarray:
    if split == "test":
        return np.isin(cache.years, TEST_YEARS)
    if split == "val":
        return np.isin(cache.years, VAL_YEARS)
    return ~np.isin(cache.years, TEST_YEARS) & ~np.isin(cache.years, VAL_YEARS)


def load_field(cache: ERA5Cache, target: str) -> np.ndarray:
    """The full target field over ocean points, (time, n_ocean)."""
    ci = CHANNEL_INDEX[target]
    ocean = cache.ocean
    n = cache.shape[0]
    out = np.empty((n, int(ocean.sum())), dtype=np.float32)
    for start in range(0, n, 20000):
        out[start:start + 20000] = cache.data[start:start + 20000, ci][:, ocean]
    return out


def metrics(pred, true):
    err = pred - true
    finite = np.isfinite(err)
    err = err[finite]
    true_f = true[finite]
    ss_tot = np.sum((true_f - true_f.mean()) ** 2)
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "bias": float(np.mean(err)),
        "r2": float(1 - np.sum(err ** 2) / ss_tot) if ss_tot > 0 else np.nan,
        "n": int(err.size),
    }


def main() -> int:
    cache = ERA5Cache()
    hours = cache.times.astype("int64")

    for target in ("swh", "mp2"):
        field = load_field(cache, target)
        unit = "m" if target == "swh" else "s"
        rows = []

        for split in ("val", "test"):
            in_split = split_mask(cache, split)
            train_field = field[split_mask(cache, "train")]
            clim = np.nanmean(train_field, axis=0)          # per-cell mean
            thresholds = np.nanpercentile(field[in_split], PERCENTILES)

            for lead in LEADS:
                # A pair (t, t+lead) is usable only if both ends are in the
                # split and the record is unbroken across the gap.
                t = np.flatnonzero(in_split)
                t = t[t + lead < len(in_split)]
                t = t[in_split[t + lead]]
                t = t[(hours[t + lead] - hours[t]) == lead]

                true = field[t + lead]
                persist = field[t]

                for name, pred in (("persistence", persist),
                                   ("climatology",
                                    np.broadcast_to(clim, true.shape))):
                    m = metrics(pred, true)
                    m.update(target=target, split=split, lead=lead,
                             model=name, stratum="all")
                    rows.append(m)

                # Same, restricted to the tail of the truth distribution.
                lo = -np.inf
                labels = [f"P0-{PERCENTILES[0]}"]
                labels += [f"P{a}-{b}" for a, b in
                           zip(PERCENTILES[:-1], PERCENTILES[1:])]
                labels += [f"P{PERCENTILES[-1]}+"]
                for label, hi in zip(labels, list(thresholds) + [np.inf]):
                    sel = (true >= lo) & (true < hi)
                    lo = hi
                    if sel.sum() == 0:
                        continue
                    for name, pred in (("persistence", persist),
                                       ("climatology",
                                        np.broadcast_to(clim, true.shape))):
                        m = metrics(pred[sel], true[sel])
                        m.update(target=target, split=split, lead=lead,
                                 model=name, stratum=label)
                        rows.append(m)

        df = pd.DataFrame(rows)
        out = TABLES / f"baseline_{target}.csv"
        df.to_csv(out, index=False)

        print("\n" + "=" * 78)
        print(f"PERSISTENCE BASELINE -- {target.upper()} ({unit}), test years "
              f"{TEST_YEARS[0]}-{TEST_YEARS[-1]}")
        print("=" * 78)
        sub = df[(df["split"] == "test") & (df["model"] == "persistence")]
        strata = ["all"] + [s for s in sub["stratum"].unique() if s != "all"]
        head = f"  {'lead':>5} " + "".join(f"{s:>12}" for s in strata)
        print(head)
        print(f"  {'':>5} " + "".join(f"{'MAE ' + unit:>12}" for _ in strata))
        for lead in LEADS:
            row = f"  {lead:>4}h "
            for s in strata:
                v = sub[(sub["lead"] == lead) & (sub["stratum"] == s)]["mae"]
                row += f"{v.iloc[0]:>12.3f}" if len(v) else f"{'-':>12}"
            print(row)

        clim_all = df[(df["split"] == "test") & (df["model"] == "climatology")
                      & (df["stratum"] == "all") & (df["lead"] == 1)]
        print(f"\n  climatology (lead-independent): "
              f"MAE {clim_all['mae'].iloc[0]:.3f} {unit}, "
              f"RMSE {clim_all['rmse'].iloc[0]:.3f} {unit}")
        print(f"  -> {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
