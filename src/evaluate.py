"""Evaluation harness: metrics in physical units, stratified, against baselines.

Three things the source paper does not do, all of which change how its numbers
should be read:

  1. Every model is scored against PERSISTENCE and CLIMATOLOGY at the same lead
     time. A skill score below zero means the model is worse than doing nothing.
  2. Metrics are stratified by SWH percentile. A domain-wide MAE is dominated by
     the calm majority of the record, so a model can post an excellent headline
     score while being useless in the storms the work is supposedly about.
  3. Only ocean points count. Including masked land inflates R^2 (land is
     perfectly predictable) and deflates MAE.

MAPE is reported for comparability with the paper but should be treated with
suspicion: it explodes as SWH approaches zero, which is most of the record.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-6


def compute_metrics(pred: np.ndarray, true: np.ndarray,
                    valid: np.ndarray) -> dict:
    """Point metrics over the valid (ocean, finite) entries only."""
    m = valid.astype(bool)
    if m.sum() == 0:
        return {k: np.nan for k in
                ("mae", "rmse", "bias", "mape", "r2", "n")}

    p, t = pred[m].astype(np.float64), true[m].astype(np.float64)
    err = p - t

    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((t - t.mean()) ** 2))

    # MAPE only over points with meaningful wave height, else it diverges.
    big = t > 0.5
    mape = float(np.mean(np.abs(err[big] / t[big])) * 100) if big.any() else np.nan

    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "bias": float(np.mean(err)),
        "mape": mape,
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
        "n": int(m.sum()),
    }


def skill_score(pred: np.ndarray, true: np.ndarray, ref: np.ndarray,
                valid: np.ndarray) -> float:
    """1 - MSE(model)/MSE(reference). Positive means it beats the reference."""
    m = valid.astype(bool)
    mse_model = float(np.mean((pred[m] - true[m]) ** 2))
    mse_ref = float(np.mean((ref[m] - true[m]) ** 2))
    return 1.0 - mse_model / mse_ref if mse_ref > 0 else np.nan


def percentile_strata(true: np.ndarray, valid: np.ndarray,
                      edges=(50, 90, 99)) -> dict:
    """Boolean masks splitting valid points by percentile of the truth field.

    Percentiles are taken over the observed distribution of the evaluation set,
    so the top bin is genuinely the extreme tail of this domain rather than an
    arbitrary wave height.
    """
    m = valid.astype(bool)
    thresholds = np.percentile(true[m], edges)

    strata, lo = {}, -np.inf
    labels = ([f"P0-{edges[0]}"]
              + [f"P{a}-{b}" for a, b in zip(edges[:-1], edges[1:])]
              + [f"P{edges[-1]}+"])
    for label, hi in zip(labels, list(thresholds) + [np.inf]):
        strata[label] = m & (true >= lo) & (true < hi)
        lo = hi

    strata["_thresholds"] = dict(zip([f"P{e}" for e in edges],
                                     [float(t) for t in thresholds]))
    return strata


def evaluate(
    predictions: dict[str, np.ndarray],
    truth: np.ndarray,
    valid: np.ndarray,
    reference: str = "persistence",
    strata: dict | None = None,
) -> pd.DataFrame:
    """Score every model in `predictions` overall and within each stratum.

    predictions : {model_name: array like truth}
    truth       : (N, 1, H, W) physical units
    valid       : (N, 1, H, W) bool
    strata      : optional {label: bool mask}; percentile strata if omitted
    """
    if strata is None:
        strata = percentile_strata(truth, valid)
    thresholds = strata.pop("_thresholds", {})

    ref_pred = predictions.get(reference)
    rows = []

    for name, pred in predictions.items():
        overall = compute_metrics(pred, truth, valid)
        overall["model"] = name
        overall["stratum"] = "all"
        if ref_pred is not None and name != reference:
            overall["skill_vs_" + reference] = skill_score(
                pred, truth, ref_pred, valid)
        rows.append(overall)

        for label, mask in strata.items():
            row = compute_metrics(pred, truth, mask)
            row["model"] = name
            row["stratum"] = label
            if ref_pred is not None and name != reference:
                row["skill_vs_" + reference] = skill_score(
                    pred, truth, ref_pred, mask)
            rows.append(row)

    df = pd.DataFrame(rows)
    cols = ["model", "stratum", "mae", "rmse", "bias", "mape", "r2", "n"]
    cols += [c for c in df.columns if c.startswith("skill_vs_")]
    df = df[cols]
    df.attrs["thresholds"] = thresholds
    return df


def format_report(df: pd.DataFrame, target: str = "swh",
                  lead_hours: int = 1) -> str:
    """Human-readable table, the thing that actually goes in the paper."""
    unit = "m" if target == "swh" else "s"
    thresholds = df.attrs.get("thresholds", {})

    lines = [
        "=" * 88,
        f"{target.upper()} forecast, lead time +{lead_hours} h",
        "=" * 88,
    ]
    if thresholds:
        lines.append("  stratum thresholds: " + ", ".join(
            f"{k}={v:.2f}{unit}" for k, v in thresholds.items()))
        lines.append("")

    skill_cols = [c for c in df.columns if c.startswith("skill_vs_")]
    skill_col = skill_cols[0] if skill_cols else None

    for stratum in ["all"] + [s for s in df["stratum"].unique() if s != "all"]:
        sub = df[df["stratum"] == stratum]
        if sub.empty:
            continue
        lines.append(f"  [{stratum}]  n={sub['n'].iloc[0]:,}")
        header = f"    {'model':<14}{'MAE':>9}{'RMSE':>9}{'bias':>9}{'R2':>8}"
        if skill_col:
            header += f"{'skill':>9}"
        lines.append(header)
        for _, r in sub.sort_values("mae").iterrows():
            line = (f"    {r['model']:<14}{r['mae']:>9.3f}{r['rmse']:>9.3f}"
                    f"{r['bias']:>9.3f}{r['r2']:>8.3f}")
            if skill_col:
                sk = r.get(skill_col)
                line += f"{sk:>9.3f}" if pd.notna(sk) else f"{'  ref':>9}"
            lines.append(line)
        lines.append("")

    lines.append("  skill = 1 - MSE/MSE_persistence;  <=0 means no better "
                 "than persistence")
    lines.append("=" * 88)
    return "\n".join(lines)
