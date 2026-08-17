"""Compare every architecture trained at one lead time, in one command.

`predict.py` scores a list of checkpoints you name by hand, which is fine for a
one-off but tedious and easy to get wrong -- name a +1 h and a +24 h model in
the same call and they are not even scored on the same samples. This finds the
checkpoints for a given lead time itself, scores them together, and prints the
comparison table with parameter counts alongside, so architectures can be judged
at the capacity they actually used.

    python compare_models.py                    # +1 h
    python compare_models.py --lead 24
    python compare_models.py --lead 1 --strata all P99+
    python compare_models.py --all-leads        # one table per lead time
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch

from config import TABLES
from models import build, count_parameters
from train import CHECKPOINTS

SRC = Path(__file__).resolve().parent

PRETTY = {"unet3d": "3D U-Net", "cnn3d": "3D CNN", "convlstm": "ConvLSTM"}


def discover(lead: int) -> list[str]:
    """Checkpoints trained at this lead time, main runs before variants."""
    found = [p.stem for p in CHECKPOINTS.glob(f"*_h{lead}.pt")]
    # Put the plain full-channel runs first so the table reads in a sensible
    # order, then anything else (paper-channel runs, tagged experiments).
    main = sorted(s for s in found if "_full_" in s and "_hp_" not in s)
    rest = sorted(s for s in found if s not in main)
    return main + rest


def describe(stem: str) -> dict:
    """Architecture, capacity and parameter count, read from the checkpoint."""
    ckpt = torch.load(CHECKPOINTS / f"{stem}.pt", map_location="cpu",
                      weights_only=False)
    a = ckpt["args"]
    caps = dict(base_channels=a.get("base_channels", 32), depth=a.get("depth", 3))
    if a["model"] == "convlstm" and "convlstm_capacity" not in ckpt:
        caps = dict(base_channels=32, depth=2)
    n_par = count_parameters(build(a["model"], in_channels=8,
                                   lookback=a["lookback"], **caps))

    label = PRETTY.get(a["model"], a["model"])
    if a.get("channel_set", "full").startswith("paper"):
        label += " (paper inputs)"
    if a.get("tag"):
        label += f" [{a['tag']}]"

    return {"checkpoint": stem, "model": label, "lr": a.get("lr"),
            "base": caps["base_channels"], "params": n_par,
            "epochs_to_best": ckpt.get("epoch")}


def score(lead: int, stems: list[str], stride: int, workers: int) -> pd.DataFrame:
    """Run predict.py over these checkpoints and read back what it wrote."""
    out = TABLES / f"test_metrics_swh_h{lead}.csv"
    cmd = [sys.executable, "-u", "predict.py", "--checkpoints", *stems,
           "--stride", str(stride), "--num-workers", str(workers)]
    print(f"[compare] scoring {len(stems)} model(s) at +{lead} h ...")
    proc = subprocess.run(cmd, cwd=SRC, text=True, capture_output=True)
    if proc.returncode != 0 or not out.exists():
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-4:]
        print("[compare] predict.py failed:\n  " + "\n  ".join(tail))
        raise SystemExit(1)
    for line in proc.stdout.splitlines():
        if "SKIPPING" in line or "test samples" in line:
            print("  " + line.strip())
    return pd.read_csv(out)


def table(lead: int, strata: list[str], stride: int, workers: int) -> None:
    stems = discover(lead)
    if not stems:
        print(f"no checkpoints found for +{lead} h")
        return

    meta = pd.DataFrame([describe(s) for s in stems]).set_index("checkpoint")
    metrics = score(lead, stems, stride, workers)
    metrics = metrics[metrics["checkpoint" if "checkpoint" in metrics
                              else "model"].isin(stems)] \
        if "checkpoint" in metrics else metrics[metrics["model"].isin(stems)]

    print("\n" + "=" * 88)
    print(f"  ARCHITECTURE COMPARISON — SWH at +{lead} h, "
          f"test years 2021–2025")
    print("=" * 88)

    for stratum in strata:
        sub = metrics[metrics["stratum"] == stratum]
        sub = sub[sub["model"].isin(stems)]
        if sub.empty:
            continue
        rows = []
        for _, r in sub.iterrows():
            m = meta.loc[r["model"]]
            rows.append({"model": m["model"], "params": m["params"],
                         "lr": m["lr"], "base": m["base"],
                         "MAE": r["mae"], "RMSE": r["rmse"],
                         "bias": r["bias"], "R2": r["r2"]})
        df = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)

        n = int(sub["n"].iloc[0])
        print(f"\n  [{stratum}]   n = {n:,} ocean points")
        print(f"    {'model':<26}{'params':>11}{'lr':>8}{'base':>6}"
              f"{'MAE':>8}{'RMSE':>8}{'bias':>9}{'R2':>8}")
        for _, r in df.iterrows():
            print(f"    {r['model']:<26}{r['params']:>11,}{r['lr']:>8g}"
                  f"{int(r['base']):>6}{r['MAE']:>8.3f}{r['RMSE']:>8.3f}"
                  f"{r['bias']:>+9.3f}{r['R2']:>8.3f}")

        if stratum == strata[0]:
            best = df.iloc[0]
            worst = df.iloc[-1]
            print(f"\n    best: {best['model']} ({best['MAE']:.3f} m)   "
                  f"worst: {worst['model']} ({worst['MAE']:.3f} m)   "
                  f"spread {worst['MAE'] / best['MAE']:.2f}x")

    dest = TABLES / f"comparison_h{lead}.csv"
    merged = metrics.merge(meta.reset_index(), left_on="model",
                           right_on="checkpoint", how="left",
                           suffixes=("", "_meta"))
    merged.to_csv(dest, index=False)
    print(f"\n  -> {dest}")
    print("=" * 88)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lead", type=int, default=1)
    ap.add_argument("--all-leads", action="store_true")
    ap.add_argument("--strata", nargs="*",
                    default=["all", "P90-99", "P99+"])
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    if args.all_leads:
        leads = sorted({int(m.group(1))
                        for p in CHECKPOINTS.glob("*.pt")
                        if (m := re.search(r"_h(\d+)$", p.stem))})
        print(f"[compare] lead times found: {leads}\n")
    else:
        leads = [args.lead]

    for lead in leads:
        table(lead, args.strata, args.stride, args.num_workers)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
