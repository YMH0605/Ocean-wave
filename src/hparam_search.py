"""Short-run hyperparameter search, so architectures are compared fairly.

The first comparison in this project trained every architecture on one fixed
set of hyperparameters. That is defensible as "same budget for everyone", but
it silently assumes one learning rate and one capacity suit all three models,
and it left ConvLSTM at 461k parameters against the U-Net's 5.9M because its
width was hard-coded. Any conclusion about which architecture is better was
therefore partly a statement about whose defaults happened to fit.

This runs a staged search instead:

  stage 1  learning rate, at fixed capacity
  stage 2  capacity, at each model's best learning rate

Runs are deliberately short (few epochs, heavily strided) — enough to rank
configurations, not to produce final numbers. The winning configuration is then
trained properly. Short-run ranking is not a perfect proxy for full-run
ranking, which is a limitation worth stating rather than hiding.

Everything is written under outputs/hparam_search/ so real results are never
touched, and completed runs are skipped so the search is resumable.

    python hparam_search.py                 # both stages
    python hparam_search.py --stage 1
    python hparam_search.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from config import OUTPUTS

SRC = Path(__file__).resolve().parent
SEARCH_DIR = OUTPUTS / "hparam_search"
SEARCH_DIR.mkdir(parents=True, exist_ok=True)
RESULTS = SEARCH_DIR / "results.csv"

MODELS = ("unet3d", "cnn3d", "convlstm")
LEARNING_RATES = (3e-4, 1e-3, 3e-3)
CAPACITIES = ((16, 3), (32, 3), (48, 3))       # (base_channels, depth)

# Short-run budget. stride 12 keeps ~12k training windows instead of ~50k.
COMMON = dict(target="swh", lead=1, lookback=48, stride=12, batch_size=32,
              epochs=6, num_workers=4, patience=6)


def run_one(model: str, lr: float, base: int, depth: int) -> dict | None:
    tag = f"hp_{model}_lr{lr:g}_c{base}d{depth}"
    log = SEARCH_DIR / "logs" / f"{model}_swh_full_lb48_h1_{tag}.json"

    if log.exists():
        hist = json.loads(log.read_text())
    else:
        cmd = [sys.executable, "-u", "train.py", "--model", model,
               "--target", COMMON["target"], "--lead", str(COMMON["lead"]),
               "--lookback", str(COMMON["lookback"]),
               "--stride", str(COMMON["stride"]),
               "--batch-size", str(COMMON["batch_size"]),
               "--epochs", str(COMMON["epochs"]),
               "--num-workers", str(COMMON["num_workers"]),
               "--patience", str(COMMON["patience"]),
               "--lr", str(lr), "--base-channels", str(base),
               "--depth", str(depth), "--tag", tag]
        env = dict(os.environ, WAVE_OUTPUT_DIR=str(SEARCH_DIR))
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=SRC, env=env, text=True,
                              capture_output=True)
        mins = (time.time() - t0) / 60
        if proc.returncode != 0 or not log.exists():
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-2:]
            print(f"    FAILED ({mins:.1f} min): {' | '.join(tail)}")
            return None
        hist = json.loads(log.read_text())
        n_par = re.search(r"([\d,]+) parameters", proc.stdout)
        params = int(n_par.group(1).replace(",", "")) if n_par else None
        best = min(hist, key=lambda r: r["val_loss"])
        print(f"    val MAE {best['val_mae']:.4f} m   "
              f"({params:,} params, {mins:.1f} min)" if params else
              f"    val MAE {best['val_mae']:.4f} m   ({mins:.1f} min)")
        return {"model": model, "lr": lr, "base_channels": base, "depth": depth,
                "params": params, "val_mae": best["val_mae"],
                "val_loss": best["val_loss"], "best_epoch": best["epoch"],
                "minutes": round(mins, 1)}

    best = min(hist, key=lambda r: r["val_loss"])
    print(f"    [cached] val MAE {best['val_mae']:.4f} m")
    return {"model": model, "lr": lr, "base_channels": base, "depth": depth,
            "params": None, "val_mae": best["val_mae"],
            "val_loss": best["val_loss"], "best_epoch": best["epoch"],
            "minutes": 0.0}


def save(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS, index=False)
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, choices=[1, 2], default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    stage1 = [(m, lr, 32, 3) for m in MODELS for lr in LEARNING_RATES]
    n_total = len(stage1) + len(MODELS) * (len(CAPACITIES) - 1)
    print(f"[search] stage 1: {len(stage1)} runs (learning rate)")
    print(f"[search] stage 2: {len(MODELS) * (len(CAPACITIES) - 1)} runs "
          f"(capacity, at each model's best lr)")
    print(f"[search] {n_total} runs total, roughly "
          f"{n_total * 6 / 60:.1f}–{n_total * 10 / 60:.1f} h\n")
    if args.dry_run:
        return 0

    rows = pd.read_csv(RESULTS).to_dict("records") if RESULTS.exists() else []
    done = {(r["model"], r["lr"], r["base_channels"], r["depth"]) for r in rows}

    if args.stage in (None, 1):
        print("=" * 66 + "\n  STAGE 1 — learning rate\n" + "=" * 66)
        for m, lr, b, d in stage1:
            if (m, lr, b, d) in done:
                continue
            print(f"  {m}  lr={lr:g}")
            r = run_one(m, lr, b, d)
            if r:
                rows.append(r)
                save(rows)

    df = save(rows)
    best_lr = {m: df[df.model == m].sort_values("val_mae")["lr"].iloc[0]
               for m in MODELS if (df.model == m).any()}
    print("\n  best learning rate so far:")
    for m, lr in best_lr.items():
        print(f"    {m:<10} {lr:g}")

    if args.stage in (None, 2):
        print("\n" + "=" * 66 + "\n  STAGE 2 — capacity\n" + "=" * 66)
        for m in MODELS:
            if m not in best_lr:
                continue
            for b, d in CAPACITIES:
                if (b, d) == (32, 3) or (m, best_lr[m], b, d) in done:
                    continue
                print(f"  {m}  base={b} depth={d}  (lr={best_lr[m]:g})")
                r = run_one(m, best_lr[m], b, d)
                if r:
                    rows.append(r)
                    save(rows)

    df = save(rows)
    print("\n" + "=" * 66 + "\n  BEST CONFIGURATION PER ARCHITECTURE\n" + "=" * 66)
    print(f"  {'model':<10}{'lr':>9}{'base':>6}{'depth':>7}{'params':>11}"
          f"{'val MAE':>10}")
    for m in MODELS:
        sub = df[df.model == m].sort_values("val_mae")
        if sub.empty:
            continue
        r = sub.iloc[0]
        p = f"{int(r['params']):,}" if pd.notna(r["params"]) else "—"
        print(f"  {m:<10}{r['lr']:>9g}{int(r['base_channels']):>6}"
              f"{int(r['depth']):>7}{p:>11}{r['val_mae']:>10.4f}")
    print(f"\n  full grid -> {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
