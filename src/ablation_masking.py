"""Does our masked metric flatter the model relative to the paper's?

The paper zeroes land in both the field and the loss, and reports metrics over
the whole 80x80 image. We exclude land from both. Since a masked land cell is
trivially predicted (target and prediction are both zero), including it adds a
large number of zero-error points to the average -- which lowers MAE and raises
R^2 for free.

This script scores the same predictions both ways so the direction and size of
that effect is measured rather than argued about.

    python ablation_masking.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import TABLES
from dataset import ERA5Cache, WaveWindowDataset
from models import build
from train import CHECKPOINTS

CHECKPOINT = "unet3d_swh_full_lb48_h1"
STRIDE = 6


def metrics(pred, true, weight):
    """weight is 1 where the point counts, 0 where it does not."""
    w = weight.astype(bool)
    p, t = pred[w].astype(np.float64), true[w].astype(np.float64)
    err = p - t
    ss_tot = np.sum((t - t.mean()) ** 2)
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2": float(1 - np.sum(err ** 2) / ss_tot) if ss_tot > 0 else np.nan,
        "n": int(w.sum()),
    }


@torch.no_grad()
def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(CHECKPOINTS / f"{CHECKPOINT}.pt", map_location=device,
                      weights_only=False)
    targs = ckpt["args"]

    cache = ERA5Cache()
    ds = WaveWindowDataset(cache=cache, split="test", target=targs["target"],
                           channel_set=targs["channel_set"],
                           lookback=targs["lookback"], lead=targs["lead"],
                           stride=STRIDE)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4,
                        pin_memory=(device == "cuda"))

    model = build(targs["model"], in_channels=ds.n_channels,
                  lookback=targs["lookback"],
                  base_channels=targs["base_channels"],
                  depth=targs["depth"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    stats = cache.stats[targs["target"]]
    scale, offset = stats["max"] - stats["min"], stats["min"]

    preds, truths, valids, persists = [], [], [], []
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        ocean = batch["valid"].to(device, non_blocking=True).float()
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=(device == "cuda")):
            out = model(x, ocean)
        preds.append((out.float() * scale + offset).cpu().numpy())
        truths.append(batch["y_phys"].numpy())
        valids.append(batch["valid"].numpy())
        persists.append(batch["persist"].numpy())

    pred = np.concatenate(preds)
    true = np.concatenate(truths)
    valid = np.concatenate(valids)
    persist = np.concatenate(persists)

    # Paper convention: land is zero in the field and counts in the average.
    #
    # The network masks its own output, but the inverse min-max transform then
    # adds the offset back, so a masked cell comes out at stats["min"] rather
    # than at zero. Re-zero the land explicitly, otherwise the comparison
    # measures an artefact of the denormalisation instead of the convention.
    everything = np.ones_like(valid)
    truth_z = np.where(valid, true, 0.0)
    pred_z = np.where(valid, pred, 0.0)
    persist_z = np.where(valid, persist, 0.0)

    rows = []
    for model_name, p in (("3D U-Net", pred_z), ("persistence", persist_z)):
        for scheme, w in (("ocean only (ours)", valid),
                          ("all points, land=0 (paper)", everything)):
            m = metrics(p, truth_z, w)
            m.update(model=model_name, scheme=scheme)
            rows.append(m)

    df = pd.DataFrame(rows)[["model", "scheme", "mae", "rmse", "r2", "n"]]

    print("\n" + "=" * 76)
    print(f"Effect of the land mask on reported metrics -- {CHECKPOINT}")
    print("=" * 76)
    print(f"  land fraction of the grid: {1 - valid.mean():.1%}")
    print()
    print(df.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    print("\n" + "-" * 76)
    for model_name in ("3D U-Net", "persistence"):
        sub = df[df["model"] == model_name].set_index("scheme")
        ours = sub.loc["ocean only (ours)"]
        theirs = sub.loc["all points, land=0 (paper)"]
        print(f"  {model_name}: MAE {ours['mae']:.4f} (ours) vs "
              f"{theirs['mae']:.4f} (paper convention)  ->  "
              f"the paper's convention reports "
              f"{(theirs['mae'] - ours['mae']) / ours['mae']:+.0%}")
    print("-" * 76)
    print("  A negative percentage means the paper's convention makes the\n"
          "  SAME predictions look better. Our reported numbers are then the\n"
          "  conservative ones, and the comparison does not favour us.")
    print("=" * 76)

    out = TABLES / "ablation_masking.csv"
    df.to_csv(out, index=False)
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
