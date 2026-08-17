"""Generate the cached forecast fields for every lead time.

Inference only -- no training. The caches feed the notebook figures and the
buoy decomposition, and writing them once means those can be regenerated in
seconds afterwards.

    python build_caches.py                 # every lead with a checkpoint
    python build_caches.py --leads 24
"""

from __future__ import annotations

import argparse
import re

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import OUTPUTS
from dataset import ERA5Cache, WaveWindowDataset
from models import build
from train import CHECKPOINTS

PRETTY = {"unet3d": "3D U-Net", "cnn3d": "3D CNN", "convlstm": "ConvLSTM"}


def checkpoints_for(lead: int) -> dict[str, str]:
    out = {}
    for p in sorted(CHECKPOINTS.glob(f"*_full_lb48_h{lead}.pt")):
        arch = p.stem.split("_")[0]
        if arch in PRETTY:
            out[PRETTY[arch]] = p.stem
    return out


@torch.no_grad()
def build(lead: int, stride: int, force: bool) -> None:  # noqa: A001
    dest = OUTPUTS / f"predictions_h{lead}_all.npz"
    if dest.exists() and not force:
        print(f"[cache] +{lead:>2} h  already built, skipping")
        return

    models = checkpoints_for(lead)
    if not models:
        print(f"[cache] +{lead:>2} h  no checkpoints")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache = ERA5Cache()
    out, shared = {}, None

    for name, stem in models.items():
        ckpt = torch.load(CHECKPOINTS / f"{stem}.pt", map_location=device,
                          weights_only=False)
        a = ckpt["args"]
        ds = WaveWindowDataset(cache=cache, split="test", target=a["target"],
                               channel_set=a["channel_set"],
                               lookback=a["lookback"], lead=a["lead"],
                               stride=stride)
        loader = DataLoader(ds, batch_size=32, num_workers=4, shuffle=False)

        caps = dict(base_channels=a.get("base_channels", 32),
                    depth=a.get("depth", 3))
        if a["model"] == "convlstm" and "convlstm_capacity" not in ckpt:
            caps = dict(base_channels=32, depth=2)
        from models import build as make
        model = make(a["model"], in_channels=ds.n_channels,
                     lookback=a["lookback"], **caps).to(device)
        model.load_state_dict(ckpt["model"]); model.eval()

        s = cache.stats[a["target"]]
        scale, offset = s["max"] - s["min"], s["min"]
        P, T, V, I = [], [], [], []
        for b in loader:
            x = b["x"].to(device); ocn = b["valid"].to(device).float()
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=(device == "cuda")):
                o = model(x, ocn)
            P.append((o.float() * scale + offset).cpu().numpy()[:, 0]
                     .astype("float32"))
            if shared is None:
                T.append(b["y_phys"].numpy()[:, 0])
                V.append(b["valid"].numpy()[:, 0])
                I.append(b["t_index"].numpy())
        out[name] = np.concatenate(P)
        if shared is None:
            idx = np.concatenate(I)
            shared = dict(true=np.concatenate(T), valid=np.concatenate(V),
                          t_index=idx, times=cache.times[idx + lead],
                          ocean=cache.ocean)
        print(f"           {name}")

    np.savez_compressed(dest, **out, **shared)
    print(f"[cache] +{lead:>2} h  {len(models)} model(s) -> {dest.name} "
          f"({dest.stat().st_size / 1e6:.0f} MB)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leads", nargs="*", type=int, default=None)
    ap.add_argument("--stride", type=int, default=12)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    leads = args.leads or sorted({int(m.group(1)) for p in CHECKPOINTS.glob("*.pt")
                                  if (m := re.search(r"_h(\d+)$", p.stem))})
    print(f"[cache] lead times: {leads}\n")
    for lead in leads:
        build(lead, args.stride, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
