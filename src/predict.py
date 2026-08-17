"""Run trained models over the held-out test years and produce the comparison.

This is the script that generates the table the reproduction is actually
about: every model, plus persistence and climatology, scored on the same test
samples, overall and within each SWH percentile stratum and storm regime.

    python predict.py --checkpoints unet3d_swh_full_lb48_h1 cnn3d_swh_full_lb48_h1
    python predict.py --all --lead 1
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import TABLES
from dataset import ERA5Cache, WaveWindowDataset
from evaluate import evaluate, format_report, percentile_strata
from models import build
from train import CHECKPOINTS


@torch.no_grad()
def run_model(model, loader, device, cache, target, amp_dtype):
    """Accumulate physical-unit predictions over the whole loader."""
    model.eval()
    stats = cache.stats[target]
    scale, offset = stats["max"] - stats["min"], stats["min"]

    out = []
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        ocean = batch["valid"].to(device, non_blocking=True).float()
        with torch.autocast("cuda", dtype=amp_dtype,
                            enabled=amp_dtype is not None):
            pred = model(x, ocean)
        out.append((pred.float() * scale + offset).cpu().numpy())
    return np.concatenate(out)


def collect_truth(loader):
    truth, valid, persist, t_index = [], [], [], []
    for batch in loader:
        truth.append(batch["y_phys"].numpy())
        valid.append(batch["valid"].numpy())
        persist.append(batch["persist"].numpy())
        t_index.append(batch["t_index"].numpy())
    return (np.concatenate(truth), np.concatenate(valid),
            np.concatenate(persist), np.concatenate(t_index))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--target", default="swh")
    ap.add_argument("--lead", type=int, default=1)
    ap.add_argument("--stride", type=int, default=6,
                    help="test-set subsampling; 6 keeps ~9k samples/year")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.bfloat16 if device == "cuda" else None

    names = args.checkpoints or []
    if args.all:
        names = [p.stem for p in CHECKPOINTS.glob("*.pt")]

    if not names:
        # Tell the caller what is actually available rather than just refusing.
        # Grouping by lead time matters because only one group can be evaluated
        # per invocation -- see the signature check below.
        found = sorted(p.stem for p in CHECKPOINTS.glob("*.pt"))
        if not found:
            print(f"No trained models in {CHECKPOINTS}.\n"
                  f"Train one first:  python train.py --model unet3d --lead 1")
            return 1

        by_lead: dict[str, list[str]] = {}
        for stem in found:
            lead = stem.rsplit("_h", 1)[-1] if "_h" in stem else "?"
            by_lead.setdefault(lead, []).append(stem)

        print("Which model(s) should I score? Pass them with --checkpoints.\n")
        print(f"Available in {CHECKPOINTS.name}/ "
              f"(one lead time per run — they are scored on different samples):\n")
        for lead in sorted(by_lead, key=lambda s: int(s) if s.isdigit() else 999):
            print(f"  +{lead} h")
            for stem in by_lead[lead]:
                print(f"      {stem}")
        example = " ".join(by_lead[min(by_lead, key=lambda s: int(s)
                                       if s.isdigit() else 999)])
        print(f"\nFor example:\n  python predict.py --checkpoints {example}")
        return 1

    cache = ERA5Cache(load_to_ram=False)
    predictions, truth = {}, None
    signature = None

    for name in names:
        ckpt_path = CHECKPOINTS / f"{name}.pt"
        if not ckpt_path.exists():
            print(f"[predict] missing {ckpt_path}, skipping")
            continue
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        targs = ckpt["args"]

        # The network outputs normalised values; converting them back to metres
        # uses the min/max of the cache. If this cache was built from a
        # different set of years than the one the checkpoint was trained on,
        # that conversion is wrong -- and wrong quietly, since the numbers stay
        # plausible. Refuse rather than report nonsense.
        saved = ckpt.get("target_stats")
        current = cache.stats[targs["target"]]
        if saved is not None:
            drift = max(abs(saved["min"] - current["min"]),
                        abs(saved["max"] - current["max"]))
            if drift > 1e-4:
                print(f"[predict] SKIPPING {name}: it was trained against a "
                      f"cache normalised to "
                      f"[{saved['min']:.3f}, {saved['max']:.3f}] but this one "
                      f"is [{current['min']:.3f}, {current['max']:.3f}]. "
                      f"Rebuild the cache from the same years, or retrain.")
                continue

        # Models are only comparable if they predict the same variable at the
        # same lead from the same window length -- otherwise the accumulated
        # truth array belongs to a different set of samples than the
        # predictions being scored against it.
        this_sig = (targs["target"], targs["lookback"], targs["lead"])
        if signature is None:
            signature = this_sig
        elif this_sig != signature:
            print(f"[predict] SKIPPING {name}: "
                  f"target/lookback/lead {this_sig} does not match "
                  f"{signature} of the models already loaded. "
                  f"Run predict.py once per configuration.")
            continue

        ds = WaveWindowDataset(
            cache=cache, split="test", target=targs["target"],
            channel_set=targs["channel_set"], lookback=targs["lookback"],
            lead=targs["lead"], stride=args.stride,
        )
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=(device == "cuda"))

        # Checkpoints written before ConvLSTM took capacity arguments have no
        # usable base_channels for it; those weights were built with the old
        # hard-coded (64, 64), which is what base_channels=32 now reproduces.
        caps = dict(base_channels=targs.get("base_channels", 32),
                    depth=targs.get("depth", 3))
        if targs["model"] == "convlstm" and "convlstm_capacity" not in targs:
            caps = dict(base_channels=32, depth=2)
        model = build(targs["model"], in_channels=ds.n_channels,
                      lookback=targs["lookback"], **caps).to(device)
        model.load_state_dict(ckpt["model"])

        print(f"[predict] {name}: {len(ds):,} test samples "
              f"(epoch {ckpt['epoch']})")
        predictions[name] = run_model(model, loader, device, cache,
                                      targs["target"], amp_dtype)

        if truth is None:
            truth, valid, persist, t_index = collect_truth(loader)
            predictions["persistence"] = persist

            # Per-cell mean over valid samples. Land cells have zero valid
            # samples, so divide explicitly rather than via nanmean, which
            # warns on the empty slices.
            counts = valid.sum(axis=0, keepdims=True).astype(np.float64)
            sums = np.where(valid, truth, 0).sum(axis=0, keepdims=True)
            mean_field = np.divide(sums, counts, out=np.zeros_like(sums),
                                   where=counts > 0)
            predictions["climatology"] = np.repeat(
                mean_field.astype(np.float32), len(truth), axis=0)

    if truth is None:
        print("no checkpoints loaded")
        return 1

    # Report against the checkpoints' own configuration, not the CLI defaults,
    # so the labels cannot disagree with what was actually evaluated.
    target, _, lead = signature

    df = evaluate(predictions, truth, valid)
    print(format_report(df, target, lead))

    # Merge into the table rather than replacing it. Scoring a subset of the
    # models used to overwrite the file with only those rows, silently
    # discarding every other model's results -- which broke the notebook three
    # times. Rows for the models just evaluated are refreshed; everything else
    # is left alone.
    dest = TABLES / f"test_metrics_{target}_h{lead}.csv"
    if dest.exists():
        old = pd.read_csv(dest)
        kept = old[~old["model"].isin(df["model"].unique())]
        if len(kept):
            print(f"[predict] keeping {kept['model'].nunique()} model(s) "
                  f"already in {dest.name}: "
                  f"{', '.join(sorted(kept['model'].unique()))}")
        df = pd.concat([kept, df], ignore_index=True)
    df.to_csv(dest, index=False)
    print(f"[predict] -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
