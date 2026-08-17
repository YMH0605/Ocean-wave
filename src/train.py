"""Training loop for the wave forecast models.

Differences from the paper's setup, all deliberate:

  * Loss is masked to ocean points (see dataset.masked_mse).
  * Train/val split is by year, not random over samples.
  * A `--stride` option subsamples the sliding windows. Consecutive hourly
    windows share 47 of their 48 frames, so stride 1 mostly buys redundant
    gradient steps; stride 3 cuts epoch time threefold at negligible cost.
  * `--extreme-weight` optionally upweights samples whose target field is in
    the tail of the training distribution. This is how the storm focus is
    retained without throwing away 97% of the record.
  * Validation reports MAE in metres/seconds and the skill score against
    persistence, so a run that is not beating the trivial baseline is visible
    immediately rather than after the fact.

Usage
-----
    python train.py --model unet3d --target swh --lead 1
    python train.py --model unet3d --target swh --lead 24 --extreme-weight 3
    python train.py --model cnn3d --channel-set paper_swh   # paper-faithful
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from config import OUTPUTS
from dataset import ERA5Cache, WaveWindowDataset, masked_mse
from models import Persistence, build, count_parameters

CHECKPOINTS = OUTPUTS / "checkpoints"
LOGS = OUTPUTS / "logs"
CHECKPOINTS.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unet3d",
                    choices=["unet3d", "cnn3d", "convlstm"])
    ap.add_argument("--target", default="swh", choices=["swh", "mp2"])
    ap.add_argument("--channel-set", default="full",
                    choices=["full", "paper_swh", "paper_mp2"])
    ap.add_argument("--lookback", type=int, default=48)
    ap.add_argument("--lead", type=int, default=1)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base-channels", type=int, default=32)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--extreme-weight", type=float, default=1.0,
                    help="sampling weight for tail samples; 1.0 disables it")
    ap.add_argument("--extreme-percentile", type=float, default=95.0)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "off"])
    ap.add_argument("--grad-clip", type=float, default=1.0,
                    help="max gradient norm; 0 disables clipping")
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--tag", default="")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing checkpoint of the same name")
    return ap.parse_args()


def run_name(args) -> str:
    parts = [args.model, args.target, args.channel_set,
             f"lb{args.lookback}", f"h{args.lead}"]
    if args.extreme_weight != 1.0:
        parts.append(f"ew{args.extreme_weight:g}")
    if args.tag:
        parts.append(args.tag)
    return "_".join(parts)


def build_sampler(ds: WaveWindowDataset, args):
    """Upweight samples whose target field reaches into the tail.

    Uses the domain-mean of the target at the forecast time as a cheap proxy
    for "is this a storm", read straight from the cache rather than by
    iterating the Dataset.
    """
    if args.extreme_weight == 1.0:
        return None

    idx = ds.indices + args.lead
    ocean = ds.cache.ocean
    field = ds.cache.data[idx, ds.target_idx]
    score = np.nanmean(np.where(ocean, field, np.nan), axis=(1, 2))

    threshold = np.nanpercentile(score, args.extreme_percentile)
    weights = np.where(score >= threshold, args.extreme_weight, 1.0)
    n_tail = int((score >= threshold).sum())
    print(f"[sampler] {n_tail:,}/{len(score):,} samples above "
          f"P{args.extreme_percentile:g} ({threshold:.2f}), "
          f"weight {args.extreme_weight:g}")
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                 num_samples=len(weights), replacement=True)


@torch.no_grad()
def validate(model, loader, device, cache, target, amp_dtype):
    """Return val loss plus MAE and persistence skill in physical units."""
    model.eval()
    stats = cache.stats[target]
    scale = stats["max"] - stats["min"]

    tot_loss = tot_ae = tot_se = tot_se_persist = 0.0
    tot_n = 0

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True) 
        valid = batch["valid"].to(device, non_blocking=True).float()
        ocean = valid

        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            pred = model(x, ocean)
        pred = pred.float()

        tot_loss += masked_mse(pred, y, valid).item() * valid.sum().item()

        # Back to physical units for the reported numbers.
        pred_phys = pred * scale + stats["min"]
        true_phys = batch["y_phys"].to(device)
        persist = batch["persist"].to(device)

        err = (pred_phys - true_phys) * valid
        tot_ae += err.abs().sum().item()
        tot_se += (err ** 2).sum().item()
        tot_se_persist += (((persist - true_phys) * valid) ** 2).sum().item()
        tot_n += valid.sum().item()

    mse = tot_se / tot_n
    return {
        "val_loss": tot_loss / tot_n,
        "val_mae": tot_ae / tot_n,
        "val_rmse": float(np.sqrt(mse)),
        "skill_vs_persistence": 1.0 - mse / (tot_se_persist / tot_n),
    }


def main() -> int:
    args = parse_args()
    name = run_name(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                 "off": None}[args.amp]
    if device != "cuda":
        amp_dtype = None

    # Refuse to clobber finished work. Every argument has a default, so a bare
    # `python train.py` resolves to unet3d/swh/full/lb48/h1 -- the name of a
    # model that may already represent hours of training. Interrupting such a
    # run leaves a one-epoch checkpoint in its place, which is exactly how the
    # first fully trained U-Net was lost.
    ckpt_path = CHECKPOINTS / f"{name}.pt"
    if ckpt_path.exists() and not args.force:
        import torch as _t
        old = _t.load(ckpt_path, map_location="cpu", weights_only=False)
        print(f"[train] {ckpt_path.name} already exists "
              f"(epoch {old.get('epoch')}, "
              f"val MAE {old.get('metrics', {}).get('val_mae', float('nan')):.4f}).\n"
              f"        Training would overwrite it. Either give this run its own\n"
              f"        name with --tag, or pass --force to replace the file.")
        return 1

    print(f"[train] run={name}  device={device}  amp={args.amp}")

    cache = ERA5Cache(load_to_ram=False)
    common = dict(cache=cache, target=args.target,
                  channel_set=args.channel_set, lookback=args.lookback,
                  lead=args.lead)

    train_ds = WaveWindowDataset(split="train", stride=args.stride, **common)
    val_ds = WaveWindowDataset(split="val", stride=max(args.stride, 6), **common)
    print(f"[train] train: {train_ds.describe()}")
    print(f"[train] val  : {val_ds.describe()}")

    sampler = build_sampler(train_ds, args)
    loader_kw = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                     pin_memory=(device == "cuda"),
                     persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(train_ds, shuffle=sampler is None,
                              sampler=sampler, drop_last=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)

    # Every architecture now takes the same capacity knobs, so a comparison can
    # be made at matched parameter counts rather than at whatever each model's
    # defaults happened to be.
    model = build(args.model, in_channels=train_ds.n_channels,
                  lookback=args.lookback, base_channels=args.base_channels,
                  depth=args.depth).to(device)
    print(f"[train] {args.model}: {count_parameters(model):,} parameters")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                           betas=(0.9, 0.999), eps=1e-8)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=3)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp == "fp16"))

    history, best, bad_epochs = [], np.inf, 0
    global_step = 0
    max_grad_seen = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running, seen = 0.0, 0
        max_grad_seen = 0.0

        for step, batch in enumerate(train_loader, 1):
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            valid = batch["valid"].to(device, non_blocking=True).float()

            # Linear warmup. The normalised target is small (typical SWH maps
            # to ~0.12 of the [0,1] range, so MSE sits around 1e-5) and early
            # Adam steps at full learning rate were enough to destabilise the
            # run; the first version of this loop diverged at epoch 9 with an
            # 800x loss spike it never fully recovered from.
            global_step += 1
            if args.warmup_steps and global_step <= args.warmup_steps:
                for g in opt.param_groups:
                    g["lr"] = args.lr * global_step / args.warmup_steps

            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dtype,
                                enabled=amp_dtype is not None):
                pred = model(x, valid)
                loss = masked_mse(pred.float(), y, valid)

            if args.amp == "fp16":
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
            else:
                loss.backward()

            if args.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip)
                max_grad_seen = max(max_grad_seen, float(grad_norm))

            if args.amp == "fp16":
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()

            running += loss.item() * x.size(0)
            seen += x.size(0)
            if step % 200 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} "
                      f"loss {running / seen:.6f} "
                      f"max|grad| {max_grad_seen:.2f}", flush=True)

        metrics = validate(model, val_loader, device, cache, args.target,
                           amp_dtype)
        metrics.update(epoch=epoch, train_loss=running / seen,
                       lr=opt.param_groups[0]["lr"],
                       max_grad=max_grad_seen,
                       minutes=(time.time() - t0) / 60)
        history.append(metrics)
        sched.step(metrics["val_loss"])

        unit = "m" if args.target == "swh" else "s"
        print(f"[epoch {epoch:3d}] train {metrics['train_loss']:.6f} | "
              f"val {metrics['val_loss']:.6f} | "
              f"MAE {metrics['val_mae']:.3f}{unit} | "
              f"skill vs persistence {metrics['skill_vs_persistence']:+.3f} | "
              f"lr {metrics['lr']:.1e} | max|g| {max_grad_seen:.1f} | "
              f"{metrics['minutes']:.1f} min", flush=True)

        if metrics["val_loss"] < best - 1e-6:
            best, bad_epochs = metrics["val_loss"], 0
            # Record the normalisation the weights were fitted against. A
            # checkpoint used with a cache built from different years would be
            # denormalised with the wrong min/max and produce silently wrong
            # values in metres; predict.py checks this and refuses.
            torch.save({"model": model.state_dict(), "args": vars(args),
                        "epoch": epoch, "metrics": metrics,
                        "convlstm_capacity": True,
                        "target_stats": cache.stats[args.target]}, ckpt_path)
            print(f"           saved {ckpt_path.name}")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"[train] early stop after {args.patience} epochs "
                      "without improvement")
                break

    (LOGS / f"{name}.json").write_text(json.dumps(history, indent=2))
    print(f"[train] best val_loss {best:.5f}; history -> "
          f"{LOGS / (name + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
