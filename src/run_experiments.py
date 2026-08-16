"""Run the experiment matrix unattended, in priority order.

Phase 1 completes the +1 h comparison table that the reproduction needs: the
same target and split, with the architecture and the input channel set varied
one at a time. Phase 2 moves to the lead times where the persistence baseline
actually degrades, which is where a learned model has something to prove.

Batch size and schedule are held fixed across every run so the table compares
architectures rather than tuning effort.

Runs are skipped if their checkpoint already exists, so the script can be
interrupted and restarted.

    python run_experiments.py              # everything outstanding
    python run_experiments.py --phase 1
    python run_experiments.py --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from train import CHECKPOINTS, run_name

SRC = Path(__file__).parent

COMMON = dict(target="swh", stride=3, batch_size=32, epochs=60,
              num_workers=4, lookback=48)

# (phase, model, channel_set, lead, note)
MATRIX = [
    (1, "cnn3d", "full", 1,
     "architecture ablation: no skip connections"),
    (1, "convlstm", "full", 1,
     "architecture ablation: recurrent instead of 3D convolution"),
    (1, "unet3d", "paper_swh", 1,
     "the paper's own channel set, which omits SWH from the inputs"),
    (2, "unet3d", "full", 6, "lead-time curve"),
    (2, "unet3d", "full", 24, "the regime with real headroom"),
    (2, "unet3d", "full", 48, "persistence has decayed to climatology by here"),
]


class Args:
    """Mirror of train.py's namespace, enough for run_name()."""

    def __init__(self, model, channel_set, lead):
        self.model = model
        self.channel_set = channel_set
        self.lead = lead
        self.target = COMMON["target"]
        self.lookback = COMMON["lookback"]
        self.extreme_weight = 1.0
        self.tag = ""


def build_command(model: str, channel_set: str, lead: int) -> list[str]:
    return [
        sys.executable, "-u", "train.py",
        "--model", model,
        "--target", COMMON["target"],
        "--channel-set", channel_set,
        "--lead", str(lead),
        "--lookback", str(COMMON["lookback"]),
        "--stride", str(COMMON["stride"]),
        "--batch-size", str(COMMON["batch_size"]),
        "--epochs", str(COMMON["epochs"]),
        "--num-workers", str(COMMON["num_workers"]),
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, choices=[1, 2], default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    jobs = [j for j in MATRIX if args.phase is None or j[0] == args.phase]

    pending, done = [], []
    for phase, model, channel_set, lead, note in jobs:
        name = run_name(Args(model, channel_set, lead))
        (done if (CHECKPOINTS / f"{name}.pt").exists() else pending).append(
            (phase, model, channel_set, lead, note, name))

    print(f"[runner] {len(done)} already trained, {len(pending)} to run\n")
    for _, _, _, _, _, name in done:
        print(f"  [skip] {name}")
    for phase, model, cs, lead, note, name in pending:
        print(f"  [ run] P{phase} {name:<34} {note}")
    print()

    if args.dry_run or not pending:
        return 0

    t_start = time.time()
    failures = []
    for i, (phase, model, cs, lead, note, name) in enumerate(pending, 1):
        print("=" * 78)
        print(f"[runner] {i}/{len(pending)}  {name}")
        print(f"[runner] {note}")
        print("=" * 78, flush=True)

        t0 = time.time()
        result = subprocess.run(build_command(model, cs, lead), cwd=SRC)
        mins = (time.time() - t0) / 60

        if result.returncode == 0:
            print(f"[runner] {name} finished in {mins:.0f} min\n", flush=True)
        else:
            failures.append(name)
            print(f"[runner] {name} FAILED (exit {result.returncode}) after "
                  f"{mins:.0f} min\n", flush=True)

    print("=" * 78)
    print(f"[runner] {len(pending) - len(failures)}/{len(pending)} runs "
          f"succeeded in {(time.time() - t_start) / 60:.0f} min")
    if failures:
        print(f"[runner] failed: {failures}")
    print("=" * 78)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
