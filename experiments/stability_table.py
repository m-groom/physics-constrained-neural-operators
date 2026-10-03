"""C2 of issue #52: the long free-running stability table, one row per arm.

Reads the 2,000-step energy traces `rollout_stability.py` wrote for every (arm, seed) and
reports, per arm and pooled over seeds, the fraction of trajectories that had never blown
up by each checkpoint step and the blow-up step of the median trajectory.

Blow-up is the kinetic energy exceeding ten times its initial value, or going non-finite;
`blowup_steps` in the npz already records the 1-based first crossing, or -1 for a
trajectory that never crossed. The median follows the repository's convention
(`test_operator_AR_2d.blowup_quantile`): nearest-rank, with a surviving trajectory counted
as infinite, so an arm whose median trajectory survives reports "never" rather than an
interpolated number over the failures alone.

Usage:
    python stability_table.py <free_2000 dir> --label E1 --out <csv>
"""

import argparse
import csv
from pathlib import Path

import numpy as np

ARMS = [
    "FNO",
    "FNO_proj_cont",
    "FNO_proj_cont_zeromean",
    "FNO_proj_cont_zeromean_energy",
    # E3 only (#83); an arm with no traces in the directory is skipped, so naming it here
    # is harmless on E1.
    "FNO_proj_cont_zeromean_energy_closed",
    "PINO",
    "PINO_cont",
]
CHECKPOINTS = (64, 128, 224, 448, 1000, 2000)


def pooled_blowup_steps(directory, arm, seeds=(1, 2, 3, 4, 5), prefix=None):
    """Every seed's blow-up steps for one arm, concatenated, with never-blown as infinity.

    ``prefix`` keeps only the first N trajectories of each seed. #51's E1 TTO arm adapts
    twenty trajectories per seed, which are the first twenty of this same ordered set, so
    the twenty-trajectory prefix is what puts the TTO and non-TTO E1 columns side by side.
    """
    steps = []
    for seed in seeds:
        path = Path(directory) / f"{arm}_seed{seed}.npz"
        if not path.exists():
            continue
        with np.load(path) as archive:
            raw = archive["blowup_steps"]
        if prefix is not None:
            raw = raw[:prefix]
        steps.append(np.where(raw < 0, np.inf, raw.astype(float)))
    return np.concatenate(steps) if steps else np.empty(0), len(steps)


def summarise(steps):
    """Surviving fraction at each checkpoint step, and the median blow-up step."""
    row = {f"frac_bounded_{step}": float((steps > step).mean()) for step in CHECKPOINTS}
    row["median_blowup_step"] = float(np.quantile(steps, 0.5, method="lower"))
    row["n_trajectories"] = int(steps.size)
    return row


def main():
    """Write one CSV for one experiment and print it as a markdown table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--prefix",
        type=int,
        default=None,
        help="keep only the first N trajectories of each seed (#51's E1 comparison)",
    )
    args = parser.parse_args()

    rows = []
    for arm in ARMS:
        steps, n_seeds = pooled_blowup_steps(args.directory, arm, prefix=args.prefix)
        if steps.size == 0:
            print(f"{arm}: no traces yet")
            continue
        rows.append({"model": arm, "seeds": n_seeds, **summarise(steps)})

    if not rows:
        return
    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    header = " | ".join(str(step) for step in CHECKPOINTS)
    print(f"\n### {args.label}\n")
    print(f"| arm | seeds | trajectories | {header} | median blow-up |")
    print("| --- | --- | --- | " + " | ".join(["---"] * len(CHECKPOINTS)) + " | --- |")
    for row in rows:
        fractions = " | ".join(f"{row[f'frac_bounded_{step}']:.3f}" for step in CHECKPOINTS)
        median = row["median_blowup_step"]
        median_text = "never" if not np.isfinite(median) else f"{median:.0f}"
        print(
            f"| {row['model']} | {row['seeds']} | {row['n_trajectories']} | "
            f"{fractions} | {median_text} |"
        )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
