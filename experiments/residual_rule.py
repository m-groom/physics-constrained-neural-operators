"""C3 of issue #52: is the E0 residual floor a time-discretisation artefact?

Appendix A reports the ground truth's own relative residual on the published 64^2 PINO
Kolmogorov data as 3.28, under the trapezoidal ("integral") time rule that training used.
This script re-measures the same truth residual under the central-difference ("fdm") rule,
on both forms of that data:

  E0, vorticity form: relative L2 of the vorticity residual against the forcing term --
      dt * f under the integral rule, f under the fdm rule -- scored at 256^2 by the
      4x Fourier refinement the campaign used.
  E1, velocity form: RMS of the continuity and momentum residuals over the matching
      ground-truth scale, exactly as utils.criterion normalises them.

Both are read straight from the npz in physical units; the loader's normalisation is a
model-side concern and the residual is not.

CPU only. Usage:
    python residual_rule.py [--n_trajectories 200] [--out residual_rule.csv]
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.criterion import (
    PINO_loss3d,
    PINO_loss3d_vel,
    ResidualScales,
    build_forcing_for_data,
    physics_from_config,
    residual_scales_from_truth,
)

CONFIGS = {
    "E0": os.path.join(os.path.dirname(__file__), "configs", "e0", "PINO_vort.yaml"),
    "E1": os.path.join(os.path.dirname(__file__), "configs", "e1_noise", "FNO.yaml"),
}


def truth_trajectories(path, split, n_trajectories, steps=64):
    """Ground-truth rollouts in physical units, shape ``(n, C, S, S, steps + 1)``.

    The npz stores one-step pairs; trajectory ``r`` occupies rows ``steps * r`` onwards, so
    its ``steps + 1`` levels are that block's first input followed by every target.
    """
    with np.load(path) as handle:
        inputs = handle[f"X_{split}"]
        targets = handle[f"y_{split}"]
        out = []
        for r in range(n_trajectories):
            block = slice(steps * r, steps * (r + 1))
            levels = np.concatenate([inputs[block][:1], targets[block]], axis=0)
            out.append(levels)
    # (n, steps + 1, C, S, S) -> (n, C, S, S, steps + 1)
    return torch.from_numpy(np.stack(out)).permute(0, 2, 3, 4, 1).contiguous()


def windows(trajectory, width):
    """Every consecutive window of ``width`` time levels, as a batch."""
    return trajectory.unfold(-1, width, 1).permute(0, 4, 1, 2, 3, 5).flatten(0, 1)


def vorticity_residual(truth, physics, forcing, dt, time_method, chunk=4):
    """Relative L2 of the truth's vorticity residual against the forcing term."""
    width = 2 if time_method == "integral" else 3
    t_interval = dt * (width - 1)
    total, count = 0.0, 0
    for start in range(0, truth.shape[0], chunk):
        batch = windows(truth[start : start + chunk], width)
        w = batch[:, 0]
        _, loss_f = PINO_loss3d(
            w,
            w[..., 0],
            forcing,
            nu=physics.nu,
            alpha=physics.alpha,
            t_interval=t_interval,
            domain_length=physics.domain_length,
            time_method=time_method,
            upsample=physics.residual_upsample,
        )
        total += float(loss_f) * batch.shape[0]
        count += batch.shape[0]
    return {"residual_rel": total / count}


def velocity_residuals(truth, physics, forcing, dt, time_method, chunk=8):
    """Continuity and momentum residuals of the truth, over the truth's own scales.

    The scales are measured once over every trajectory, as the evaluation measures them
    over the test batch it scores, and under the same time rule as the residual.
    """
    width = 2 if time_method == "integral" else 3
    horizon = truth.shape[-1] - 1
    squares = np.zeros(3)
    for start in range(0, truth.shape[0], chunk):
        block = truth[start : start + chunk]
        scale = residual_scales_from_truth(block, physics, dt * horizon, time_method=time_method)
        squares += np.array([scale.cont, scale.momx, scale.momy]) ** 2 * block.shape[0]
    cont, momx, momy = np.sqrt(squares / truth.shape[0])
    scales = ResidualScales(cont=float(cont), momx=float(momx), momy=float(momy))
    totals, count = np.zeros(3), 0
    for start in range(0, truth.shape[0], chunk):
        batch = windows(truth[start : start + chunk], width)
        out = PINO_loss3d_vel(
            batch,
            batch[..., 0],
            forcing,
            nu=physics.nu,
            alpha=physics.alpha,
            t_interval=dt * (width - 1),
            domain_length=physics.domain_length,
            scales=scales,
            time_method=time_method,
        )
        totals += np.array([float(out[4]), float(out[5]), float(out[6])]) * batch.shape[0]
        count += batch.shape[0]
    cont_rel, momx_rel, momy_rel = totals / count
    return {
        "cont_rel": cont_rel,
        "momx_rel": momx_rel,
        "momy_rel": momy_rel,
        "scale_cont": scales.cont,
        "scale_momx": scales.momx,
        "scale_momy": scales.momy,
    }


def main():
    """Measure the truth residual under both time rules and write one CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n_trajectories", type=int, default=200)
    parser.add_argument(
        "--data_dir",
        default="DATA_ROOT/pino_kf",
        help="directory holding pino_kf_vorticity.npz and pino_kf_velocity.npz "
        "(default: DATA_ROOT/pino_kf)",
    )
    parser.add_argument(
        "--out",
        default="DATA_ROOT/residual_rule.csv",
        help="CSV path to write the truth-residual table to (default: DATA_ROOT/residual_rule.csv)",
    )
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    rows = []
    for case, filename, worker in (
        ("E0", "pino_kf_vorticity.npz", vorticity_residual),
        ("E1", "pino_kf_velocity.npz", velocity_residuals),
    ):
        with open(CONFIGS[case]) as handle:
            physics = physics_from_config(yaml.safe_load(handle)["data"])
        forcing = build_forcing_for_data(physics, (physics.nx, physics.ny))
        truth = truth_trajectories(f"{args.data_dir}/{filename}", "test", args.n_trajectories)
        print(f"{case}: truth {tuple(truth.shape)} from {filename}", flush=True)
        # 1/64 is the spacing every config declares; 0.01554 is the least-squares fit the
        # E1 campaign made against the momentum residual (see METADATA_CORRECTION.md).
        for dt_label, dt in (("1/64", 1.0 / 64.0), ("fitted", 0.01554)):
            for time_method in ("integral", "fdm"):
                numbers = worker(truth, physics, forcing, dt, time_method)
                rows.append(
                    {
                        "case": case,
                        "formulation": physics.formulation,
                        "dt_label": dt_label,
                        "dt": dt,
                        "time_method": time_method,
                        "n_trajectories": args.n_trajectories,
                        **{k: float(v) for k, v in numbers.items()},
                    }
                )
                print(rows[-1], flush=True)

    fields = list(dict.fromkeys(key for row in rows for key in row))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
