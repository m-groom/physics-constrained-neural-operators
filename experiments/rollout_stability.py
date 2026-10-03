"""Free-running rollout stability: how far past the truth horizon a checkpoint survives.

A rollout evaluated over the ground truth's own 63 or 64 steps can only report whether the
model failed inside that window. It cannot report how close to failing it was, and on this
data the two are very different questions: every FNO measured for issue #42 eventually blows
up, and the models differ only in when (step 68 for the E1 baseline, step 126 for the old
128 x 128 runs, step 608-768 for the energy-balance projector).

This module rolls a checkpoint on for as long as asked, tracking only kinetic energy and its
spectrum, so no ground truth is needed past step 64. The reported number is the step at which
each trajectory's energy first exceeds a multiple of its initial energy.

Usage:
    python rollout_stability.py --config_path <run.yaml> --seed 1 --steps 400
"""

import argparse

import numpy as np
import torch

# The loader, the model and the evaluation helpers are imported inside main(), because they
# resolve only when this file is run from within experiments/ (as every script here is), and
# the two pure functions below must stay importable from the test suite.


def shell_energy(field, side):
    """Shell-summed kinetic energy of a batch of fields.

    Args:
        field: Tensor of shape `(B, S, S, C)`; the first `min(2, C)` channels are the state
            whose energy is summed, so a one-channel vorticity state gives shell enstrophy.
        side: The grid side `S`, used for the shell binning.

    Returns:
        Tensor of shape `(B, S // 2)`, the energy in shells `0 .. S // 2 - 1`.
    """
    channels = min(2, field.shape[-1])
    spectrum = torch.fft.rfft2(field[..., :channels].permute(0, 3, 1, 2), dim=(-2, -1))
    per_mode = (spectrum.abs() ** 2).sum(1)
    ix = torch.fft.fftfreq(side, d=1.0 / side, device=field.device)
    iy = torch.fft.rfftfreq(side, d=1.0 / side, device=field.device)
    shell = torch.sqrt(ix[:, None] ** 2 + iy[None, :] ** 2).floor().long()
    # rfft symmetry: interior ky columns stand for two modes, ky = 0 and Nyquist for one.
    weight = torch.full_like(per_mode[0], 2.0)
    weight[:, 0] = 1.0
    weight[:, -1] = 1.0
    n_bins = side // 2
    out = torch.zeros(per_mode.shape[0], n_bins, dtype=per_mode.dtype, device=field.device)
    out.index_add_(
        1,
        shell.clamp(max=n_bins - 1).reshape(-1),
        (per_mode * weight).reshape(per_mode.shape[0], -1),
    )
    return out


def blowup_steps(energy_ratio, threshold=10.0):
    """First step at which each trajectory's energy leaves the bounded range.

    Args:
        energy_ratio: Array of shape `(T, B)`, energy at each step over energy at step 0.
        threshold: The multiple of the initial energy that counts as a blow-up.

    Returns:
        List of length `B`; the 1-based step for each trajectory that blew up, or `-1` for one
        that did not. A non-finite entry counts as a blow-up at the step it appears.
    """
    ratio = np.asarray(energy_ratio)
    gone = (ratio > threshold) | ~np.isfinite(ratio)
    return [
        int(np.argmax(gone[:, b])) + 1 if gone[:, b].any() else -1 for b in range(gone.shape[1])
    ]


def free_rollout(
    model, initial, grid, steps, side, band_start, use_residual=False, constrain=False
):
    """Roll a model on with no ground truth, recording total and high-band energy.

    Args:
        model: The trained operator.
        initial: Initial condition, `(B, S, S, C)`, in normalised units.
        grid: Coordinate grid the model is fed alongside the state.
        steps: How many autoregressive steps to take.
        side: The grid side `S`.
        use_residual: Whether the model predicts an increment rather than the next state.
        constrain: Whether to apply the model's output constraint at every step.
        band_start: First shell of the high band, `|k| >= band_start`. Callers pass the
            model's own spectral-mode cutoff: the band no spectral layer can write into and
            no one-step loss term weighs, which is where the instability lives.

    Returns:
        Array of shape `(steps, B, 2)`: energy over initial energy, and high-band energy over
        initial energy. A trajectory whose state goes non-finite is frozen and recorded as
        infinite from that step on, so one blow-up never truncates the rest of the batch.
    """
    from test_operator_AR_2d import align_prediction

    prev = initial
    grid_batch = grid.unsqueeze(0).expand(prev.shape[0], -1, -1, -1)
    initial_total = shell_energy(prev, side).sum(1)
    top = int(band_start)
    record = []
    # A trajectory that has gone non-finite is unrecoverable and its neighbours in the batch
    # are not: over 2,000 steps almost every arm loses one, so it is frozen at the initial
    # condition (a finite state, cheaper than reshaping the batch) and recorded as infinite,
    # leaving the survivors to roll the full horizon.
    gone = torch.zeros(prev.shape[0], dtype=torch.bool, device=prev.device)
    with torch.no_grad():
        for _ in range(steps):
            pred = align_prediction(model(torch.cat((prev, grid_batch), dim=-1)), prev)
            if use_residual:
                pred = pred + prev
            if constrain:
                pred = model.apply_output_constraint(pred, x_old=prev)
            gone |= ~torch.isfinite(pred).flatten(1).all(1)
            pred = torch.where(gone.reshape(-1, *([1] * (pred.dim() - 1))), initial, pred)
            prev = pred
            shells = shell_energy(pred, side)
            step_record = torch.stack(
                [shells.sum(1) / initial_total, shells[:, top:].sum(1) / initial_total], dim=-1
            )
            step_record[gone] = float("inf")
            record.append(step_record.cpu().numpy())
    return np.stack(record)


def main():
    """Run the free rollout for one (config, seed) and report its blow-up steps."""
    import os
    import sys

    import yaml
    from torch.utils.data import DataLoader, Subset

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data_utils.datasets_dedalus import NSLoader2D
    from test_operator_AR_2d import resolve_checkpoint_path

    from models.fno import FNO2d
    from utils.criterion import build_forcing_for_data, physics_from_config
    from utils.utilities import torch2dgrid_2d

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n_trajectories", type=int, default=20)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--threshold", type=float, default=10.0)
    parser.add_argument(
        "--band_cutoff",
        type=int,
        default=None,
        help=(
            "First shell of the high band, overriding the model's own modes1. A 64x64 grid "
            "has shells 0..31, so a model with modes1 = 32 reports an identically zero band "
            "unless this is given, and two models with different cutoffs are otherwise "
            "measured over different bands"
        ),
    )
    parser.add_argument("--checkpoint", default="best", choices=("best", "last"))
    parser.add_argument("--save", default=None, help="npz path for the (steps, B, 2) energy trace")
    args = parser.parse_args()

    with open(args.config_path) as handle:
        config = yaml.safe_load(handle)
    data_config, model_config = config["data"], config["model"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_set = NSLoader2D(
        datapath=data_config["datapath"],
        state="test",
        train=False,
        normalizer_path=data_config.get("normalizer_path"),
        velocity_channels=data_config.get("velocity_channels", (0, 1)),
        filename=data_config.get("filename", "kolmogorov_dataset.npz"),
    )
    test_set.transform_rollout(T=data_config["nt"])
    side = test_set.S[0]
    grid = torch2dgrid_2d(
        side, test_set.S[1], form=data_config["grid_form"], device=device, dtype=torch.float32
    )

    constraint = model_config.get("output_constraint")
    constrain = bool(constraint and constraint.get("enabled"))
    physics = physics_from_config(data_config) if constrain else None
    model = FNO2d(
        in_dim=model_config.get("in_dim", 3),
        out_dim=model_config.get("out_dim", 1),
        modes1=model_config["modes1"],
        modes2=model_config["modes2"],
        fc_dim=model_config["fc_dim"],
        layers=model_config["layers"],
        act=model_config["act"],
        output_constraint=constraint,
        physics=physics,
    ).to(device)
    if constrain:
        model.set_output_normalizer(test_set.mean, test_set.std)
        if constraint.get("energy_balance", {}).get("enabled"):
            model.set_energy_forcing(
                build_forcing_for_data(physics, test_set.S, device=device).squeeze(-1)
            )
    path = resolve_checkpoint_path(
        config["train"]["save_dir"], config["train"]["save_name"], args.seed, which=args.checkpoint
    )
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False)["model"])
    model.eval()

    loader = DataLoader(
        Subset(test_set, range(min(args.n_trajectories, len(test_set)))),
        batch_size=args.n_trajectories,
    )
    sequence, _ = next(iter(loader))
    trace = free_rollout(
        model,
        sequence.to(device)[..., 0, :],
        grid,
        args.steps,
        side,
        args.band_cutoff if args.band_cutoff is not None else min(model_config["modes1"]),
        use_residual=model_config.get("residual", False),
        constrain=constrain,
    )
    steps = blowup_steps(trace[..., 0], args.threshold)
    blown = sorted(step for step in steps if step > 0)
    print(f"checkpoint {path}")
    print(
        f"blow-up ({args.threshold}x initial energy): {len(blown)}/{len(steps)} trajectories "
        f"within {trace.shape[0]} steps"
    )
    print(f"first blow-up: {blown[0] if blown else 'none'}; all: {blown}")
    if args.save:
        np.savez(
            args.save,
            trace=trace,
            blowup_steps=np.array(steps),
            band_start=np.array(
                args.band_cutoff if args.band_cutoff is not None else min(model_config["modes1"])
            ),
        )


if __name__ == "__main__":
    main()
