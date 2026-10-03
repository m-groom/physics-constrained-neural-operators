"""Fit the constant (climatological) subgrid energy source S from a coarse training split.

The filtered energy balance is ``dE/dt = -nu Z - 2 alpha E + P + Pi``, and Pi -- the net
nonlinear transfer across the coarse cut-off -- is the only term that does not close. This
fits the laziest closure that needs nothing but the coarse field, ``Pi(t) ~ S``, one
constant, which the energy projector then carries as
``output_constraint.energy_balance.subgrid_source`` (issue #83).

S is fitted so the CLOSED discrete residual has zero mean on the training pairs:

    h_n = E_{n+1} - E_n + dt/2 [ nu (Z_n + Z_{n+1}) + 2 alpha (E_n + E_{n+1})
                                 - (P_n + P_{n+1}) ]
    S   = mean_n(h_n) / dt

so that ``h_n - dt S`` has mean zero there. The discretisation is not re-derived: ``h`` comes
from :func:`test_operator_AR_2d.compute_energy_balance_per_step`, the same trapezoidal
balance and the same spectral enstrophy the projector enforces, so the fit and the layer
cannot drift apart. The fields are the ones the constraint layer actually sees --
:class:`NSLoader2D`'s split, taken back to physical units through the same normaliser
:meth:`models.fno.FNO2d.apply_output_constraint` inverts, including its 1e-8 guard.

Only the training split decides S. Every other split is reported so the fit can be checked
where it was not made: a closed residual that is near zero on validation as well is a
climatology, and one that is not is an artefact of the split.

Run from ``experiments/``:
    python fit_subgrid_source.py configs/e3_final/FNO_proj_cont_zeromean_energy.yaml
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_utils.datasets_dedalus import NSLoader2D
from test_operator_AR_2d import compute_energy_balance_per_step

from utils.criterion import build_forcing_for_data, physics_from_config

NORMALIZER_EPS = 1e-8  # the guard models.fno.apply_output_constraint denormalises with


def physical_pairs(data_config, split):
    """The split's (state_n, state_n+1) pairs in physical units.

    Args:
        data_config: The ``data`` mapping of the run configuration.
        split: ``train``, ``val`` or ``test``.

    Returns:
        torch.Tensor: Shape ``(N, H, W, C, 2)``, float64, the last axis being the pair.
    """
    loader = NSLoader2D(
        datapath=data_config["datapath"],
        state=split,
        train=False,
        normalizer_path=data_config.get("normalizer_path"),
        velocity_channels=data_config.get("velocity_channels", (0, 1)),
        filename=data_config.get("filename", "kolmogorov_dataset.npz"),
    )
    mean = loader.mean.permute(1, 2, 0)  # (C, 1, 1) -> (1, 1, C)
    std = loader.std.permute(1, 2, 0)

    def denormalise(field):
        return (field * (std + NORMALIZER_EPS) + mean).double()

    return torch.stack([denormalise(loader.X_data), denormalise(loader.y_data)], dim=-1)


def residual_terms(data_config, split, chunk=512):
    """The truth's un-closed balance residual and its ingredients, per pair of the split.

    Args:
        data_config: The ``data`` mapping of the run configuration.
        split: ``train``, ``val`` or ``test``.
        chunk: Pairs per call, so a large split does not need one allocation.

    Returns:
        numpy.ndarray: Shape ``(5, N)`` -- the residual h, the energy increment, and the
        energy, dissipation and injection of the first state of each pair.
    """
    physics = physics_from_config(data_config)
    pairs = physical_pairs(data_config, split)
    forcing = build_forcing_for_data(physics, (physics.nx, physics.ny)).double()
    out = []
    for start in range(0, pairs.shape[0], chunk):
        terms = compute_energy_balance_per_step(
            pairs[start : start + chunk],
            velocity_channels=list(data_config.get("velocity_channels", (0, 1))),
            forcing=forcing,
            viscosity=physics.nu,
            friction=physics.alpha,
            dt=physics.dt,
            domain_lengths=(physics.domain_length,) * 2,
        )
        out.append(
            np.stack(
                [
                    terms["balance_residual"][:, 0],
                    terms["energy"][:, 1] - terms["energy"][:, 0],
                    terms["energy"][:, 0],
                    terms["dissipation"][:, 0],
                    terms["injection"][:, 0],
                ]
            )
        )
    return np.concatenate(out, axis=1)


def main():
    """Fit S on the first split given and report the closed residual on every split."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="run configuration; only its data block is read")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="the first is fitted on, the rest are held out",
    )
    parser.add_argument("--out", default=None, help="optional json report")
    args = parser.parse_args()

    with open(args.config) as handle:
        data_config = yaml.safe_load(handle)["data"]
    dt = float(data_config["dt"])

    report = {"config": args.config, "dt": dt, "fitted_on": args.splits[0], "splits": {}}
    for split in args.splits:
        h, delta, energy, dissipation, injection = residual_terms(data_config, split)
        report["splits"][split] = {
            "n_pairs": int(h.size),
            "h_mean": float(h.mean()),
            "h_std": float(h.std()),
            "h_sem": float(h.std() / np.sqrt(h.size)),
            "S_bar": float(h.mean() / dt),
            "delta_energy_mean": float(delta.mean()),
            "delta_energy_abs_mean": float(np.abs(delta).mean()),
            "energy_mean": float(energy.mean()),
            "dissipation_mean": float(dissipation.mean()),
            "injection_mean": float(injection.mean()),
        }

    fit = report["splits"][args.splits[0]]["S_bar"]
    report["S_bar_fitted"] = fit
    report["S_bar_times_dt"] = fit * dt
    for split, row in report["splits"].items():
        row["closed_h_mean"] = row["h_mean"] - dt * fit
        # The residual measured against the truth's own energy change: how much of the
        # budget the unclosed balance mis-states, which is what makes E3 and E1 differ.
        row["h_over_delta_energy"] = row["h_mean"] / (row["delta_energy_abs_mean"] + 1e-300)
        print(
            f"{split:>10}: n={row['n_pairs']:6d}  h_mean={row['h_mean']:.6e} "
            f"(+/- {row['h_sem']:.2e} sem)  S={row['S_bar']:.6e}  "
            f"closed h_mean={row['closed_h_mean']:.6e}  "
            f"|dE|_mean={row['delta_energy_abs_mean']:.6e}  "
            f"h/|dE|={row['h_over_delta_energy']:.4f}"
        )
    print(f"\nfitted S_bar = {fit:.6e} per unit time; S_bar*dt = {fit * dt:.6e} per step")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
