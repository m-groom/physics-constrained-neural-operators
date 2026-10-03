"""Autoregressive evaluation for 2D Navier-Stokes operators.

Runs a trained model in autoregressive rollout on the test set, computing:
  - Per-step relative L2 errors
  - Per-step PDE residuals (continuity, x-momentum, y-momentum)
  - Energy and enstrophy spectra at selected snapshots
  - Aggregate spectral metrics (MEAPE, MELR)

Optionally, this script can perform PINO-style test-time optimization (TTO)
by adapting the pretrained operator weights on each test trajectory using the
rollout PDE residual objective before running a second evaluation pass.
"""

import json
import math
import os
import sys
import time
from argparse import ArgumentParser
from collections import Counter

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd
from aggregate_seeds import BOUNDED_L2
from data_utils.datasets_dedalus import NSLoader2D
from rollout_stability import blowup_steps, free_rollout, shell_energy
from run_protocol import seed_everything

from models.fno import FNO2d
from utils.compute_physical_statistics import compute_spectra
from utils.criterion import (
    LpLoss,
    MeanEnergyAbsolutePercentageError,
    MeanEnergyLogRatioError,
    PINO_loss3d_physics,
    build_forcing_for_data,
    carries_velocity,
    check_model_channels,
    max_abs_divergence_2d_velocity,
    physics_from_config,
    residual_has_scale,
    residual_scales_from_truth,
    velocity_from_vorticity,
)
from utils.utilities import torch2dgrid_2d

# ── Spectra snapshot configuration ──────────────────────────────────
# 0-based rollout step indices at which to save energy/enstrophy spectra.
# Set to None to default to all predicted rollout steps.
SPECTRA_TIME_INDICES = None

NORM_EPS = 1e-8
LOSS_GUARD_EPS = 1e-12


def denormalise(x, mean, std, device):
    """Map normalised tensor back to physical space.

    Args:
        x:    (N, H, W, C, ...) normalised tensor
        mean: (C, 1, 1) global per-channel mean
        std:  (C, 1, 1) global per-channel scale

    Returns:
        Tensor of the same shape in physical units.
    """
    # (C, 1, 1) -> (1, 1, 1, C, 1) for broadcasting with (N, H, W, C, T)
    m = mean.permute(1, 2, 0).to(device)[None, ..., None]
    s = std.permute(1, 2, 0).to(device)[None, ..., None]
    return x * (s + NORM_EPS) + m


def append_file_suffix(stem, suffix):
    """Append a filename suffix if provided."""
    return f"{stem}{suffix}" if suffix else stem


def evaluation_variant_suffix(split, rollout_steps, band_cutoff=None):
    """Filename suffix that keeps one evaluation from overwriting another.

    Every artefact is named after the experiment and the seed alone, so a second
    evaluation of the same checkpoint -- another split, a longer horizon, or a
    different high band -- lands on the first one's files. The suffix names what
    differs, and the default evaluation keeps its bare name so existing campaigns
    are unchanged.

    Args:
        split: The dataset split being evaluated.
        rollout_steps: The horizon given on the command line, or None to take
            the horizon from ``config['data']['nt']``.
        band_cutoff: The high-band cutoff given on the command line, or None to
            take it from the model's own ``modes1``. Two evaluations that measure
            different bands report a differently defined diagnostic under the same
            column name, so the cutoff has to name the file when it is chosen.

    Returns:
        str: The suffix, empty for the default evaluation.
    """
    parts = []
    if split != "test":
        parts.append(str(split))
    if band_cutoff is not None:
        parts.append(f"k{int(band_cutoff)}")
    if rollout_steps is not None:
        parts.append(f"T{int(rollout_steps)}")
    return "_" + "_".join(parts) if parts else ""


def limit_trajectories(dataset, limit):
    """The first ``limit`` rollout windows of an evaluation split, or all of them.

    A figure needs one window of one seed where the campaign evaluates the whole split.
    The split is ordered, so the first ``limit`` of it are the windows the full
    evaluation begins with, over the same data.

    A window is what the evaluation calls a trajectory: ``transform_rollout`` cuts each
    raw trajectory into as many windows of the rollout horizon as it holds, so E2's test
    split is 42 windows at T = 64 and 6 at T = 448.

    A deterministic rollout of the first window is the rollout the full evaluation
    measured; a test-time-optimised one is not, because TTO adapts per window and never
    saves its weights, so the short run adapts afresh over the same data.

    Args:
        dataset: The evaluation split, already cut into rollout windows.
        limit: How many windows to keep, or ``None`` for all of them.

    Returns:
        The dataset itself when ``limit`` is ``None``, otherwise a view of its first
        ``limit`` windows.
    """
    if limit is None:
        return dataset
    return Subset(dataset, range(min(limit, len(dataset))))


def check_split_available(datapath, filename, split):
    """Refuse a split the dataset does not carry.

    ``NSLoader2D`` falls back to the first ``X``/``y`` key in the file when
    ``X_<split>`` is missing, and in every dataset this repository builds that
    first key is ``X_train``. A mistyped split would therefore report training
    numbers as test numbers, with nothing in the output to say so.

    Args:
        datapath: Directory holding the dataset.
        filename: Name of the npz file within it.
        split: The split the evaluation asks for.

    Raises:
        KeyError: The file carries no ``X_<split>``/``y_<split>`` pair.
    """
    with np.load(os.path.join(datapath, filename)) as data:
        keys = set(data.keys())
    missing = [key for key in (f"X_{split}", f"y_{split}") if key not in keys]
    if missing:
        states = sorted(key[2:] for key in keys if key.startswith("X_"))
        raise KeyError(
            f"the dataset carries no split {split!r} (missing {missing}); "
            f"available splits: {states}"
        )


def check_rollout_horizon(datapath, filename, split, steps):
    """Refuse a horizon that would run one rollout across two stretches of flow.

    ``NSLoader2D.transform_rollout`` reshapes the flat frame list into
    ``(n, steps)`` and checks that ``steps`` divides the frame count, nothing
    more. A split built from several realisations, or from several trajectories,
    is a concatenation of contiguous runs, and a horizon that divides the total
    but not the length of each run stitches independent stretches of the flow
    into one rollout and reports the result as though it were continuous.

    A run is a maximal stretch in which each pair begins where the previous one
    ended, read from ``times_<split>``. A dataset that records no times cannot
    be checked and is passed through.

    Args:
        datapath: Directory holding the dataset.
        filename: Name of the npz file within it.
        split: The split the evaluation asks for.
        steps: The rollout horizon.

    Raises:
        ValueError: A contiguous run of the split is not a multiple of ``steps``.
    """
    with np.load(os.path.join(datapath, filename)) as data:
        key = f"times_{split}"
        if key not in data:
            return
        times = np.asarray(data[key], dtype=np.float64)
    if times.ndim != 2 or times.shape[1] != 2 or len(times) == 0:
        return

    # The stored times carry a jitter of a few parts in 1e9 of dt, so the
    # comparison is absolute against a small fraction of the shortest interval.
    interval = float(np.median(times[:, 1] - times[:, 0]))
    continues = np.isclose(times[1:, 0], times[:-1, 1], rtol=0.0, atol=abs(interval) * 1e-3)
    boundaries = np.flatnonzero(~continues) + 1
    runs = np.diff(np.concatenate([[0], boundaries, [len(times)]]))

    offenders = sorted({int(run) for run in runs if run % steps})
    if offenders:
        raise ValueError(
            f"a rollout of {steps} steps does not fit the {split!r} split: it is "
            f"{len(runs)} contiguous run(s) of length(s) {sorted(set(runs.tolist()))}, "
            f"and {offenders} are not multiples of {steps}. A horizon that divides "
            "the total but not each run would join independent stretches of the flow "
            "into one rollout."
        )


def model_uses_output_constraint(model):
    return bool(getattr(model, "output_constraint_enabled", False))


def get_model_constraint_domain_lengths(model):
    """Domain of the model's output projector, as configured from config['data']."""
    domain_lengths = getattr(getattr(model, "output_projector", None), "domain_lengths", None)
    if domain_lengths is None:
        raise ValueError("the model's output projector carries no domain lengths")
    return tuple(float(length) for length in domain_lengths)


def velocity_sequence_from_vorticity(sequence_phys, domain_length):
    """Velocity channels of a one-channel vorticity sequence.

    The spectra and the vorticity snapshots are defined on the velocity field,
    which the vorticity formulation does not carry; it is recovered here by the
    same spectral inversion the vorticity residual uses.

    Args:
        sequence_phys: (N, H, W, 1, T) vorticity in physical units.

    Returns:
        Tensor of shape (N, H, W, 2, T) holding (ux, uy).
    """
    ux, uy = velocity_from_vorticity(sequence_phys[..., 0, :], domain_length)
    return torch.stack([ux, uy], dim=-2)


def compute_div_max_per_step(sequence_phys, velocity_channels, domain_lengths):
    """Return per-sample, per-step max |div u| for a trajectory.

    Args:
        sequence_phys: (N, H, W, C, T)
    """
    n_samples, nx, ny, _, n_steps = sequence_phys.shape
    velocity = (
        torch.stack(
            [sequence_phys[..., channel_idx, :] for channel_idx in velocity_channels],
            dim=-2,
        )
        .permute(0, 4, 1, 2, 3)
        .reshape(n_samples * n_steps, nx, ny, 2)
    )
    div_max = max_abs_divergence_2d_velocity(
        velocity,
        domain_lengths=domain_lengths,
    )
    return div_max.reshape(n_samples, n_steps).detach().cpu().numpy()


def compute_domain_mean_velocity_per_step(sequence_phys, velocity_channels):
    """Return per-sample, per-step domain-mean velocity diagnostics.

    Args:
        sequence_phys: (N, H, W, C, T)

    Returns:
        dict with keys:
            mean_ux: (N, T)
            mean_uy: (N, T)
            mean_speed: (N, T)
    """
    mean_ux = sequence_phys[..., velocity_channels[0], :].mean(dim=(1, 2))
    mean_uy = sequence_phys[..., velocity_channels[1], :].mean(dim=(1, 2))
    mean_speed = torch.sqrt(mean_ux.square() + mean_uy.square())
    return {
        "mean_ux": mean_ux.detach().cpu().numpy(),
        "mean_uy": mean_uy.detach().cpu().numpy(),
        "mean_speed": mean_speed.detach().cpu().numpy(),
    }


def compute_energy_balance_per_step(
    sequence_phys, velocity_channels, forcing, viscosity, friction, dt, domain_lengths
):
    """Return per-sample, per-step global energy-balance diagnostics.

    The residual matches the trapezoidal-rule balance enforced by the
    spectral output projector:

        h = E_{n+1} - E_n
            + (dt / 2) * [nu * (Z_n + Z_{n+1})
                          + 2 * alpha * (E_n + E_{n+1})
                          - (Inj_n + Inj_{n+1})]

    where E is kinetic energy, Z is the spectral dissipation integral
    ``int |grad u|^2 dx`` (equal to enstrophy for divergence-free periodic
    velocity fields), and Inj is the energy injection ``int u.f dx``.

    This is the UN-closed residual. An arm carrying a subgrid source S
    (``output_constraint.energy_balance.subgrid_source``, #83) satisfies
    ``h = dt * S``, not ``h = 0``, so its reported residual sits at
    ``dt * S`` by construction. That is the right comparison rather than a
    violation: the truth's own residual on that data sits there too, which
    is where S was fitted from.

    Args:
        sequence_phys: (N, H, W, C, T) in physical units, including the IC.
        forcing: (1, 2, H, W, 1) or (1, 2, H, W) velocity forcing.

    Returns:
        dict containing per-state energy terms with shape (N, T) and
        per-transition residuals with shape (N, T-1).
    """
    velocity = torch.stack(
        [sequence_phys[..., channel_idx, :] for channel_idx in velocity_channels],
        dim=-2,
    ).permute(0, 4, 3, 1, 2)  # (N, T, 2, H, W)

    n_samples, n_steps, _, nx, ny = velocity.shape
    area = float(domain_lengths[0]) * float(domain_lengths[1])
    n_grid = nx * ny
    cell_area = area / n_grid

    forcing_velocity = forcing.squeeze(-1) if forcing.ndim == 5 else forcing
    if forcing_velocity.ndim != 4 or forcing_velocity.shape[-3] != 2:
        raise ValueError("forcing must have shape (1, 2, H, W, 1) or (1, 2, H, W)")
    forcing_velocity = forcing_velocity.to(device=velocity.device, dtype=velocity.dtype)
    forcing_velocity = forcing_velocity.unsqueeze(1).expand(n_samples, n_steps, -1, -1, -1)

    energy = 0.5 * cell_area * velocity.square().sum(dim=(-3, -2, -1))
    injection = cell_area * (velocity * forcing_velocity).sum(dim=(-3, -2, -1))

    velocity_hat = torch.fft.fft2(velocity, dim=(-2, -1))
    dx = float(domain_lengths[0]) / nx
    dy = float(domain_lengths[1]) / ny
    kx = 2 * math.pi * torch.fft.fftfreq(nx, d=dx).to(device=velocity.device, dtype=velocity.dtype)
    ky = 2 * math.pi * torch.fft.fftfreq(ny, d=dy).to(device=velocity.device, dtype=velocity.dtype)
    k_sq = kx.reshape(1, 1, nx, 1).square() + ky.reshape(1, 1, 1, ny).square()
    dissipation = (cell_area / n_grid) * (k_sq * velocity_hat.abs().square()).sum(dim=(-3, -2, -1))

    balance_residual = (
        energy[:, 1:]
        - energy[:, :-1]
        + 0.5
        * dt
        * (
            viscosity * (dissipation[:, :-1] + dissipation[:, 1:])
            + 2.0 * friction * (energy[:, :-1] + energy[:, 1:])
            - (injection[:, :-1] + injection[:, 1:])
        )
    )
    balance_scale = (energy[:, 1:] - energy[:, :-1]).abs() + 0.5 * dt * (
        viscosity * (dissipation[:, :-1] + dissipation[:, 1:])
        + 2.0 * friction * (energy[:, :-1] + energy[:, 1:])
        + injection[:, :-1].abs()
        + injection[:, 1:].abs()
    )

    return {
        "energy": energy.detach().cpu().numpy(),
        "dissipation": dissipation.detach().cpu().numpy(),
        "injection": injection.detach().cpu().numpy(),
        "balance_residual": balance_residual.detach().cpu().numpy(),
        "balance_abs_residual": balance_residual.abs().detach().cpu().numpy(),
        "balance_rel_residual": (balance_residual.abs() / (balance_scale + LOSS_GUARD_EPS))
        .detach()
        .cpu()
        .numpy(),
    }


def align_prediction(pred, prev):
    """Drop a trailing singleton axis of a model output, never the channel axis.

    A model may return the next state as ``(B, S, S, C)``, ``(B, S, S, C, 1)`` or
    ``(B, S, S, 1, T)``. Squeezing whatever is trailing also eats the channel axis
    of a one-channel state, which is what the vorticity formulation carries.

    Args:
        pred: The model's output for one step.
        prev: The state it continues, whose rank the result must match.

    Returns:
        Tensor of the same rank as ``prev``.
    """
    if pred.dim() == prev.dim() + 1:
        return pred.squeeze(-2) if pred.shape[-2] == 1 else pred.squeeze(-1)
    return pred


def autoregressive_rollout(model, initial_condition, grid, rollout_steps, use_residual=False):
    """Roll out a model from a single initial condition tensor.

    Args:
        initial_condition: (B, S, S, C)
        rollout_steps: number of predicted steps

    Returns:
        Tensor of shape (B, S, S, C, rollout_steps + 1) including the IC.
    """
    preds = [initial_condition]
    prev = initial_condition
    grid_batch = grid.unsqueeze(0).expand(prev.shape[0], -1, -1, -1)
    for _ in range(rollout_steps):
        x_in = torch.cat((prev, grid_batch), dim=-1)
        pred = align_prediction(model(x_in), prev)
        if use_residual:
            pred = pred + prev
        preds.append(pred)
        prev = pred
    return torch.stack(preds, dim=-1)


def autoregressive_rollout_with_output_constraint(
    model, initial_condition, grid, rollout_steps, use_residual=False
):
    """Roll out a model while constraining each predicted next state."""
    preds = [initial_condition]
    prev = initial_condition
    grid_batch = grid.unsqueeze(0).expand(prev.shape[0], -1, -1, -1)
    for _ in range(rollout_steps):
        x_in = torch.cat((prev, grid_batch), dim=-1)
        pred = align_prediction(model(x_in), prev)
        if use_residual:
            pred = pred + prev
        pred = model.apply_output_constraint(pred, x_old=prev)
        preds.append(pred)
        prev = pred
    return torch.stack(preds, dim=-1)


def autoregressive_predict(
    model, test_loader, device, grid, use_residual=False, constrain_output=False
):
    """Run autoregressive rollout on full sequences.

    Returns:
        initial_condition: (N, S, S, C)
        pred_seq:  (N, S, S, C, T+1) — IC at index 0, predictions at 1..T
        truth_seq: (N, S, S, C, T)   — ground truth at steps 1..T
    """
    model.eval()
    total_pred = []
    initial_condition = []
    total_ground_truth = []
    with torch.no_grad():
        for seq, truth in test_loader:
            seq = seq.to(device)  # (B, S, S, T, C)
            truth = truth.to(device)  # (B, S, S, T, C)
            T = seq.shape[-2]
            prev = seq[..., 0, :]  # IC: (B, S, S, C)
            initial_condition.append(prev)
            total_ground_truth.append(truth)
            if constrain_output:
                pred_seq = autoregressive_rollout_with_output_constraint(
                    model,
                    prev,
                    grid,
                    T,
                    use_residual=use_residual,
                )
            else:
                pred_seq = autoregressive_rollout(
                    model,
                    prev,
                    grid,
                    T,
                    use_residual=use_residual,
                )
            total_pred.append(pred_seq)
        total_pred = torch.cat(total_pred, dim=0)
        initial_condition = torch.cat(initial_condition, dim=0)
        total_ground_truth = torch.cat(total_ground_truth, dim=0).permute(0, 1, 2, 4, 3)
    print(
        f"Rollout complete — pred: {total_pred.shape}, "
        f"truth: {total_ground_truth.shape}, IC: {initial_condition.shape}"
    )
    return initial_condition, total_pred, total_ground_truth


def compute_per_step_l2(pred_seq, truth_seq):
    """Relative L2 error at each rollout step.

    Args:
        pred_seq:  (N, S, S, C, T+1) — IC at index 0
        truth_seq: (N, S, S, C, T)   — ground truth at steps 1..T

    Returns:
        np.ndarray of shape (N, T) with one relative L2 error per
        sample and rollout step.
    """
    lploss = LpLoss(reduction=False)
    T = truth_seq.shape[-1]
    per_step = [
        lploss(pred_seq[..., t + 1], truth_seq[..., t]).detach().cpu().numpy() for t in range(T)
    ]
    return np.stack(per_step, axis=1)


def compute_blowup_step(l2_per_step):
    """First rollout step at which each trajectory's relative L2 leaves the bounded range.

    A rollout evaluated over a fixed horizon reports only whether a trajectory failed inside
    the window, not how close to failing it came; on this data those are very different
    questions (#42).

    Args:
        l2_per_step: (N, T) relative L2 per trajectory and rollout step.

    Returns:
        np.ndarray of shape (N,), the 1-based step at which the relative L2 first exceeds
        ``BOUNDED_L2`` or goes non-finite, and infinity for a trajectory that never does.
    """
    steps = blowup_steps(np.asarray(l2_per_step).T, threshold=BOUNDED_L2)
    return np.array([np.inf if step < 0 else float(step) for step in steps])


def blowup_quantile(blowup_step, q):
    """Nearest-rank quantile of the blow-up steps.

    Nearest-rank rather than interpolated, so the reported value is a step some trajectory
    actually reached, and so infinity (a trajectory that never blew up) propagates as
    infinity rather than as an interpolated NaN.

    Args:
        blowup_step: (N,) blow-up steps, infinity for a trajectory that never blew up.
        q: The quantile in [0, 1].

    Returns:
        The quantile as a float, possibly infinite.
    """
    return float(np.quantile(np.asarray(blowup_step, dtype=float), q, method="lower"))


def compute_high_band_fraction_per_step(sequence_phys, velocity_channels, mode_cutoff):
    """Share of the kinetic energy above the model's spectral-mode cutoff, per step.

    The FNO's spectral layers write only the ``mode_cutoff`` lowest wavenumbers per axis, so
    every shell from ``|k| = mode_cutoff`` upwards is written by the pointwise path alone and
    is weighed by no term of the one-step loss. Energy accumulating there is the mechanism
    behind the rollout blow-ups (#42), and the two datasets start from very different
    fractions, so the diagnostic is reported for the prediction and for the truth alike.

    The energy is summed in float64: a diverging rollout overflows float32, and the ratio of
    two infinities carries no information about the band.

    Args:
        sequence_phys: (N, S, S, C, T) rollout in physical units.
        velocity_channels: The two channels holding (u, v).
        mode_cutoff: The model's ``modes1``; shells with |k| >= cutoff are above it.

    Returns:
        np.ndarray of shape (N, T), the energy fraction in shells |k| >= ``mode_cutoff``.
    """
    velocity = sequence_phys[..., list(velocity_channels), :]
    side = velocity.shape[1]
    per_step = []
    # Cast one step at a time: the whole rollout in float64 is a gigabyte on E1.
    for t in range(velocity.shape[-1]):
        shells = shell_energy(velocity[..., t].to(torch.float64), side)
        per_step.append((shells[:, mode_cutoff:].sum(1) / shells.sum(1)).detach().cpu().numpy())
    return np.stack(per_step, axis=1)


def compute_PDE_loss_per_step(pred_seq, forcing, physics, scales):
    """PDE residuals for each consecutive pair in a trajectory.

    Uses the same residual as training, applied to each (u_t, u_{t+1})
    transition individually.

    Args:
        pred_seq: (N, S, S, C, T+1) — full trajectory including IC at index 0
        forcing:  the run's forcing, as returned by ``build_forcing``
        physics:  the run's ``Physics``
        scales:   ``ResidualScales`` measured on the ground-truth trajectories,
                  or None for the vorticity formulation, whose single residual is
                  relative to the forcing and needs no measured scale

    Returns:
        dict mapping metric names to np.ndarray of shape (N, T)
    """
    N = pred_seq.shape[0]
    T = pred_seq.shape[-1] - 1
    keys = [
        "loss_ic",
        "loss_cont",
        "loss_momx",
        "loss_momy",
        "loss_cont_rel",
        "loss_momx_rel",
        "loss_momy_rel",
    ]
    if not residual_has_scale(physics, forcing):
        # An unforced vorticity run has no residual scale, so every entry is missing
        # rather than wrong; the aggregator renders a missing metric as "--" (#18).
        return {k: np.full((N, T), np.nan, dtype=np.float32) for k in keys}

    results = {k: np.zeros((N, T), dtype=np.float32) for k in keys}

    with torch.no_grad():
        for t in range(T):
            for b in range(N):
                pair = pred_seq[b : b + 1, ..., t : t + 2]  # (1, S, S, C, 2)
                u = pair.permute(0, 3, 1, 2, 4)  # (1, C, S, S, 2)
                (
                    loss_ic,
                    loss_cont,
                    loss_momx,
                    loss_momy,
                    loss_cont_rel,
                    loss_momx_rel,
                    loss_momy_rel,
                ) = PINO_loss3d_physics(u, forcing, physics, t_interval=physics.dt, scales=scales)

                results["loss_ic"][b, t] = loss_ic.item()
                results["loss_cont"][b, t] = loss_cont.item()
                results["loss_momx"][b, t] = loss_momx.item()
                results["loss_momy"][b, t] = loss_momy.item()
                results["loss_cont_rel"][b, t] = loss_cont_rel.item()
                results["loss_momx_rel"][b, t] = loss_momx_rel.item()
                results["loss_momy_rel"][b, t] = loss_momy_rel.item()

    return results


def compute_rollout_pde_objective(
    pred_phys, forcing, physics, scales, cont_weight, momx_weight, momy_weight
):
    """Whole-trajectory PDE objective used during test-time optimization."""
    rollout_steps = pred_phys.shape[-1] - 1
    if rollout_steps <= 0:
        raise ValueError("Need at least one predicted step for rollout PDE loss.")
    u = pred_phys.permute(0, 3, 1, 2, 4)
    (
        loss_ic,
        loss_cont,
        loss_momx,
        loss_momy,
        loss_cont_rel,
        loss_momx_rel,
        loss_momy_rel,
    ) = PINO_loss3d_physics(
        u, forcing, physics, t_interval=physics.dt * rollout_steps, scales=scales
    )
    total_loss = (
        cont_weight * loss_cont_rel + momx_weight * loss_momx_rel + momy_weight * loss_momy_rel
    )
    return total_loss, {
        "loss_ic": loss_ic,
        "loss_cont": loss_cont,
        "loss_momx": loss_momx,
        "loss_momy": loss_momy,
        "loss_cont_rel": loss_cont_rel,
        "loss_momx_rel": loss_momx_rel,
        "loss_momy_rel": loss_momy_rel,
    }


def set_requires_grad(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad


def configure_tto_trainable_scope(model, model_name, trainable_scope):
    """Configure which parameters are updated during test-time optimization."""
    scope = trainable_scope.lower()
    if scope == "full":
        set_requires_grad(model, True)
    elif scope == "last_layers":
        set_requires_grad(model, False)
        if model_name == "fno2d":
            modules = [model.sp_convs[-1], model.ws[-1], model.fc1, model.fc2, model.fc3]
        else:
            raise ValueError(
                f'TTO trainable_scope="last_layers" is not supported for model {model_name}.'
            )
        for module in modules:
            set_requires_grad(module, True)
    else:
        raise ValueError(f"Unsupported TTO trainable_scope: {trainable_scope}")

    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        raise RuntimeError("TTO did not leave any trainable parameters enabled.")
    return params


def clone_model_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def tensor_is_finite(tensor):
    return bool(torch.isfinite(tensor).all().item())


def tensor_dict_is_finite(values):
    return all(tensor_is_finite(value) for value in values.values())


def gradients_are_finite(params):
    for param in params:
        if param.grad is None:
            continue
        if not tensor_is_finite(param.grad):
            return False
    return True


def metric_mean(values, axis=None):
    """Strict mean: a non-finite sample propagates instead of being dropped.

    ``np.nanmean`` would average a diverged trajectory away silently; the
    non-finite entries are counted and reported by ``count_non_finite`` instead.
    """
    return np.mean(np.asarray(values, dtype=float), axis=axis)


def metric_std(values, axis=None):
    """Strict standard deviation; see :func:`metric_mean`."""
    return np.std(np.asarray(values, dtype=float), axis=axis)


def count_non_finite(arrays):
    """Number of non-finite entries in each named metric array."""
    return {
        name: int((~np.isfinite(np.asarray(values, dtype=float))).sum())
        for name, values in arrays.items()
    }


def resolve_checkpoint_path(save_dir, save_name, seed, which="best"):
    """Path of the checkpoint to evaluate.

    ``"best"`` is the checkpoint selected on the validation metric during
    training; ``"last"`` is the final one written.
    """
    if which not in ("best", "last"):
        raise ValueError('checkpoint must be "best" or "last"')
    stem = os.path.join(save_dir, save_name).replace(".pt", f"_seed{seed}")
    suffix = "_best.pt" if which == "best" else ".pt"
    return stem + suffix


def save_tto_diagnostics(sample_reports, save_dir, experiment_name, seed, file_suffix=""):
    """Persist per-sample TTO diagnostics for later inspection."""
    results_dir = os.path.join(save_dir, "evaluation_metrics")
    os.makedirs(results_dir, exist_ok=True)
    status_counts = Counter(report["status"] for report in sample_reports)
    payload = {
        "summary": {
            "num_samples": len(sample_reports),
            "status_counts": dict(status_counts),
            "fallback_sample_indices_1based": [
                report["sample_index_1based"]
                for report in sample_reports
                if report["fallback_used"]
            ],
            "failed_sample_indices_1based": [
                report["sample_index_1based"]
                for report in sample_reports
                if report["failure_reason"] is not None
            ],
        },
        "samples": sample_reports,
    }
    report_path = os.path.join(
        results_dir,
        append_file_suffix(f"{experiment_name}_seed{seed}_diagnostics", file_suffix) + ".json",
    )
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"TTO diagnostics saved to {report_path}")


def resolve_free_rollout_band(tto_cfg, mode_cutoff):
    """The shell the adapted arm's free-running trace reports its high band above.

    Set apart from the metrics' own cutoff: the adapted arm has to be measured over the band
    the campaign's plain free rollouts were measured over, while the T = 64 metrics stay
    above the model's own spectral cutoff. Resolved in one place, because the band recorded
    in the trace file and the band the trace was measured over must never disagree.

    Args:
        tto_cfg: The evaluation's ``tto`` block.
        mode_cutoff: The band the metrics use, or None when the model declares none.

    Returns:
        The cutoff, or None when neither is given.
    """
    return tto_cfg.get("free_rollout_band_cutoff", mode_cutoff)


def run_test_time_optimization(
    model,
    reference_state,
    test_loader,
    device,
    grid,
    forcing,
    norm_mean,
    norm_std,
    use_residual,
    tto_cfg,
    model_name,
    cont_weight,
    momx_weight,
    momy_weight,
    physics,
    scales,
    mode_cutoff=None,
):
    """Adapt the pretrained model on each test trajectory using PDE loss."""
    num_iter = int(tto_cfg.get("num_iter", 0))
    base_lr = float(tto_cfg.get("base_lr", 1e-3))
    milestones = list(tto_cfg.get("milestones", []))
    scheduler_gamma = float(tto_cfg.get("scheduler_gamma", 0.5))
    trainable_scope = tto_cfg.get("trainable_scope", "full")
    grad_clip_norm_cfg = tto_cfg.get("grad_clip_norm", 1.0)
    grad_clip_norm = None if grad_clip_norm_cfg is None else float(grad_clip_norm_cfg)
    early_stop_patience = int(tto_cfg.get("early_stop_patience", 0))
    improvement_tol = float(tto_cfg.get("improvement_tol", 0.0))
    max_loss_multiplier_cfg = tto_cfg.get("max_loss_multiplier", 10.0)
    max_loss_multiplier = (
        None if max_loss_multiplier_cfg is None else float(max_loss_multiplier_cfg)
    )
    restore_best_weights = bool(tto_cfg.get("restore_best_weights", True))
    fallback_on_failure = bool(tto_cfg.get("fallback_on_failure", True))
    # Adapting over a 448-step graph costs 448 forward passes and their backward pass at
    # every iteration. adapt_steps caps the horizon the objective is taken over; the
    # adapted weights are still evaluated on the whole one. Unset means the whole horizon,
    # so the campaigns that predate this key are unchanged.
    adapt_steps_cfg = tto_cfg.get("adapt_steps")
    adapt_steps = None if adapt_steps_cfg is None else int(adapt_steps_cfg)
    # Free-running stability of the adapted weights. The weights are per trajectory, so the
    # rollout has to happen here, under the weights this trajectory ended on.
    free_rollout_steps = int(tto_cfg.get("free_rollout_steps", 0) or 0)
    free_band_cutoff = resolve_free_rollout_band(tto_cfg, mode_cutoff)
    if free_rollout_steps > 0 and free_band_cutoff is None:
        raise ValueError(
            "the free rollout reports the energy above a spectral cutoff, so mode_cutoff "
            "or tto.free_rollout_band_cutoff has to be given: two models measured over "
            "different bands report a differently defined column under one name"
        )

    total_samples = len(test_loader.dataset)
    print(
        f"\nStarting TTO over {total_samples} trajectories with "
        f"{num_iter} iterations each (scope={trainable_scope}, lr={base_lr:.2e})."
    )
    if grad_clip_norm is not None:
        print(f"Using TTO grad clipping with max norm {grad_clip_norm:.2f}.")
    if max_loss_multiplier is not None:
        print(f"Using TTO loss explosion guard with multiplier {max_loss_multiplier:.2f}.")
    if early_stop_patience > 0:
        print(f"Using TTO early stopping patience of {early_stop_patience} iterations.")
    if adapt_steps is not None:
        print(f"Adapting on the first {adapt_steps} steps of each trajectory.")
    if free_rollout_steps > 0:
        print(f"Free-running each adapted trajectory for {free_rollout_steps} steps.")

    initial_condition = []
    total_pred = []
    total_ground_truth = []
    sample_reports = []
    free_traces = []

    for sample_idx, (seq, truth) in enumerate(test_loader, start=1):
        seq = seq.to(device)
        truth = truth.to(device)
        rollout_steps = seq.shape[-2]
        adapt_rollout_steps = (
            min(adapt_steps, rollout_steps) if adapt_steps is not None else rollout_steps
        )
        prev = seq[..., 0, :]

        model.load_state_dict(reference_state)
        trainable_params = configure_tto_trainable_scope(
            model,
            model_name,
            trainable_scope,
        )
        optimizer = torch.optim.Adam(trainable_params, lr=base_lr)
        scheduler = None
        if milestones:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=milestones,
                gamma=scheduler_gamma,
            )

        print(f"\n── TTO sample {sample_idx}/{total_samples} ──")
        model.train()
        best_state = reference_state
        best_loss = math.inf
        best_iteration = 0
        initial_loss = None
        failure_reason = None
        no_improvement_iters = 0
        successful_iterations = 0
        stopped_early = False
        fallback_used = False
        status = "adapted"
        for iteration in range(num_iter):
            optimizer.zero_grad(set_to_none=True)
            pred_seq = autoregressive_rollout(
                model,
                prev,
                grid,
                adapt_rollout_steps,
                use_residual=use_residual,
            )
            if not tensor_is_finite(pred_seq):
                failure_reason = f"Non-finite rollout prediction at iteration {iteration + 1}"
                print(f"  stopping: {failure_reason}.")
                del pred_seq
                break
            pred_phys = denormalise(pred_seq, norm_mean, norm_std, device)
            total_loss, loss_dict = compute_rollout_pde_objective(
                pred_phys,
                forcing,
                physics,
                scales,
                cont_weight=cont_weight,
                momx_weight=momx_weight,
                momy_weight=momy_weight,
            )
            if not tensor_is_finite(pred_phys) or not tensor_is_finite(total_loss):
                failure_reason = f"Non-finite rollout PDE objective at iteration {iteration + 1}"
                print(f"  stopping: {failure_reason}.")
                del pred_seq, pred_phys, total_loss, loss_dict
                break
            if not tensor_dict_is_finite(loss_dict):
                failure_reason = f"Non-finite PDE residual component at iteration {iteration + 1}"
                print(f"  stopping: {failure_reason}.")
                del pred_seq, pred_phys, total_loss, loss_dict
                break

            loss_value = total_loss.item()
            if initial_loss is None:
                initial_loss = loss_value

            if restore_best_weights and loss_value + improvement_tol < best_loss:
                best_state = clone_model_state(model)
                best_loss = loss_value
                best_iteration = iteration + 1
                no_improvement_iters = 0
            elif loss_value + improvement_tol < best_loss:
                best_loss = loss_value
                best_iteration = iteration + 1
                no_improvement_iters = 0
            else:
                no_improvement_iters += 1

            if (
                max_loss_multiplier is not None
                and best_iteration > 0
                and loss_value > max_loss_multiplier * max(best_loss, LOSS_GUARD_EPS)
            ):
                failure_reason = (
                    f"Loss explosion at iteration {iteration + 1}: "
                    f"{loss_value:.4e} exceeds "
                    f"{max_loss_multiplier:.2f} x best loss {best_loss:.4e}"
                )
                print(f"  stopping: {failure_reason}.")
                del pred_seq, pred_phys, total_loss, loss_dict
                break

            total_loss.backward()
            if not gradients_are_finite(trainable_params):
                failure_reason = f"Non-finite gradients at iteration {iteration + 1}"
                print(f"  stopping: {failure_reason}.")
                optimizer.zero_grad(set_to_none=True)
                del pred_seq, pred_phys, total_loss, loss_dict
                break
            if grad_clip_norm is not None and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            successful_iterations = iteration + 1

            report_stride = max(1, num_iter // 5)
            if iteration == 0 or iteration == num_iter - 1 or (iteration + 1) % report_stride == 0:
                print(
                    f"  iter {iteration + 1:4d}/{num_iter}: "
                    f"total={loss_value:.4e}, "
                    f"cont_rel={loss_dict['loss_cont_rel'].item():.4e}, "
                    f"momx_rel={loss_dict['loss_momx_rel'].item():.4e}, "
                    f"momy_rel={loss_dict['loss_momy_rel'].item():.4e}"
                )

            if early_stop_patience > 0 and no_improvement_iters >= early_stop_patience:
                stopped_early = True
                print(
                    f"  early stop at iter {iteration + 1}: "
                    f"best_iter={best_iteration}, best_loss={best_loss:.4e}"
                )
                del pred_seq, pred_phys, total_loss, loss_dict
                break
            del pred_seq, pred_phys, total_loss, loss_dict

        if failure_reason is not None:
            if restore_best_weights and best_iteration > 0:
                model.load_state_dict(best_state)
                fallback_used = True
                status = "recovered_best_after_failure"
                print(
                    f"  restoring best finite TTO weights from iter {best_iteration} "
                    f"(loss={best_loss:.4e})."
                )
            elif fallback_on_failure:
                model.load_state_dict(reference_state)
                fallback_used = True
                status = "recovered_base_after_failure"
                print("  restoring pretrained reference weights for this sample.")
            else:
                raise RuntimeError(
                    f"TTO failed for sample {sample_idx} without fallback enabled: {failure_reason}"
                )
        elif restore_best_weights and best_iteration > 0:
            model.load_state_dict(best_state)
            status = "early_stopped_best" if stopped_early else "adapted_best"
        elif num_iter <= 0:
            model.load_state_dict(reference_state)
            status = "no_tto_iterations"
        else:
            status = "adapted_last"

        model.eval()
        with torch.no_grad():
            pred_seq = autoregressive_rollout(
                model,
                prev,
                grid,
                rollout_steps,
                use_residual=use_residual,
            )
        if not tensor_is_finite(pred_seq):
            print(
                "  final adapted rollout was non-finite; "
                "falling back to pretrained reference weights."
            )
            model.load_state_dict(reference_state)
            with torch.no_grad():
                pred_seq = autoregressive_rollout(
                    model,
                    prev,
                    grid,
                    rollout_steps,
                    use_residual=use_residual,
                )
            if not tensor_is_finite(pred_seq):
                raise RuntimeError(
                    f"Fallback rollout also became non-finite for sample {sample_idx}."
                )
            fallback_used = True
            if failure_reason is None:
                failure_reason = "Non-finite final rollout after adaptation"
            status = "recovered_base_after_nonfinite_rollout"

        if free_rollout_steps > 0:
            # One trajectory at a time, which is what tto_loader hands over: the weights are
            # this trajectory's, and the padding below marks the whole batch as gone at the
            # first member's failure step, so a wider batch would be wrong on both counts.
            assert prev.shape[0] == 1, "the TTO free rollout adapts one trajectory at a time"
            with torch.no_grad():
                trace = free_rollout(
                    model,
                    prev,
                    grid,
                    free_rollout_steps,
                    prev.shape[1],
                    band_start=int(free_band_cutoff),
                    use_residual=use_residual,
                )
            # A trajectory whose rollout goes non-finite stops short. Pad it out so every
            # trajectory shares one time axis; the padding sits after the first crossing,
            # which is what blowup_steps reports, so no reported number moves.
            if trace.shape[0] < free_rollout_steps:
                trace = np.concatenate(
                    [
                        trace,
                        np.full(
                            (free_rollout_steps - trace.shape[0], *trace.shape[1:]),
                            np.inf,
                            dtype=trace.dtype,
                        ),
                    ],
                    axis=0,
                )
            free_traces.append(trace)
            print(
                f"  free rollout: energy ratio {trace[-1, 0, 0]:.3e} at step "
                f"{free_rollout_steps}, blow-up step {blowup_steps(trace[..., 0])[0]}"
            )

        initial_condition.append(prev.detach())
        total_pred.append(pred_seq.detach())
        total_ground_truth.append(truth.detach().permute(0, 1, 2, 4, 3))
        sample_reports.append(
            {
                "sample_index_0based": sample_idx - 1,
                "sample_index_1based": sample_idx,
                "status": status,
                "fallback_used": fallback_used,
                "failure_reason": failure_reason,
                "initial_loss": None if initial_loss is None else float(initial_loss),
                "best_loss": None if best_iteration == 0 else float(best_loss),
                "best_iteration": int(best_iteration),
                "successful_iterations": int(successful_iterations),
                "stopped_early": bool(stopped_early),
            }
        )
        print(
            f"  finished sample {sample_idx}: status={status}, "
            f"best_iter={best_iteration}, fallback={fallback_used}"
        )

        del optimizer, scheduler, seq, truth, pred_seq
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    status_counts = Counter(report["status"] for report in sample_reports)
    print("\n── TTO Status Summary ──")
    for status_name, count in sorted(status_counts.items()):
        print(f"  {status_name}: {count}")

    return (
        torch.cat(initial_condition, dim=0),
        torch.cat(total_pred, dim=0),
        torch.cat(total_ground_truth, dim=0),
        sample_reports,
        np.concatenate(free_traces, axis=1) if free_traces else None,
    )


def compute_vorticity_sequence(ux, uy):
    """Compute vorticity omega = d uy / dx - d ux / dy for all timesteps."""
    nx, ny, _ = ux.shape
    device = ux.device
    kx = torch.cat(
        (torch.arange(0, nx // 2, device=device), torch.arange(-nx // 2, 0, device=device)),
        dim=0,
    ).reshape(nx, 1, 1)
    ky = torch.cat(
        (torch.arange(0, ny // 2, device=device), torch.arange(-ny // 2, 0, device=device)),
        dim=0,
    ).reshape(1, ny, 1)

    ux_h = torch.fft.fft2(ux, dim=(0, 1))
    uy_h = torch.fft.fft2(uy, dim=(0, 1))
    duy_dx = torch.fft.ifft2(1j * kx * uy_h, dim=(0, 1)).real
    dux_dy = torch.fft.ifft2(1j * ky * ux_h, dim=(0, 1)).real
    return duy_dx - dux_dy


def compute_save_vorticity(
    pred_phys,
    truth_phys,
    save_dir,
    model_name,
    seed,
    sample_idx=0,
    file_suffix="",
    state_is_vorticity=False,
):
    """Save full vorticity trajectories for one sample.

    ``state_is_vorticity`` says the state carries the vorticity itself, as the
    vorticity formulation does; otherwise it is computed from (ux, uy).
    """
    save_dir = os.path.join(save_dir, "saved_plots")
    os.makedirs(save_dir, exist_ok=True)

    pred_sample = pred_phys[sample_idx].detach().cpu()  # (H, W, C, T+1)
    truth_sample = truth_phys[sample_idx].detach().cpu()  # (H, W, C, T+1)

    if state_is_vorticity:
        pred_vorticity = pred_sample[..., 0, :]
        truth_vorticity = truth_sample[..., 0, :]
    else:
        pred_vorticity = compute_vorticity_sequence(pred_sample[..., 0, :], pred_sample[..., 1, :])
        truth_vorticity = compute_vorticity_sequence(
            truth_sample[..., 0, :], truth_sample[..., 1, :]
        )

    save_path = os.path.join(
        save_dir,
        append_file_suffix(
            f"{model_name}_seed{seed}_vorticity_sample{sample_idx}",
            file_suffix,
        )
        + ".npz",
    )
    np.savez(
        save_path,
        vorticity_pred=pred_vorticity.numpy(),
        vorticity_truth=truth_vorticity.numpy(),
        step_indices=np.arange(pred_vorticity.shape[-1]),
    )
    print(f"Saved vorticity trajectories to {save_path}")


def evaluate_model(
    truth_seq,
    pred_seq,
    model_name,
    seed,
    save_dir,
    Lx=2 * math.pi,
    Ly=2 * math.pi,
    save_csv=False,
    file_suffix="",
    extra=None,
    spectra_seqs=None,
):
    """Compute spectral and L2 metrics at selected time indices.

    Args:
        truth_seq: (B, H, W, C, T) — channels: (ux, uy, p)
        pred_seq:  (B, H, W, C, T) — predictions aligned with truth_seq
        spectra_seqs: optional ``(truth, pred)`` pair carrying (ux, uy) in the
            first two channels, for a state that is not itself a velocity; the
            L2 stays on the state channels either way.
    """
    spectra_truth, spectra_pred = (
        spectra_seqs if spectra_seqs is not None else (truth_seq, pred_seq)
    )
    lploss = LpLoss(size_average=True)
    meape = MeanEnergyAbsolutePercentageError()
    melr = MeanEnergyLogRatioError()

    T = truth_seq.shape[-1]
    time_idx = [0, T // 2, T - 1]
    metrics_name = ["l2", "SMLR", "EMLR", "SMAE", "EMAE"]
    metrics_dict = {}

    for metric in metrics_name:
        for t in time_idx:
            metrics_dict[f"{metric}_step{t + 1}"] = 0
    model_label = append_file_suffix(model_name, file_suffix)
    metrics_dict["seed"] = seed
    metrics_dict["model"] = model_label
    if extra:
        metrics_dict.update(extra)

    for t in time_idx:
        truth_t = truth_seq[..., t]  # (B, H, W, C)
        pred_t = pred_seq[..., t]

        ux_true = spectra_truth[..., 0, t].detach().cpu().numpy()  # (B, H, W)
        uy_true = spectra_truth[..., 1, t].detach().cpu().numpy()
        ux_pred = spectra_pred[..., 0, t].detach().cpu().numpy()
        uy_pred = spectra_pred[..., 1, t].detach().cpu().numpy()

        B = ux_true.shape[0]
        H, W = ux_true.shape[1], ux_true.shape[2]
        valid_k = min(H, W) // 2

        Ek_true_list, Zk_true_list = [], []
        Ek_pred_list, Zk_pred_list = [], []
        for b in range(B):
            k_bins, Ek_t, Zk_t = compute_spectra(ux_true[b], uy_true[b], Lx, Ly)
            _, Ek_p, Zk_p = compute_spectra(ux_pred[b], uy_pred[b], Lx, Ly)
            Ek_true_list.append(Ek_t)
            Zk_true_list.append(Zk_t)
            Ek_pred_list.append(Ek_p)
            Zk_pred_list.append(Zk_p)

        Ek_true = torch.tensor(np.stack(Ek_true_list), dtype=torch.float32)
        Zk_true = torch.tensor(np.stack(Zk_true_list), dtype=torch.float32)
        Ek_pred = torch.tensor(np.stack(Ek_pred_list), dtype=torch.float32)
        Zk_pred = torch.tensor(np.stack(Zk_pred_list), dtype=torch.float32)

        valid_mask = slice(1, valid_k)
        Ek_true_trunc = Ek_true[:, valid_mask]
        Ek_pred_trunc = Ek_pred[:, valid_mask]
        Zk_true_trunc = Zk_true[:, valid_mask]
        Zk_pred_trunc = Zk_pred[:, valid_mask]

        step_l2 = lploss(pred_t, truth_t).item()
        step_spectral_melr = melr(Ek_pred_trunc, Ek_true_trunc).item()
        step_spectral_meape = meape(Ek_pred_trunc, Ek_true_trunc).item()
        step_enstrophy_melr = melr(Zk_pred_trunc, Zk_true_trunc).item()
        step_enstrophy_meape = meape(Zk_pred_trunc, Zk_true_trunc).item()

        print(
            f"{model_label} seed: {seed}, step: {t}, L2: {step_l2:.4f}, "
            f"SMLR: {step_spectral_melr:.4f}, EMLR: {step_enstrophy_melr:.4f}, "
            f"SMAE: {step_spectral_meape:.4f}, EMAE: {step_enstrophy_meape:.4f}"
        )

        metrics_dict[f"l2_step{t + 1}"] = step_l2
        metrics_dict[f"SMLR_step{t + 1}"] = step_spectral_melr
        metrics_dict[f"EMLR_step{t + 1}"] = step_enstrophy_melr
        metrics_dict[f"SMAE_step{t + 1}"] = step_spectral_meape
        metrics_dict[f"EMAE_step{t + 1}"] = step_enstrophy_meape

    if save_csv:
        save_folder = os.path.join(save_dir, "evaluation_metrics")
        os.makedirs(save_folder, exist_ok=True)
        df = pd.Series(metrics_dict).to_frame().T
        df.to_csv(
            os.path.join(
                save_folder,
                append_file_suffix(f"{model_name}_seed{seed}_metrics", file_suffix) + ".csv",
            ),
            index=False,
        )

    return metrics_dict


def save_ground_truth_and_predictions(
    initial_condition, truth_seq, pred_seq, time_indices, save_dir, model_name, seed
):
    """Save snapshots of ground truth and predictions as .npz files.

    Args:
        truth_seq: (B, H, W, C, T)
        pred_seq:  (B, H, W, C, T) — aligned with truth_seq
    """
    save_dir = os.path.join(save_dir, "saved_plots")
    os.makedirs(save_dir, exist_ok=True)
    ic_np = initial_condition.detach().cpu().numpy()
    for t in time_indices:
        truth_np = truth_seq[0, ..., t].detach().cpu().numpy()
        pred_np = pred_seq[0, ..., t].detach().cpu().numpy()
        error_np = pred_np - truth_np
        save_path = os.path.join(
            save_dir,
            f"{model_name}_seed{seed}_prediction_t{t + 1}.npz",
        )
        np.savez(
            save_path,
            initial_condition=ic_np,
            truth_seq_t=truth_np,
            pred_seq_t=pred_np,
            error_seq_t=error_np,
        )
        print(f"Saved ground truth and predictions to {save_path}")


def compute_save_energy_spectra(
    truth_seq,
    pred_seq,
    time_indices,
    save_dir,
    model_name,
    seed,
    Lx=2 * math.pi,
    Ly=2 * math.pi,
    file_suffix="",
):
    """Save energy and enstrophy spectra at selected timesteps.

    Args:
        truth_seq: (B, H, W, C, T) — channels: (ux, uy, p)
        pred_seq:  (B, H, W, C, T) — aligned with truth_seq
        time_indices: list of 0-based rollout step indices
    """
    save_dir = os.path.join(save_dir, "saved_plots")
    os.makedirs(save_dir, exist_ok=True)
    B = truth_seq.shape[0]
    for t in time_indices:
        pred_frame = pred_seq[..., t].detach().cpu().numpy()  # (B, H, W, C)
        truth_frame = truth_seq[..., t].detach().cpu().numpy()

        H, W = truth_frame.shape[1], truth_frame.shape[2]
        valid_mask = slice(1, min(H, W) // 2)
        Ek_true_list, Zk_true_list = [], []
        Ek_pred_list, Zk_pred_list = [], []
        for b in range(B):
            ux_pred, uy_pred = pred_frame[b, ..., 0], pred_frame[b, ..., 1]
            ux_true, uy_true = truth_frame[b, ..., 0], truth_frame[b, ..., 1]

            k_bins, Ek_pred, Zk_pred = compute_spectra(ux_pred, uy_pred, Lx, Ly)
            _, Ek_true, Zk_true = compute_spectra(ux_true, uy_true, Lx, Ly)
            Ek_true_list.append(Ek_true[valid_mask])
            Zk_true_list.append(Zk_true[valid_mask])
            Ek_pred_list.append(Ek_pred[valid_mask])
            Zk_pred_list.append(Zk_pred[valid_mask])

        save_path = os.path.join(
            save_dir,
            append_file_suffix(f"{model_name}_seed{seed}_spectra_t{t + 1}", file_suffix) + ".npz",
        )
        np.savez(
            save_path,
            k_bins=k_bins[valid_mask],
            Ek_true=np.stack(Ek_true_list, axis=0),
            Ek_pred=np.stack(Ek_pred_list, axis=0),
            Zk_true=np.stack(Zk_true_list, axis=0),
            Zk_pred=np.stack(Zk_pred_list, axis=0),
        )
        print(f"Saved energy/enstrophy spectra to {save_path}")


def compute_rollout_statistics(
    initial_condition,
    pred_seq,
    truth_seq,
    norm_mean,
    norm_std,
    device,
    forcing,
    physics,
    constraint_domain_lengths=None,
    velocity_channels=(0, 1),
    energy_balance_params=None,
    mode_cutoff=None,
):
    """Compute all rollout metrics for a given prediction set."""
    l2_per_step = compute_per_step_l2(pred_seq, truth_seq)
    blowup_step = compute_blowup_step(l2_per_step)
    l2_mean = metric_mean(l2_per_step, axis=0)
    l2_std = metric_std(l2_per_step, axis=0)

    pred_phys = denormalise(pred_seq, norm_mean, norm_std, device)
    truth_full = torch.cat([initial_condition.unsqueeze(-1), truth_seq], dim=-1)
    truth_phys = denormalise(truth_full, norm_mean, norm_std, device)

    # The relative residuals divide by scales measured on the ground-truth
    # trajectories, so predicted and true residuals share one denominator. The
    # vorticity residual is relative to the forcing instead, and its one-channel
    # state carries no velocity to measure the velocity scales on.
    has_velocity = carries_velocity(physics)
    scales = None
    if has_velocity:
        scales = residual_scales_from_truth(
            truth_phys.permute(0, 3, 1, 2, 4), physics, physics.dt * (truth_phys.shape[-1] - 1)
        )
    pde_pred = compute_PDE_loss_per_step(pred_phys, forcing, physics, scales)
    pde_truth = compute_PDE_loss_per_step(truth_phys, forcing, physics, scales)

    # Divergence is a diagnostic of every model, constrained or not: the
    # unconstrained baseline is what the projected models are compared against.
    # It is a velocity diagnostic, so the vorticity formulation reports none.
    div_domain_lengths = constraint_domain_lengths
    if div_domain_lengths is None:
        div_domain_lengths = (float(physics.domain_length), float(physics.domain_length))
    pred_div_max = None
    truth_div_max = None
    pred_domain_mean_velocity = None
    truth_domain_mean_velocity = None
    if has_velocity:
        pred_div_max = compute_div_max_per_step(
            pred_phys[..., 1:],
            velocity_channels=velocity_channels,
            domain_lengths=div_domain_lengths,
        )
        truth_div_max = compute_div_max_per_step(
            truth_phys[..., 1:],
            velocity_channels=velocity_channels,
            domain_lengths=div_domain_lengths,
        )

        pred_domain_mean_velocity = compute_domain_mean_velocity_per_step(
            pred_phys[..., 1:],
            velocity_channels=velocity_channels,
        )
        truth_domain_mean_velocity = compute_domain_mean_velocity_per_step(
            truth_phys[..., 1:],
            velocity_channels=velocity_channels,
        )

    # The band above the model's spectral-mode cutoff, where the rollouts fail.
    pred_high_band = None
    truth_high_band = None
    if has_velocity and mode_cutoff is not None:
        pred_high_band = compute_high_band_fraction_per_step(
            pred_phys[..., 1:], velocity_channels, mode_cutoff
        )
        truth_high_band = compute_high_band_fraction_per_step(
            truth_phys[..., 1:], velocity_channels, mode_cutoff
        )

    pred_energy_balance = None
    truth_energy_balance = None
    if energy_balance_params is not None:
        pred_energy_balance = compute_energy_balance_per_step(
            pred_phys,
            velocity_channels=velocity_channels,
            forcing=forcing,
            viscosity=physics.nu,
            friction=physics.alpha,
            dt=physics.dt,
            domain_lengths=energy_balance_params["domain_lengths"],
        )
        truth_energy_balance = compute_energy_balance_per_step(
            truth_phys,
            velocity_channels=velocity_channels,
            forcing=forcing,
            viscosity=physics.nu,
            friction=physics.alpha,
            dt=physics.dt,
            domain_lengths=energy_balance_params["domain_lengths"],
        )

    counted = {"l2": l2_per_step}
    counted.update({f"pred_{name}": values for name, values in pde_pred.items()})
    counted.update({f"truth_{name}": values for name, values in pde_truth.items()})
    if pred_div_max is not None:
        counted["pred_div_max"] = pred_div_max
        counted["truth_div_max"] = truth_div_max
        counted.update(
            {f"pred_domain_{name}": values for name, values in pred_domain_mean_velocity.items()}
        )
        counted.update(
            {f"truth_domain_{name}": values for name, values in truth_domain_mean_velocity.items()}
        )
    if pred_energy_balance is not None:
        counted.update({f"pred_energy_{name}": v for name, v in pred_energy_balance.items()})
        counted.update({f"truth_energy_{name}": v for name, v in truth_energy_balance.items()})
    non_finite_counts = count_non_finite(counted)

    return {
        "initial_condition": initial_condition,
        "pred_seq": pred_seq,
        "truth_seq": truth_seq,
        "l2_per_step": l2_per_step,
        "l2_mean": l2_mean,
        "l2_std": l2_std,
        "blowup_step": blowup_step,
        "pred_phys": pred_phys,
        "truth_phys": truth_phys,
        "residual_scales": scales,
        "pde_pred": pde_pred,
        "pde_truth": pde_truth,
        "pred_aligned": pred_phys[..., 1:],
        "truth_aligned": truth_phys[..., 1:],
        "pred_div_max": pred_div_max,
        "truth_div_max": truth_div_max,
        "pred_domain_mean_velocity": pred_domain_mean_velocity,
        "truth_domain_mean_velocity": truth_domain_mean_velocity,
        "pred_energy_balance": pred_energy_balance,
        "truth_energy_balance": truth_energy_balance,
        "pred_high_band": pred_high_band,
        "truth_high_band": truth_high_band,
        "mode_cutoff": mode_cutoff,
        "n_trajectories": int(l2_per_step.shape[0]),
        "non_finite_counts": non_finite_counts,
        "physics": physics,
    }


def print_rollout_statistics(stats, rollout_label=None):
    """Print rollout metrics in the current human-readable format."""
    label_suffix = f" ({rollout_label})" if rollout_label else ""

    n_steps = stats["l2_per_step"].shape[1]
    print(
        f"\n── Rollout summary{label_suffix} ──\n"
        f"  {stats['n_trajectories']} trajectories x {n_steps} steps; "
        f"means below are over the trajectories at each step."
    )
    blowup = stats["blowup_step"]
    never = int(np.sum(~np.isfinite(blowup)))
    print(
        f"  Blow-up step (rel-L2 > {BOUNDED_L2} or non-finite): "
        f"median {blowup_quantile(blowup, 0.5)}, "
        f"IQR [{blowup_quantile(blowup, 0.25)}, {blowup_quantile(blowup, 0.75)}]; "
        f"{never}/{len(blowup)} trajectories never blew up."
    )
    if stats["residual_scales"] is not None:
        print(
            "  Residual reference scales (ground truth): "
            f"cont={stats['residual_scales'].cont:.6e}, "
            f"momx={stats['residual_scales'].momx:.6e}, "
            f"momy={stats['residual_scales'].momy:.6e}"
        )
    non_finite = stats["non_finite_counts"]
    if any(non_finite.values()):
        print(
            f"  NON-FINITE entries ({sum(non_finite.values())} in total): "
            + ", ".join(f"{name}={count}" for name, count in non_finite.items() if count)
        )
    else:
        print(f"  Non-finite entries: none, across {len(non_finite)} metric arrays")

    print(f"\n── Per-step relative L2{label_suffix} ──")
    for t, (l2_m, l2_s) in enumerate(zip(stats["l2_mean"], stats["l2_std"], strict=False)):
        print(f"  Step {t:3d} -> {t + 1:3d}: L2 = {l2_m:.6f} +/- {l2_s:.6f}")

    print(f"\n── Per-step PDE residuals (predicted trajectory){label_suffix} ──")
    pde_pred = stats["pde_pred"]
    for t in range(stats["l2_per_step"].shape[1]):
        print(
            f"  Step {t:3d} -> {t + 1:3d}: "
            f"cont_rel={metric_mean(pde_pred['loss_cont_rel'][:, t]):.4e} +/- {metric_std(pde_pred['loss_cont_rel'][:, t]):.4e}, "
            f"momx_rel={metric_mean(pde_pred['loss_momx_rel'][:, t]):.4e} +/- {metric_std(pde_pred['loss_momx_rel'][:, t]):.4e}, "
            f"momy_rel={metric_mean(pde_pred['loss_momy_rel'][:, t]):.4e} +/- {metric_std(pde_pred['loss_momy_rel'][:, t]):.4e}"
        )

    print(f"\n── Per-step PDE residuals (ground truth trajectory){label_suffix} ──")
    pde_truth = stats["pde_truth"]
    for t in range(stats["l2_per_step"].shape[1]):
        print(
            f"  Step {t:3d} -> {t + 1:3d}: "
            f"cont_rel={metric_mean(pde_truth['loss_cont_rel'][:, t]):.4e} +/- {metric_std(pde_truth['loss_cont_rel'][:, t]):.4e}, "
            f"momx_rel={metric_mean(pde_truth['loss_momx_rel'][:, t]):.4e} +/- {metric_std(pde_truth['loss_momx_rel'][:, t]):.4e}, "
            f"momy_rel={metric_mean(pde_truth['loss_momy_rel'][:, t]):.4e} +/- {metric_std(pde_truth['loss_momy_rel'][:, t]):.4e}"
        )

    if not carries_velocity(stats["physics"]):
        print(
            f"\n── Velocity diagnostics{label_suffix} ── "
            f"not defined for the {stats['physics'].formulation} formulation; skipped."
        )
        # Everything below is a velocity diagnostic, the energy balance included:
        # it is measured only when the output projector enforces it, and that
        # projector needs a velocity state.
        return

    print(f"\n── Per-step max |div u|{label_suffix} ──")
    for t in range(stats["l2_per_step"].shape[1]):
        print(
            f"  Step {t:3d} -> {t + 1:3d}: "
            f"pred={metric_mean(stats['pred_div_max'][:, t]):.4e} +/- "
            f"{metric_std(stats['pred_div_max'][:, t]):.4e}, "
            f"truth={metric_mean(stats['truth_div_max'][:, t]):.4e} +/- "
            f"{metric_std(stats['truth_div_max'][:, t]):.4e}"
        )

    print(f"\n── Per-step domain-mean velocity (predicted trajectory){label_suffix} ──")
    pred_domain_mean = stats["pred_domain_mean_velocity"]
    for t in range(stats["l2_per_step"].shape[1]):
        print(
            f"  Step {t:3d} -> {t + 1:3d}: "
            f"<ux>={metric_mean(pred_domain_mean['mean_ux'][:, t]):.4e} +/- "
            f"{metric_std(pred_domain_mean['mean_ux'][:, t]):.4e}, "
            f"<uy>={metric_mean(pred_domain_mean['mean_uy'][:, t]):.4e} +/- "
            f"{metric_std(pred_domain_mean['mean_uy'][:, t]):.4e}, "
            f"|<u>|={metric_mean(pred_domain_mean['mean_speed'][:, t]):.4e} +/- "
            f"{metric_std(pred_domain_mean['mean_speed'][:, t]):.4e}"
        )

    print(f"\n── Per-step domain-mean velocity (ground truth trajectory){label_suffix} ──")
    truth_domain_mean = stats["truth_domain_mean_velocity"]
    for t in range(stats["l2_per_step"].shape[1]):
        print(
            f"  Step {t:3d} -> {t + 1:3d}: "
            f"<ux>={metric_mean(truth_domain_mean['mean_ux'][:, t]):.4e} +/- "
            f"{metric_std(truth_domain_mean['mean_ux'][:, t]):.4e}, "
            f"<uy>={metric_mean(truth_domain_mean['mean_uy'][:, t]):.4e} +/- "
            f"{metric_std(truth_domain_mean['mean_uy'][:, t]):.4e}, "
            f"|<u>|={metric_mean(truth_domain_mean['mean_speed'][:, t]):.4e} +/- "
            f"{metric_std(truth_domain_mean['mean_speed'][:, t]):.4e}"
        )

    if stats["pred_energy_balance"] is not None:
        print(f"\n── Per-step energy balance (predicted trajectory){label_suffix} ──")
        pred_energy_balance = stats["pred_energy_balance"]
        for t in range(stats["l2_per_step"].shape[1]):
            print(
                f"  Step {t:3d} -> {t + 1:3d}: "
                f"res={metric_mean(pred_energy_balance['balance_residual'][:, t]):.4e} +/- "
                f"{metric_std(pred_energy_balance['balance_residual'][:, t]):.4e}, "
                f"|res|={metric_mean(pred_energy_balance['balance_abs_residual'][:, t]):.4e} +/- "
                f"{metric_std(pred_energy_balance['balance_abs_residual'][:, t]):.4e}, "
                f"rel={metric_mean(pred_energy_balance['balance_rel_residual'][:, t]):.4e} +/- "
                f"{metric_std(pred_energy_balance['balance_rel_residual'][:, t]):.4e}"
            )

        print(f"\n── Per-step energy balance (ground truth trajectory){label_suffix} ──")
        truth_energy_balance = stats["truth_energy_balance"]
        for t in range(stats["l2_per_step"].shape[1]):
            print(
                f"  Step {t:3d} -> {t + 1:3d}: "
                f"res={metric_mean(truth_energy_balance['balance_residual'][:, t]):.4e} +/- "
                f"{metric_std(truth_energy_balance['balance_residual'][:, t]):.4e}, "
                f"|res|={metric_mean(truth_energy_balance['balance_abs_residual'][:, t]):.4e} +/- "
                f"{metric_std(truth_energy_balance['balance_abs_residual'][:, t]):.4e}, "
                f"rel={metric_mean(truth_energy_balance['balance_rel_residual'][:, t]):.4e} +/- "
                f"{metric_std(truth_energy_balance['balance_rel_residual'][:, t]):.4e}"
            )


def save_rollout_artifacts(stats, save_dir, experiment_name, seed, file_suffix="", summary=None):
    """Save CSV/NPZ/diagnostic artifacts for a rollout result set."""
    results_dir = os.path.join(save_dir, "evaluation_metrics")
    os.makedirs(results_dir, exist_ok=True)
    T_data = stats["l2_per_step"].shape[1]

    df_per_step = pd.DataFrame(
        {
            "step": np.arange(1, T_data + 1),
            "l2_mean": stats["l2_mean"],
            "l2_std": stats["l2_std"],
            "pred_cont_abs_mean": metric_mean(stats["pde_pred"]["loss_cont"], axis=0),
            "pred_cont_abs_std": metric_std(stats["pde_pred"]["loss_cont"], axis=0),
            "pred_momx_abs_mean": metric_mean(stats["pde_pred"]["loss_momx"], axis=0),
            "pred_momx_abs_std": metric_std(stats["pde_pred"]["loss_momx"], axis=0),
            "pred_momy_abs_mean": metric_mean(stats["pde_pred"]["loss_momy"], axis=0),
            "pred_momy_abs_std": metric_std(stats["pde_pred"]["loss_momy"], axis=0),
            "pred_cont_rel_mean": metric_mean(stats["pde_pred"]["loss_cont_rel"], axis=0),
            "pred_cont_rel_std": metric_std(stats["pde_pred"]["loss_cont_rel"], axis=0),
            "pred_momx_rel_mean": metric_mean(stats["pde_pred"]["loss_momx_rel"], axis=0),
            "pred_momx_rel_std": metric_std(stats["pde_pred"]["loss_momx_rel"], axis=0),
            "pred_momy_rel_mean": metric_mean(stats["pde_pred"]["loss_momy_rel"], axis=0),
            "pred_momy_rel_std": metric_std(stats["pde_pred"]["loss_momy_rel"], axis=0),
            "truth_cont_abs_mean": metric_mean(stats["pde_truth"]["loss_cont"], axis=0),
            "truth_cont_abs_std": metric_std(stats["pde_truth"]["loss_cont"], axis=0),
            "truth_momx_abs_mean": metric_mean(stats["pde_truth"]["loss_momx"], axis=0),
            "truth_momx_abs_std": metric_std(stats["pde_truth"]["loss_momx"], axis=0),
            "truth_momy_abs_mean": metric_mean(stats["pde_truth"]["loss_momy"], axis=0),
            "truth_momy_abs_std": metric_std(stats["pde_truth"]["loss_momy"], axis=0),
            "truth_cont_rel_mean": metric_mean(stats["pde_truth"]["loss_cont_rel"], axis=0),
            "truth_cont_rel_std": metric_std(stats["pde_truth"]["loss_cont_rel"], axis=0),
            "truth_momx_rel_mean": metric_mean(stats["pde_truth"]["loss_momx_rel"], axis=0),
            "truth_momx_rel_std": metric_std(stats["pde_truth"]["loss_momx_rel"], axis=0),
            "truth_momy_rel_mean": metric_mean(stats["pde_truth"]["loss_momy_rel"], axis=0),
            "truth_momy_rel_std": metric_std(stats["pde_truth"]["loss_momy_rel"], axis=0),
        }
    )
    has_velocity = carries_velocity(stats["physics"])
    if has_velocity:
        df_per_step["pred_div_max_mean"] = metric_mean(stats["pred_div_max"], axis=0)
        df_per_step["pred_div_max_std"] = metric_std(stats["pred_div_max"], axis=0)
        df_per_step["truth_div_max_mean"] = metric_mean(stats["truth_div_max"], axis=0)
        df_per_step["truth_div_max_std"] = metric_std(stats["truth_div_max"], axis=0)
        pred_domain_mean = stats["pred_domain_mean_velocity"]
        truth_domain_mean = stats["truth_domain_mean_velocity"]
        df_per_step["pred_domain_mean_ux_mean"] = metric_mean(pred_domain_mean["mean_ux"], axis=0)
        df_per_step["pred_domain_mean_ux_std"] = metric_std(pred_domain_mean["mean_ux"], axis=0)
        df_per_step["pred_domain_mean_uy_mean"] = metric_mean(pred_domain_mean["mean_uy"], axis=0)
        df_per_step["pred_domain_mean_uy_std"] = metric_std(pred_domain_mean["mean_uy"], axis=0)
        df_per_step["pred_domain_mean_speed_mean"] = metric_mean(
            pred_domain_mean["mean_speed"], axis=0
        )
        df_per_step["pred_domain_mean_speed_std"] = metric_std(
            pred_domain_mean["mean_speed"], axis=0
        )
        df_per_step["truth_domain_mean_ux_mean"] = metric_mean(truth_domain_mean["mean_ux"], axis=0)
        df_per_step["truth_domain_mean_ux_std"] = metric_std(truth_domain_mean["mean_ux"], axis=0)
        df_per_step["truth_domain_mean_uy_mean"] = metric_mean(truth_domain_mean["mean_uy"], axis=0)
        df_per_step["truth_domain_mean_uy_std"] = metric_std(truth_domain_mean["mean_uy"], axis=0)
        if stats["pred_high_band"] is not None:
            df_per_step["pred_frac_energy_above_cutoff_mean"] = metric_mean(
                stats["pred_high_band"], axis=0
            )
            df_per_step["pred_frac_energy_above_cutoff_std"] = metric_std(
                stats["pred_high_band"], axis=0
            )
            df_per_step["truth_frac_energy_above_cutoff_mean"] = metric_mean(
                stats["truth_high_band"], axis=0
            )
            df_per_step["truth_frac_energy_above_cutoff_std"] = metric_std(
                stats["truth_high_band"], axis=0
            )
        df_per_step["truth_domain_mean_speed_mean"] = metric_mean(
            truth_domain_mean["mean_speed"], axis=0
        )
        df_per_step["truth_domain_mean_speed_std"] = metric_std(
            truth_domain_mean["mean_speed"], axis=0
        )
    if stats["pred_energy_balance"] is not None:
        pred_energy_balance = stats["pred_energy_balance"]
        truth_energy_balance = stats["truth_energy_balance"]
        df_per_step["pred_energy_balance_res_mean"] = metric_mean(
            pred_energy_balance["balance_residual"], axis=0
        )
        df_per_step["pred_energy_balance_res_std"] = metric_std(
            pred_energy_balance["balance_residual"], axis=0
        )
        df_per_step["pred_energy_balance_abs_mean"] = metric_mean(
            pred_energy_balance["balance_abs_residual"], axis=0
        )
        df_per_step["pred_energy_balance_abs_std"] = metric_std(
            pred_energy_balance["balance_abs_residual"], axis=0
        )
        df_per_step["pred_energy_balance_rel_mean"] = metric_mean(
            pred_energy_balance["balance_rel_residual"], axis=0
        )
        df_per_step["pred_energy_balance_rel_std"] = metric_std(
            pred_energy_balance["balance_rel_residual"], axis=0
        )
        df_per_step["truth_energy_balance_res_mean"] = metric_mean(
            truth_energy_balance["balance_residual"], axis=0
        )
        df_per_step["truth_energy_balance_res_std"] = metric_std(
            truth_energy_balance["balance_residual"], axis=0
        )
        df_per_step["truth_energy_balance_abs_mean"] = metric_mean(
            truth_energy_balance["balance_abs_residual"], axis=0
        )
        df_per_step["truth_energy_balance_abs_std"] = metric_std(
            truth_energy_balance["balance_abs_residual"], axis=0
        )
        df_per_step["truth_energy_balance_rel_mean"] = metric_mean(
            truth_energy_balance["balance_rel_residual"], axis=0
        )
        df_per_step["truth_energy_balance_rel_std"] = metric_std(
            truth_energy_balance["balance_rel_residual"], axis=0
        )
        # ``energy`` carries the initial condition at index 0; drop it to align
        # with the predicted steps 1..T that index the rest of this frame.
        df_per_step["pred_energy_mean"] = metric_mean(pred_energy_balance["energy"][:, 1:], axis=0)
        df_per_step["pred_energy_std"] = metric_std(pred_energy_balance["energy"][:, 1:], axis=0)
        df_per_step["truth_energy_mean"] = metric_mean(
            truth_energy_balance["energy"][:, 1:], axis=0
        )
        df_per_step["truth_energy_std"] = metric_std(truth_energy_balance["energy"][:, 1:], axis=0)
    csv_path = os.path.join(
        results_dir,
        append_file_suffix(f"{experiment_name}_seed{seed}_per_step_metrics", file_suffix) + ".csv",
    )
    df_per_step.to_csv(csv_path, index=False)
    print(f"\nPer-step metrics saved to {csv_path}")

    per_sample_path = os.path.join(
        results_dir,
        append_file_suffix(f"{experiment_name}_seed{seed}_per_sample_metrics", file_suffix)
        + ".npz",
    )
    per_sample_payload = {
        "step": np.arange(1, T_data + 1),
        "l2": stats["l2_per_step"],
        "pred_cont_abs": stats["pde_pred"]["loss_cont"],
        "pred_momx_abs": stats["pde_pred"]["loss_momx"],
        "pred_momy_abs": stats["pde_pred"]["loss_momy"],
        "pred_cont_rel": stats["pde_pred"]["loss_cont_rel"],
        "pred_momx_rel": stats["pde_pred"]["loss_momx_rel"],
        "pred_momy_rel": stats["pde_pred"]["loss_momy_rel"],
        "truth_cont_abs": stats["pde_truth"]["loss_cont"],
        "truth_momx_abs": stats["pde_truth"]["loss_momx"],
        "truth_momy_abs": stats["pde_truth"]["loss_momy"],
        "truth_cont_rel": stats["pde_truth"]["loss_cont_rel"],
        "truth_momx_rel": stats["pde_truth"]["loss_momx_rel"],
        "truth_momy_rel": stats["pde_truth"]["loss_momy_rel"],
    }
    per_sample_payload["blowup_step"] = stats["blowup_step"]
    if stats["pred_high_band"] is not None:
        per_sample_payload["pred_frac_energy_above_cutoff"] = stats["pred_high_band"]
        per_sample_payload["truth_frac_energy_above_cutoff"] = stats["truth_high_band"]
    if has_velocity:
        per_sample_payload["pred_div_max"] = stats["pred_div_max"]
        per_sample_payload["truth_div_max"] = stats["truth_div_max"]
        per_sample_payload["pred_domain_mean_ux"] = pred_domain_mean["mean_ux"]
        per_sample_payload["pred_domain_mean_uy"] = pred_domain_mean["mean_uy"]
        per_sample_payload["pred_domain_mean_speed"] = pred_domain_mean["mean_speed"]
        per_sample_payload["truth_domain_mean_ux"] = truth_domain_mean["mean_ux"]
        per_sample_payload["truth_domain_mean_uy"] = truth_domain_mean["mean_uy"]
        per_sample_payload["truth_domain_mean_speed"] = truth_domain_mean["mean_speed"]
    if stats["pred_energy_balance"] is not None:
        pred_energy_balance = stats["pred_energy_balance"]
        truth_energy_balance = stats["truth_energy_balance"]
        per_sample_payload["pred_energy"] = pred_energy_balance["energy"]
        per_sample_payload["pred_energy_dissipation"] = pred_energy_balance["dissipation"]
        per_sample_payload["pred_energy_injection"] = pred_energy_balance["injection"]
        per_sample_payload["pred_energy_balance_residual"] = pred_energy_balance["balance_residual"]
        per_sample_payload["pred_energy_balance_abs_residual"] = pred_energy_balance[
            "balance_abs_residual"
        ]
        per_sample_payload["pred_energy_balance_rel_residual"] = pred_energy_balance[
            "balance_rel_residual"
        ]
        per_sample_payload["truth_energy"] = truth_energy_balance["energy"]
        per_sample_payload["truth_energy_dissipation"] = truth_energy_balance["dissipation"]
        per_sample_payload["truth_energy_injection"] = truth_energy_balance["injection"]
        per_sample_payload["truth_energy_balance_residual"] = truth_energy_balance[
            "balance_residual"
        ]
        per_sample_payload["truth_energy_balance_abs_residual"] = truth_energy_balance[
            "balance_abs_residual"
        ]
        per_sample_payload["truth_energy_balance_rel_residual"] = truth_energy_balance[
            "balance_rel_residual"
        ]
    np.savez(per_sample_path, **per_sample_payload)
    print(f"Per-sample metrics saved to {per_sample_path}")

    blowup = stats["blowup_step"]
    run_summary = {
        "n_trajectories": stats["n_trajectories"],
        "n_rollout_steps": T_data,
        # The stability column: where a trajectory first loses all skill, infinite for one
        # that keeps it to the end of the horizon.
        "blowup_step_median": blowup_quantile(blowup, 0.5),
        "blowup_step_q1": blowup_quantile(blowup, 0.25),
        "blowup_step_q3": blowup_quantile(blowup, 0.75),
        "blowup_never_frac": float(np.mean(~np.isfinite(blowup))),
        "mode_cutoff": (stats["mode_cutoff"] if stats["mode_cutoff"] is not None else float("nan")),
        "non_finite_total": sum(stats["non_finite_counts"].values()),
        # Only the offending arrays are named, so the row stays the same width run to run.
        **{
            f"non_finite_{name}": count
            for name, count in stats["non_finite_counts"].items()
            if count
        },
    }
    if stats["residual_scales"] is not None:
        run_summary["cont_ref"] = stats["residual_scales"].cont
        run_summary["momx_ref"] = stats["residual_scales"].momx
        run_summary["momy_ref"] = stats["residual_scales"].momy
    if summary:
        run_summary.update(summary)

    # The spectra are defined on the velocity, which the vorticity formulation
    # carries only implicitly; recover it once for both spectral diagnostics.
    domain_length = float(stats["physics"].domain_length)
    if has_velocity:
        spectra_seqs = None
        spectra_truth = stats["truth_aligned"]
        spectra_pred = stats["pred_aligned"]
    else:
        spectra_truth = velocity_sequence_from_vorticity(stats["truth_aligned"], domain_length)
        spectra_pred = velocity_sequence_from_vorticity(stats["pred_aligned"], domain_length)
        spectra_seqs = (spectra_truth, spectra_pred)

    evaluate_model(
        stats["truth_aligned"],
        stats["pred_aligned"],
        experiment_name,
        seed=seed,
        save_dir=save_dir,
        save_csv=True,
        file_suffix=file_suffix,
        extra=run_summary,
        spectra_seqs=spectra_seqs,
    )

    spectra_indices = (
        SPECTRA_TIME_INDICES if SPECTRA_TIME_INDICES is not None else list(range(T_data))
    )
    compute_save_energy_spectra(
        spectra_truth,
        spectra_pred,
        spectra_indices,
        save_dir,
        experiment_name,
        seed=seed,
        file_suffix=file_suffix,
    )
    compute_save_vorticity(
        stats["pred_phys"],
        stats["truth_phys"],
        save_dir,
        experiment_name,
        seed=seed,
        file_suffix=file_suffix,
        state_is_vorticity=not has_velocity,
    )


def print_base_vs_tto_summary(base_stats, tto_stats):
    """Print a concise comparison between the base and adapted rollouts."""
    print("\n── Base vs TTO Summary ──")
    print(
        f"  Mean rollout L2: "
        f"base={metric_mean(base_stats['l2_per_step']):.6f}, "
        f"tto={metric_mean(tto_stats['l2_per_step']):.6f}"
    )
    print(
        f"  Mean cont_rel: "
        f"base={metric_mean(base_stats['pde_pred']['loss_cont_rel']):.4e}, "
        f"tto={metric_mean(tto_stats['pde_pred']['loss_cont_rel']):.4e}"
    )
    print(
        f"  Mean momx_rel: "
        f"base={metric_mean(base_stats['pde_pred']['loss_momx_rel']):.4e}, "
        f"tto={metric_mean(tto_stats['pde_pred']['loss_momx_rel']):.4e}"
    )
    print(
        f"  Mean momy_rel: "
        f"base={metric_mean(base_stats['pde_pred']['loss_momy_rel']):.4e}, "
        f"tto={metric_mean(tto_stats['pde_pred']['loss_momy_rel']):.4e}"
    )


def main():
    parser = ArgumentParser(description="Evaluate 2D operator autoregressively")
    parser.add_argument("--config_path", type=str, help="Path to the configuration file")
    parser.add_argument(
        "--seed",
        "--test_seed",
        dest="seed",
        type=int,
        default=42,
        help="Seed of the run being evaluated; names the checkpoint and the outputs",
    )
    parser.add_argument(
        "--checkpoint",
        choices=("best", "last"),
        default="best",
        help="Evaluate the validation-selected checkpoint (default) or the last one",
    )
    parser.add_argument(
        "--max_test_samples",
        type=int,
        default=None,
        help="Evaluate only the first N test trajectories, for a smoke run or one figure",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Override train.save_dir, so one config can serve a whole seed campaign",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate (e.g. test, test_time); names the outputs when not test",
    )
    parser.add_argument(
        "--rollout_steps",
        type=int,
        default=None,
        help="Rollout horizon, overriding config['data']['nt']; names the outputs when given",
    )
    parser.add_argument(
        "--band_cutoff",
        type=int,
        default=None,
        help=(
            "Wavenumber the high-band diagnostic measures above, overriding the model's own "
            "modes1. Two models with different spectral cutoffs are otherwise measured over "
            "different bands, so a mechanism that widens the cutoff cannot be compared with "
            "one that does not"
        ),
    )
    args = parser.parse_args()

    seed_everything(args.seed)

    with open(args.config_path) as stream:
        config = yaml.load(stream, yaml.FullLoader)
    if args.save_dir is not None:
        config["train"]["save_dir"] = args.save_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_config = config["data"]
    model_cfg = config["model"]
    tto_cfg = config.get("tto", {})
    output_constraint_cfg = model_cfg.get("output_constraint", {})
    output_constraint_enabled = bool(output_constraint_cfg.get("enabled", False))
    energy_balance_enabled = bool(
        output_constraint_cfg.get("energy_balance", {}).get("enabled", False)
    )
    physics = physics_from_config(data_config)

    if output_constraint_enabled and tto_cfg.get("enabled", False):
        raise ValueError(
            "output_constraint and tto.enabled cannot be used in the same evaluation run"
        )

    # Experiment name from config filename (e.g. "FNO", "PINO")
    experiment_name = os.path.splitext(os.path.basename(args.config_path))[0]

    # ── Data ────────────────────────────────────────────────────────
    dataset_filename = data_config.get("filename", "kolmogorov_dataset.npz")
    check_split_available(data_config["datapath"], dataset_filename, args.split)

    test_set = NSLoader2D(
        datapath=data_config["datapath"],
        state=args.split,
        train=False,
        normalizer_path=data_config.get("normalizer_path", None),
        velocity_channels=data_config.get("velocity_channels", (0, 1)),
        filename=dataset_filename,
    )
    norm_mean = test_set.mean  # (C, H, W) — kept before permutation in __init__
    norm_std = test_set.std

    rollout_steps = args.rollout_steps if args.rollout_steps is not None else data_config["nt"]
    variant_suffix = evaluation_variant_suffix(args.split, args.rollout_steps, args.band_cutoff)
    check_rollout_horizon(data_config["datapath"], dataset_filename, args.split, rollout_steps)
    test_set.transform_rollout(T=rollout_steps)
    eval_set = limit_trajectories(test_set, args.max_test_samples)
    test_loader = DataLoader(
        eval_set,
        batch_size=config["train"]["batchsize"],
        shuffle=False,
        num_workers=config["train"].get("num_workers", 1),
    )
    tto_loader = None
    if tto_cfg.get("enabled", False):
        tto_loader = DataLoader(
            eval_set,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )
    S_data = test_set.S
    T_data = test_set.T
    grid = torch2dgrid_2d(
        S_data[0],
        S_data[1],
        form=data_config["grid_form"],
        device=device,
        dtype=torch.float32,
    )
    forcing = build_forcing_for_data(physics, S_data, device=device)
    constraint_domain_lengths = (
        tuple(
            float(length)
            for length in output_constraint_cfg.get(
                "domain_lengths",
                [physics.domain_length, physics.domain_length],
            )
        )
        if output_constraint_enabled
        else None
    )
    # The energy balance reads its physics from config['data']; only the geometry
    # of the projector is carried here. Energy and its balance residual are
    # diagnostics of every velocity model, not only of the one whose projector
    # enforces them, so the baseline has something for the projection to beat.
    energy_balance_params = (
        {
            "domain_lengths": constraint_domain_lengths
            or (float(physics.domain_length), float(physics.domain_length))
        }
        if physics.formulation == "velocity"
        else None
    )

    # ── Model ───────────────────────────────────────────────────────
    use_residual = model_cfg.get("residual", False)
    # The band diagnostic is defined against the model's own spectral-mode cutoff: the
    # shells no spectral layer can write into. A model without one reports no band.
    modes = model_cfg.get("modes1")
    mode_cutoff = int(min(modes)) if modes else None
    if args.band_cutoff is not None:
        mode_cutoff = int(args.band_cutoff)
    check_model_channels(physics, model_cfg.get("out_dim", 1))
    model_name = model_cfg.get("name", "fno2d").lower()
    if output_constraint_enabled and model_name != "fno2d":
        raise ValueError('output_constraint is only supported for model.name == "fno2d"')
    if model_name == "fno2d":
        model = FNO2d(
            in_dim=model_cfg.get("in_dim", 3),
            out_dim=model_cfg.get("out_dim", 1),
            modes1=model_cfg["modes1"],
            modes2=model_cfg["modes2"],
            fc_dim=model_cfg["fc_dim"],
            layers=model_cfg["layers"],
            act=model_cfg["act"],
            output_constraint=output_constraint_cfg,
            physics=physics,
        ).to(device)
    else:
        raise ValueError(f"Model {model_name} not supported")
    if output_constraint_enabled:
        model.set_output_normalizer(norm_mean, norm_std)
        if energy_balance_enabled:
            # Match training-time initialisation so checkpoints that include
            # the forcing buffer load without state-dict mismatches.
            model.set_energy_forcing(forcing.squeeze(-1))

    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    # ── Load checkpoint ─────────────────────────────────────────────
    save_dir = config.get("train", {}).get("save_dir")
    save_name = config.get("train", {}).get("save_name")
    ckpt_path = resolve_checkpoint_path(save_dir, save_name, args.seed, which=args.checkpoint)
    checkpoint_epoch = None
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        checkpoint_epoch = ckpt.get("epoch")
        reference_state = {key: value.detach().clone() for key, value in ckpt["model"].items()}
        model_state = model.state_dict()
        allowed_missing_keys = {"output_projector.forcing_hat"}
        patched_state = dict(reference_state)
        for key in allowed_missing_keys:
            if key in model_state and key not in patched_state:
                patched_state[key] = model_state[key]

        load_result = model.load_state_dict(patched_state, strict=False)
        remaining_missing = set(load_result.missing_keys) - allowed_missing_keys
        if remaining_missing or load_result.unexpected_keys:
            raise RuntimeError(
                "Error loading checkpoint state_dict for FNO2d: "
                f"missing keys={sorted(remaining_missing)}, "
                f"unexpected keys={sorted(load_result.unexpected_keys)}"
            )
        print(f"Weights loaded from {ckpt_path} (epoch {checkpoint_epoch})")
    else:
        raise FileNotFoundError(
            f"no {args.checkpoint} checkpoint at {ckpt_path}; evaluating random weights would "
            "produce a full set of results that traces to no trained model"
        )

    print(
        f"Evaluating {len(eval_set)} trajectories with the {args.checkpoint} checkpoint, "
        f"resolution {S_data[0]}x{S_data[1]}, {T_data} steps."
    )
    print(f"Using snapshot dt = {physics.dt:.6f}")

    # ── Base autoregressive rollout ─────────────────────────────────
    # Timed end to end, batching and host-to-device copies included: the cost of
    # the projector is the difference between a constrained and an unconstrained
    # rollout of the same length, and TTO is compared against it, so what matters
    # is that every variant is measured the same way.
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    rollout_start = time.perf_counter()
    initial_condition, pred_seq, truth_seq = autoregressive_predict(
        model,
        test_loader,
        device,
        grid,
        use_residual=use_residual,
        constrain_output=output_constraint_enabled,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    rollout_seconds = time.perf_counter() - rollout_start
    base_stats = compute_rollout_statistics(
        initial_condition,
        pred_seq,
        truth_seq,
        norm_mean=norm_mean,
        norm_std=norm_std,
        device=device,
        forcing=forcing,
        physics=physics,
        constraint_domain_lengths=(
            get_model_constraint_domain_lengths(model)
            if output_constraint_enabled
            else constraint_domain_lengths
        ),
        velocity_channels=tuple(getattr(model, "constraint_velocity_channels", (0, 1))),
        energy_balance_params=(
            {
                **energy_balance_params,
                "domain_lengths": get_model_constraint_domain_lengths(model),
            }
            if energy_balance_params is not None and output_constraint_enabled
            else energy_balance_params
        ),
        mode_cutoff=mode_cutoff,
    )
    print_rollout_statistics(base_stats)
    checkpoint_summary = {
        "split": args.split,
        "checkpoint": args.checkpoint,
        "checkpoint_path": ckpt_path,
        "checkpoint_epoch": checkpoint_epoch,
        "rollout_seconds": rollout_seconds,
        "rollout_seconds_per_trajectory": rollout_seconds / max(1, len(eval_set)),
    }
    print(
        f"Rollout wall-clock: {rollout_seconds:.2f} s for {len(eval_set)} trajectories "
        f"({checkpoint_summary['rollout_seconds_per_trajectory']:.4f} s each)"
    )
    save_rollout_artifacts(
        base_stats,
        save_dir,
        experiment_name,
        args.seed,
        file_suffix=variant_suffix,
        summary=checkpoint_summary,
    )

    if tto_cfg.get("enabled", False):
        save_suffix = tto_cfg.get("save_suffix", "_tto")
        if not save_suffix:
            raise ValueError("TTO save_suffix must be non-empty to avoid overwriting base outputs.")
        # The adapted rollout is a variant of this evaluation, not of the default
        # one, so it carries both names. With neither flag given the variant part
        # is empty and the suffix is the bare "_tto" it has always been.
        save_suffix = variant_suffix + save_suffix

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        tto_start = time.perf_counter()
        tto_initial_condition, tto_pred_seq, tto_truth_seq, tto_sample_reports, tto_free_trace = (
            run_test_time_optimization(
                model,
                reference_state,
                tto_loader,
                device,
                grid,
                forcing,
                norm_mean,
                norm_std,
                use_residual,
                tto_cfg,
                model_name,
                cont_weight=config["train"].get("f_loss", 1.0),
                momx_weight=config["train"].get("momx_loss", 1.0),
                momy_weight=config["train"].get("momy_loss", 1.0),
                physics=physics,
                scales=base_stats["residual_scales"],
                mode_cutoff=mode_cutoff,
            )
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        tto_seconds = time.perf_counter() - tto_start
        tto_summary = {
            **checkpoint_summary,
            "tto_seconds": tto_seconds,
            "tto_seconds_per_trajectory": tto_seconds / max(1, len(eval_set)),
        }
        print(
            f"TTO wall-clock: {tto_seconds:.2f} s for {len(eval_set)} trajectories "
            f"({tto_summary['tto_seconds_per_trajectory']:.4f} s each)"
        )
        tto_stats = compute_rollout_statistics(
            tto_initial_condition,
            tto_pred_seq,
            tto_truth_seq,
            norm_mean=norm_mean,
            norm_std=norm_std,
            device=device,
            forcing=forcing,
            physics=physics,
            constraint_domain_lengths=constraint_domain_lengths,
            velocity_channels=tuple(getattr(model, "constraint_velocity_channels", (0, 1))),
            energy_balance_params=energy_balance_params,
            mode_cutoff=mode_cutoff,
        )
        print_rollout_statistics(tto_stats, rollout_label="TTO-adapted")
        save_rollout_artifacts(
            tto_stats,
            save_dir,
            experiment_name,
            args.seed,
            file_suffix=save_suffix,
            summary=tto_summary,
        )
        save_tto_diagnostics(
            tto_sample_reports,
            save_dir,
            experiment_name,
            args.seed,
            file_suffix=save_suffix,
        )
        if tto_free_trace is not None:
            # The same layout rollout_stability.py writes, so the TTO arm's free-running
            # stability is read by whatever already reads the plain arms'.
            free_steps = tto_free_trace.shape[0]
            free_path = os.path.join(
                save_dir, append_file_suffix(f"free_rollout_{free_steps}", save_suffix) + ".npz"
            )
            free_blowup = blowup_steps(tto_free_trace[..., 0])
            np.savez(
                free_path,
                trace=tto_free_trace,
                blowup_steps=np.array(free_blowup),
                band_start=np.array(resolve_free_rollout_band(tto_cfg, mode_cutoff)),
            )
            never = sum(1 for step in free_blowup if step < 0)
            print(
                f"Free-running stability of the adapted weights: {never}/{len(free_blowup)} "
                f"trajectories survive {free_steps} steps; saved to {free_path}"
            )
        print_base_vs_tto_summary(base_stats, tto_stats)


if __name__ == "__main__":
    main()
