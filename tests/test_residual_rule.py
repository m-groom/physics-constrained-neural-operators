"""Truth residual under the two time rules (issue #52, C3), on a tiny synthetic field."""

import math

import pytest
import torch

from experiments.residual_rule import velocity_residuals, vorticity_residual, windows
from utils.criterion import (
    PINO_loss3d,
    PINO_loss3d_vel,
    build_forcing_for_data,
    physics_from_config,
    residual_scales_from_truth,
)

S = 8  # tiny grid: only the wrapper's plumbing is under test, not the physics itself


def _physics(formulation):
    """A small, self-consistent Physics for a tiny synthetic field."""
    return physics_from_config(
        {
            "nu": 1.0 / 500,
            "alpha": 1e-8,
            "dt": 1.0 / 64,
            "domain_length": 2 * math.pi,
            "nx": S,
            "ny": S,
            "formulation": formulation,
            "forcing": {"type": "pino_cos4y", "amplitude": 1.0, "wavenumber": 4.0},
            "residual_upsample": 1,  # keep the residual grid the same tiny size in this test
        }
    )


def test_windows_extracts_every_consecutive_slice():
    trajectory = torch.arange(2 * 1 * 2 * 2 * 4, dtype=torch.float32).reshape(2, 1, 2, 2, 4)

    batch = windows(trajectory, width=3)

    # 2 trajectories x (4 - 3 + 1) windows each.
    assert batch.shape == (4, 1, 2, 2, 3)
    assert torch.equal(batch[0], trajectory[0, :, :, :, 0:3])
    assert torch.equal(batch[1], trajectory[0, :, :, :, 1:4])
    assert torch.equal(batch[2], trajectory[1, :, :, :, 0:3])
    assert torch.equal(batch[3], trajectory[1, :, :, :, 1:4])


def test_vorticity_residual_matches_a_direct_call_on_the_same_windows():
    """The chunked loop must reassemble the same relative L2 a direct call would give.

    ``vorticity_residual``'s aggregation is a plain mean of a mean, so it is exact for
    any chunk size; this checks the wrapper's windowing and chunking against the
    underlying ``PINO_loss3d`` call it is built on.
    """
    torch.manual_seed(0)
    physics = _physics("vorticity")
    forcing = build_forcing_for_data(physics, (S, S))
    truth = torch.randn(3, 1, S, S, 4, dtype=torch.float64)  # 3 trajectories, 4 time levels

    chunked = vorticity_residual(
        truth, physics, forcing, dt=1.0 / 64, time_method="integral", chunk=1
    )

    batch = windows(truth, width=2)
    w = batch[:, 0]
    _, expected = PINO_loss3d(
        w,
        w[..., 0],
        forcing,
        nu=physics.nu,
        alpha=physics.alpha,
        t_interval=1.0 / 64,
        domain_length=physics.domain_length,
        time_method="integral",
        upsample=physics.residual_upsample,
    )
    assert chunked["residual_rel"] == pytest.approx(float(expected))


def test_velocity_residuals_matches_a_direct_call_when_nothing_is_chunked():
    """With ``chunk >= n_trajectories`` the loop runs once, so this checks the wrapper's
    windowing, scale computation and index lookup ([4], [5], [6]) against a direct call.
    """
    torch.manual_seed(1)
    physics = _physics("velocity")
    forcing = build_forcing_for_data(physics, (S, S))
    truth = torch.randn(2, 3, S, S, 3, dtype=torch.float64)  # 2 trajectories, 3 time levels

    result = velocity_residuals(truth, physics, forcing, dt=1.0 / 64, time_method="fdm", chunk=8)

    scales = residual_scales_from_truth(truth, physics, (1.0 / 64) * 2, time_method="fdm")
    batch = windows(truth, width=3)
    out = PINO_loss3d_vel(
        batch,
        batch[..., 0],
        forcing,
        nu=physics.nu,
        alpha=physics.alpha,
        t_interval=(1.0 / 64) * 2,
        domain_length=physics.domain_length,
        scales=scales,
        time_method="fdm",
    )
    assert result["cont_rel"] == pytest.approx(float(out[4]))
    assert result["momx_rel"] == pytest.approx(float(out[5]))
    assert result["momy_rel"] == pytest.approx(float(out[6]))
    assert result["scale_cont"] == pytest.approx(scales.cont)
    assert result["scale_momx"] == pytest.approx(scales.momx)
    assert result["scale_momy"] == pytest.approx(scales.momy)
