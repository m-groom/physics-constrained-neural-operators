"""Free-running rollout stability diagnostic: shell binning and blow-up detection."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# free_rollout imports the evaluator by its bare module name, as every script under
# experiments/ does when it is run from that directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from experiments.rollout_stability import blowup_steps, free_rollout, shell_energy

L = 2 * np.pi


def single_mode_field(side, kx, ky, amplitude=1.0):
    """A velocity field whose energy sits in exactly one shell."""
    x = torch.arange(side, dtype=torch.float64) * (L / side)
    field = torch.zeros(1, side, side, 2, dtype=torch.float32)
    field[0, :, :, 0] = (amplitude * torch.cos(kx * x[:, None] + ky * x[None, :])).float()
    return field


def test_shell_energy_puts_a_single_mode_in_its_own_shell():
    side = 32
    for kx, ky, shell in [(3, 0, 3), (0, 5, 5), (3, 4, 5)]:
        energy = shell_energy(single_mode_field(side, kx, ky), side)[0]
        assert energy.shape == (side // 2,)
        assert torch.argmax(energy).item() == shell
        others = torch.cat([energy[:shell], energy[shell + 1 :]])
        assert others.abs().max() < 1e-6 * energy[shell]


def test_shell_energy_matches_parseval():
    """The shell sum is the physical-space sum of |u|^2 up to the FFT's normalisation."""
    side = 16
    torch.manual_seed(0)
    field = torch.randn(3, side, side, 2)
    total = shell_energy(field, side).sum(1)
    expected = (field**2).sum((1, 2, 3)) * side * side
    assert torch.allclose(total, expected, rtol=1e-4)


def test_shell_energy_uses_one_channel_for_a_vorticity_state():
    side = 16
    field = torch.randn(2, side, side, 1)
    assert shell_energy(field, side).shape == (2, side // 2)


def test_blowup_steps_reports_the_first_crossing_one_based():
    ratio = np.ones((10, 3))
    ratio[4:, 0] = 20.0  # crosses at index 4 -> step 5
    ratio[7:, 1] = 11.0  # crosses at index 7 -> step 8
    assert blowup_steps(ratio, threshold=10.0) == [5, 8, -1]


def test_blowup_steps_counts_a_non_finite_entry():
    ratio = np.ones((5, 2))
    ratio[3, 0] = np.inf
    ratio[2, 1] = np.nan
    assert blowup_steps(ratio) == [4, 3]


def test_blowup_steps_respects_the_threshold():
    ratio = np.ones((6, 1))
    ratio[2:] = 5.0
    assert blowup_steps(ratio, threshold=10.0) == [-1]
    assert blowup_steps(ratio, threshold=2.0) == [3]


@pytest.mark.parametrize("side", [16, 32])
def test_shell_energy_is_translation_invariant(side):
    """Shifting a field in space moves phase, not shell energy."""
    torch.manual_seed(1)
    field = torch.randn(1, side, side, 2)
    shifted = torch.roll(field, shifts=(3, 5), dims=(1, 2))
    assert torch.allclose(shell_energy(field, side), shell_energy(shifted, side), rtol=1e-4)


class BlowsUpOneTrajectory:
    """Identity model whose trajectory `index` goes non-finite at call `at_step`."""

    def __init__(self, channels, index, at_step):
        self.channels = channels
        self.index = index
        self.at_step = at_step
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        out = x[..., : self.channels].clone()
        if self.calls >= self.at_step:
            out[self.index] = float("nan")
        return out


@pytest.mark.parametrize("channels", [1, 2, 3])
def test_free_rollout_keeps_rolling_when_one_trajectory_goes_non_finite(channels):
    """One blow-up must not truncate the batch: over 2,000 steps almost every arm has one.

    Every state the campaigns carry is covered: one channel for the vorticity formulation,
    three for the velocity one, and two for the plain velocity pair the tests above use.
    """
    side, steps = 8, 6
    initial = torch.ones(2, side, side, channels)
    grid = torch.zeros(side, side, 2)
    model = BlowsUpOneTrajectory(channels=channels, index=1, at_step=3)
    trace = free_rollout(model, initial, grid, steps, side, band_start=side // 4)
    assert trace.shape == (steps, 2, 2)
    assert model.calls == steps
    # The surviving trajectory is untouched by its neighbour's blow-up.
    assert np.isfinite(trace[:, 0, :]).all()
    assert np.allclose(trace[:, 0, 0], 1.0)
    assert blowup_steps(trace[..., 0]) == [-1, 3]


def test_free_rollout_leaves_a_clean_rollout_alone():
    """Nothing goes non-finite, so the trace is what it always was: full length and finite."""
    side, steps = 8, 5
    initial = torch.ones(3, side, side, 2)
    grid = torch.zeros(side, side, 2)
    # at_step past the horizon: the model is the identity throughout.
    trace = free_rollout(
        BlowsUpOneTrajectory(channels=2, index=0, at_step=steps + 1),
        initial,
        grid,
        steps,
        side,
        band_start=side // 4,
    )
    assert trace.shape == (steps, 3, 2)
    assert np.isfinite(trace).all()
    assert np.allclose(trace[..., 0], 1.0)
