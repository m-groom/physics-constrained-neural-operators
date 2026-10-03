"""Stability table (issue #52, C2): pooling blow-up steps and summarising them."""

import numpy as np

from experiments.stability_table import CHECKPOINTS, pooled_blowup_steps, summarise


def test_pooled_blowup_steps_reads_one_seed_and_maps_never_blown_to_infinity(tmp_path):
    # A 3-step trace: trajectory 0 blows up at step 2, trajectory 1 never blows up.
    np.savez(tmp_path / "ARM_seed1.npz", blowup_steps=np.array([2, -1]))

    steps, n_seeds = pooled_blowup_steps(tmp_path, "ARM", seeds=(1,))

    assert n_seeds == 1
    assert list(steps) == [2.0, np.inf]


def test_pooled_blowup_steps_skips_a_missing_seed_file(tmp_path):
    np.savez(tmp_path / "ARM_seed1.npz", blowup_steps=np.array([2, -1]))

    steps, n_seeds = pooled_blowup_steps(tmp_path, "ARM", seeds=(1, 2))

    assert n_seeds == 1
    assert list(steps) == [2.0, np.inf]


def test_summarise_reports_the_surviving_fraction_and_median():
    steps = np.array([2.0, np.inf])

    row = summarise(steps)

    assert row["n_trajectories"] == 2
    # Only the surviving trajectory is bounded past every checkpoint.
    for checkpoint in CHECKPOINTS:
        assert row[f"frac_bounded_{checkpoint}"] == 0.5
    # nearest-rank median over two entries is the lower of the two: the blow-up step.
    assert row["median_blowup_step"] == 2.0


def test_summarise_reports_never_when_the_median_trajectory_survives():
    steps = np.array([np.inf, np.inf, 2.0])

    row = summarise(steps)

    assert not np.isfinite(row["median_blowup_step"])
