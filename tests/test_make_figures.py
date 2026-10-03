"""Figure and table machinery, on synthetic campaign files.

Covers the parts that turn files into numbers --- discovery, reshaping over
seeds, counting stability out of the free rollouts, and the LaTeX a summary row
becomes --- and not the drawing, which has no assertable output.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib.inset import InsetIndicator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "figures"))

import make_figures

STEPS = np.array([1, 2, 3, 4])

SUMMARY = pd.DataFrame(
    {
        "model": ["FNO", "PINO"],
        "l2_rollout_mean_mean": [0.300, 0.004],
        "l2_rollout_mean_std": [0.350, 0.001],
        "seconds_per_epoch_mean": [77.2, 84.8],
        "seconds_per_epoch_std": [0.4, 1.8],
    }
)

TABLE_SPEC = {
    "output": "t.tex",
    "experiments": ["E1", "E3"],
    "columns": ["l2_rollout_mean", "seconds_per_epoch"],
    "rows": ["FNO", "PINO"],
}
TABLE_LABELS = ({"E1": "E1 (resolved)", "E3": "E2"}, {"FNO": "FNO", "PINO": "PINO"})


def write_run(runs_dir, model, seed, l2, suffix=""):
    """One run variant's per-step metrics inside its own run directory."""
    directory = runs_dir / f"{model}_seed{seed}" / "evaluation_metrics"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{model}_seed{seed}_per_step_metrics{suffix}.csv"
    pd.DataFrame({"step": STEPS, "l2_mean": l2}).to_csv(path, index=False)
    return path


def write_spectra(runs_dir, model, seed, step, predicted, suffix=""):
    """One run variant's saved spectra at one rollout step, over two trajectories."""
    directory = runs_dir / f"{model}_seed{seed}" / "saved_plots"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{model}_seed{seed}_spectra_t{step}{suffix}.npz"
    np.savez(
        path,
        k_bins=np.array([1.0, 2.0, 3.0]),
        Ek_pred=np.stack([predicted, predicted]),
        Ek_true=np.ones((2, 3)),
    )
    return path


def spectra_over(wavenumbers, truth, **arms):
    """Spectra shaped as ``seed_spectra`` returns them: one seed, one truth, named arms."""
    k = np.asarray(wavenumbers, dtype=float)
    reference = np.asarray(truth, dtype=float)[None, :]
    return {
        name: (k, np.asarray(values, dtype=float)[None, :], reference)
        for name, values in arms.items()
    }


def write_free(path, blowup, steps=4):
    """One free-running rollout archive, shaped as ``rollout_stability.py`` writes it.

    The energy trace carries the blow-ups: a trajectory listed as blowing up at
    step ``s`` runs above the guard from that step on.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    trace = np.ones((steps, len(blowup), 2), dtype=np.float32)
    for trajectory, step in enumerate(blowup):
        if step > 0:
            trace[step - 1 :, trajectory, 0] = 100.0
    np.savez(path, trace=trace, blowup_steps=np.array(blowup))
    return path


def write_training_log(runs_dir, model, seed, train, evaluated):
    """One run's per-epoch training log, named for its architecture and not its arm."""
    directory = runs_dir / f"{model}_seed{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"FNO2d-mode16_seed{seed}_training_log.csv"
    pd.DataFrame(
        {"epoch": np.arange(1, len(train) + 1), "train_l2": train, "eval_l2": evaluated}
    ).to_csv(path, index=False)
    return path


def write_metrics(runs_dir, model, seed, epoch, suffix=""):
    """One evaluation's summary row, which records the checkpoint's epoch."""
    directory = runs_dir / f"{model}_seed{seed}" / "evaluation_metrics"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{model}_seed{seed}_metrics{suffix}.csv"
    pd.DataFrame({"l2_step1": [0.1], "checkpoint_epoch": [epoch]}).to_csv(path, index=False)
    return path


def test_discover_reads_the_variant_from_the_file_not_the_directory(tmp_path):
    # The test-time-optimised rollout is written into the run directory of the
    # checkpoint it adapts, so directory and variant genuinely disagree.
    write_run(tmp_path, "PINO", 1, np.ones(4))
    write_run(tmp_path, "PINO_TTO", 1, np.ones(4))
    write_run(tmp_path, "PINO_TTO", 1, np.ones(4), suffix="_tto")

    found = make_figures.discover(tmp_path, make_figures.PER_STEP_NAME, "evaluation_metrics")

    assert set(found) == {"PINO", "PINO_TTO", "PINO_TTO_tto"}
    assert set(found["PINO_TTO_tto"]) == {1}


def test_discover_refuses_two_files_for_one_variant_and_seed(tmp_path):
    write_run(tmp_path, "FNO", 1, np.ones(4))
    duplicate = tmp_path / "FNO_rerun_seed1" / "evaluation_metrics"
    duplicate.mkdir(parents=True)
    pd.DataFrame({"step": STEPS, "l2_mean": np.ones(4)}).to_csv(
        duplicate / "FNO_seed1_per_step_metrics.csv", index=False
    )

    with pytest.raises(ValueError, match="FNO seed 1"):
        make_figures.discover(tmp_path, make_figures.PER_STEP_NAME, "evaluation_metrics")


def test_seed_curves_refuses_seeds_that_rolled_out_over_different_horizons(tmp_path):
    write_run(tmp_path, "FNO", 1, np.ones(4))
    short = tmp_path / "FNO_seed2" / "evaluation_metrics"
    short.mkdir(parents=True)
    pd.DataFrame({"step": STEPS[:3], "l2_mean": np.ones(3)}).to_csv(
        short / "FNO_seed2_per_step_metrics.csv", index=False
    )

    frames = make_figures.read_variants(tmp_path, ["FNO"])

    with pytest.raises(ValueError, match="covers step"):
        make_figures.seed_curves(frames["FNO"], "l2_mean")


def test_seed_curves_stacks_seeds_in_ascending_order(tmp_path):
    write_run(tmp_path, "FNO", 2, np.full(4, 2.0))
    write_run(tmp_path, "FNO", 1, np.full(4, 1.0))

    frames = make_figures.read_variants(tmp_path, ["FNO"])
    steps, values = make_figures.seed_curves(frames["FNO"], "l2_mean")

    assert np.array_equal(steps, STEPS)
    assert values.shape == (2, 4)
    assert np.array_equal(values[:, 0], [1.0, 2.0])


def test_seed_curves_reads_a_training_log_against_its_epoch():
    # A training log is measured against the epoch, not the rollout step, and the
    # same stacking has to serve both.
    frames = {
        1: pd.DataFrame({"epoch": [1, 2, 3], "eval_l2": [0.3, 0.2, 0.25]}),
        2: pd.DataFrame({"epoch": [1, 2, 3], "eval_l2": [0.5, 0.4, 0.45]}),
    }

    epochs, values = make_figures.seed_curves(frames, "eval_l2", index="epoch")

    assert np.array_equal(epochs, [1, 2, 3])
    assert np.array_equal(values[:, 1], [0.2, 0.4])


def test_read_variants_leaves_out_a_variant_the_campaign_has_not_run(tmp_path):
    write_run(tmp_path, "FNO", 1, np.ones(4))

    frames = make_figures.read_variants(tmp_path, ["FNO", "PINO"])

    assert set(frames) == {"FNO"}


def test_seed_spectra_take_the_last_step_and_average_the_trajectories(tmp_path):
    write_spectra(tmp_path, "FNO", 1, 1, np.array([9.0, 9.0, 9.0]))
    write_spectra(tmp_path, "FNO", 1, 64, np.array([1.0, 2.0, 3.0]))
    write_spectra(tmp_path, "FNO", 2, 1, np.array([9.0, 9.0, 9.0]))
    write_spectra(tmp_path, "FNO", 2, 64, np.array([3.0, 4.0, 5.0]))

    spectra = make_figures.seed_spectra(tmp_path, ["FNO"])
    wavenumbers, predicted, truth = spectra["FNO"]

    assert np.array_equal(wavenumbers, [1.0, 2.0, 3.0])
    assert predicted.shape == (2, 3)
    assert np.array_equal(predicted[0], [1.0, 2.0, 3.0])
    assert np.array_equal(truth, np.ones((2, 3)))


def test_seed_spectra_are_empty_when_a_campaign_saved_none(tmp_path):
    write_run(tmp_path, "FNO", 1, np.ones(4))

    assert make_figures.seed_spectra(tmp_path, ["FNO"]) == {}


def test_inset_limits_are_set_by_the_curves_that_still_carry_the_band():
    # Adapting at test time damps the field, so an adapted arm's spectrum falls
    # orders of magnitude below the truth at high wavenumber, which is what makes
    # the panel unreadable at the un-adapted arms' own scale.
    spectra = spectra_over(
        [1.0, 2.0, 3.0, 4.0],
        [8.0, 8.0, 1.0, 2.0],
        FNO=[9.0, 9.0, 3.0, 4.0],
        PINO_TTO_tto=[9.0, 9.0, 1e-6, 1e-6],
    )

    lower, upper = make_figures.inset_limits(spectra, (3, 4))

    assert lower == pytest.approx(1.0 / make_figures.INSET_PAD)
    assert upper == pytest.approx(4.0 * make_figures.INSET_PAD)


def test_inset_limits_read_an_arm_only_where_it_still_carries_the_band():
    # An arm that holds the truth's spectrum at the foot of the band and sheds
    # it inside would otherwise stretch the axis over the orders of magnitude
    # the inset exists to close up.
    spectra = spectra_over(
        [1.0, 2.0, 3.0, 4.0],
        [8.0, 8.0, 2.0, 2.0],
        FNO=[9.0, 9.0, 3.0, 4.0],
        PINO_TTO_tto=[9.0, 9.0, 2.0, 1e-9],
    )

    lower, upper = make_figures.inset_limits(spectra, (3, 4))

    assert lower == pytest.approx(2.0 / make_figures.INSET_PAD)
    assert upper == pytest.approx(4.0 * make_figures.INSET_PAD)


def test_inset_limits_are_unavailable_where_no_wavenumber_falls_in_the_band():
    spectra = spectra_over([1.0, 2.0], [1.0, 1.0], FNO=[1.0, 1.0])

    assert make_figures.inset_limits(spectra, (3, 4)) is None
    assert make_figures.inset_limits({}, (3, 4)) is None


def test_the_spectrum_panel_repeats_every_curve_inside_its_inset():
    spectra = spectra_over(
        [1.0, 2.0, 3.0, 4.0],
        [8.0, 8.0, 1.0, 2.0],
        FNO=[9.0, 9.0, 3.0, 4.0],
        PINO_TTO_tto=[9.0, 9.0, 1e-6, 1e-6],
    )
    styles = {
        "FNO": {"colour": "#2a78d6", "marker": "o", "label": "FNO"},
        "PINO_TTO_tto": {"colour": "#eb6834", "marker": "P", "dash": (0, (4.2, 1.5)), "label": "T"},
    }
    figure, ax = make_figures.plt.subplots()

    make_figures.panel_spectrum(ax, spectra, styles, (3, 4))

    (inner,) = ax.child_axes
    assert inner.get_xlim() == (3, 4)
    assert inner.get_ylim() == pytest.approx(make_figures.inset_limits(spectra, (3, 4)))
    assert inner.get_xscale() == "log"
    assert inner.get_yscale() == "log"
    # Every arm and the truth, in the hues, markers and strokes the panel draws
    # them in, and no legend: the panel names the arms once.
    assert len(inner.lines) == len(ax.lines) == len(spectra) + 1
    for drawn, panelled in zip(inner.lines, ax.lines, strict=True):
        assert drawn.get_color() == panelled.get_color()
        assert drawn.get_marker() == panelled.get_marker()
        assert drawn.get_linestyle() == panelled.get_linestyle()
    assert inner.get_legend() is None
    # The rectangle on the panel spans exactly what the inset shows.
    (indicator,) = ax.artists
    assert isinstance(indicator, InsetIndicator)
    x, y, width, height = indicator.rectangle.get_bbox().bounds
    lower, upper = inner.get_ylim()
    assert (x, x + width) == (3, 4)
    assert (y, y + height) == pytest.approx((lower, upper))
    make_figures.plt.close(figure)


def test_the_spectrum_panel_carries_no_inset_where_the_band_is_off_its_axis():
    spectra = spectra_over([1.0, 2.0], [1.0, 1.0], FNO=[1.0, 1.0])
    styles = {"FNO": {"colour": "#2a78d6", "marker": "o", "label": "FNO"}}
    figure, ax = make_figures.plt.subplots()

    make_figures.panel_spectrum(ax, spectra, styles, (3, 4))

    assert ax.child_axes == []
    make_figures.plt.close(figure)


def test_closed_panel_spectra_file_the_suffixed_archives_under_the_plain_name(tmp_path):
    # The closed campaign evaluated at a pinned band cutoff, so its spectra carry
    # the cutoff's suffix; the figure's styles and step-448 archive use the plain
    # arm name, and the campaign arms come through under their own names.
    write_spectra(
        tmp_path, "X_closed", 1, 64, np.array([1.0, 2.0, 3.0]), suffix=make_figures.CLOSED_SUFFIX
    )
    experiment = {"spectra": spectra_over([1.0, 2.0, 3.0], [1.0, 1.0, 1.0], X=[2.0, 2.0, 2.0])}

    spectra = make_figures.closed_panel_spectra(experiment, tmp_path, ["X", "X_closed"], 64)

    assert list(spectra) == ["X", "X_closed"]
    assert np.array_equal(spectra["X_closed"][1], [[1.0, 2.0, 3.0]])


def test_archived_median_reads_the_finite_rollouts_alone():
    # A NaN row marks a rollout whose step-448 state is non-finite; it must not
    # drag the median while the finite rollouts still define one.
    stacked = np.array([[[1.0, 1.0], [3.0, 3.0]], [[np.nan, np.nan], [5.0, 7.0]]])

    curve = make_figures.archived_median({"pred_X": stacked}, "X")

    assert curve == pytest.approx([3.0, 3.0])
    assert make_figures.archived_median({"pred_X": stacked}, "Y") is None


def test_free_traces_name_the_arm_from_the_archive_or_from_its_directory(tmp_path):
    # E3 writes one archive per arm into a directory of its own; E1 writes one
    # into each run directory, under a name that says only the horizon.
    write_free(tmp_path / "free" / "FNO_seed1.npz", [-1, 3])
    write_free(tmp_path / "runs" / "PINO_seed2" / "free_rollout_400.npz", [2, -1])

    named = make_figures.free_traces([str(tmp_path / "free" / "*.npz")])
    from_directory = make_figures.free_traces(
        [str(tmp_path / "runs" / "*" / "free_rollout_400.npz")]
    )

    assert set(named) == {"FNO"}
    assert set(from_directory) == {"PINO"}
    assert np.array_equal(from_directory["PINO"][2][1], [2, -1])


def test_free_traces_take_the_first_pattern_that_matches_anything(tmp_path):
    # The longer rollout supersedes the shorter one by being named ahead of it, so
    # a re-run at 2,000 steps needs no other change to reach the figures.
    write_free(tmp_path / "free" / "FNO_seed1.npz", [-1], steps=4)
    write_free(tmp_path / "free_2000" / "FNO_seed1.npz", [-1], steps=9)

    preferred = make_figures.free_traces(
        [str(tmp_path / "free_2000" / "*.npz"), str(tmp_path / "free" / "*.npz")]
    )
    fallback = make_figures.free_traces(
        [str(tmp_path / "never" / "*.npz"), str(tmp_path / "free" / "*.npz")]
    )

    assert preferred["FNO"][1][0].shape[0] == 9
    assert fallback["FNO"][1][0].shape[0] == 4


def test_free_traces_keep_the_finished_rollout_over_a_campaign_still_writing(tmp_path):
    # A campaign writes its archives one job at a time. Until it covers every arm
    # and seed the rollout it supersedes had, preferring it would drop most of the
    # seeds and count the survivors as bounded over steps never rolled for.
    for seed in (1, 2, 3):
        write_free(tmp_path / "free" / f"FNO_seed{seed}.npz", [-1], steps=4)
    write_free(tmp_path / "free_2000" / "FNO_seed1.npz", [-1], steps=9)

    patterns = [str(tmp_path / "free_2000" / "*.npz"), str(tmp_path / "free" / "*.npz")]
    partial = make_figures.free_traces(patterns)

    assert sorted(partial["FNO"]) == [1, 2, 3]
    assert partial["FNO"][1][0].shape[0] == 4

    for seed in (2, 3):
        write_free(tmp_path / "free_2000" / f"FNO_seed{seed}.npz", [-1], steps=9)
    complete = make_figures.free_traces(patterns)

    assert complete["FNO"][1][0].shape[0] == 9


def test_free_traces_count_coverage_and_not_files(tmp_path):
    # The same number of archives spread over more arms leaves some of them with
    # one seed where the settled rollout had five, which a count cannot see.
    for seed in (1, 2, 3):
        write_free(tmp_path / "free" / f"FNO_seed{seed}.npz", [-1], steps=4)
    write_free(tmp_path / "free_2000" / "FNO_seed1.npz", [-1], steps=9)
    write_free(tmp_path / "free_2000" / "PINO_seed1.npz", [-1], steps=9)
    write_free(tmp_path / "free_2000" / "PINO_seed2.npz", [-1], steps=9)

    traces = make_figures.free_traces(
        [str(tmp_path / "free_2000" / "*.npz"), str(tmp_path / "free" / "*.npz")]
    )

    assert set(traces) == {"FNO"}
    assert traces["FNO"][1][0].shape[0] == 4


def test_free_traces_file_an_adapted_rollout_under_the_adapted_variant(tmp_path):
    # A TTO campaign names the archive for the run it adapted, but the trace is of
    # the adapted weights, which the evaluation calls <run>_tto.
    write_free(tmp_path / "free" / "PINO_TTO_seed1.npz", [-1])
    write_free(tmp_path / "free" / "PINO_seed1.npz", [-1])

    traces = make_figures.free_traces([str(tmp_path / "free" / "*.npz")])

    assert set(traces) == {"PINO_TTO_tto", "PINO"}


def test_free_traces_are_empty_when_no_pattern_matches(tmp_path):
    assert make_figures.free_traces([str(tmp_path / "nothing" / "*.npz")]) == {}
    assert make_figures.free_traces([]) == {}


def test_exit_steps_count_a_drain_as_leaving_the_bounded_set():
    # Column 0 never leaves; column 1 passes the guard at step 3; column 2 drains
    # below a tenth at step 2, which the archive's own blow-up steps never count;
    # column 3 goes non-finite at step 4.
    ratio = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 2.0, 0.05, 1.0],
            [1.0, 20.0, 0.05, 1.0],
            [1.0, 20.0, 0.05, np.inf],
        ]
    )

    assert np.array_equal(make_figures.exit_steps(ratio, guard=10.0), [-1, 3, 2, 4])


def test_bounded_fraction_counts_a_trajectory_bounded_up_to_the_step_it_leaves():
    # -1 is a trajectory that never blew up; 3 is one whose energy first passed the
    # guard at step 3, so it is bounded at 2 and not at 3.
    fraction = make_figures.bounded_fraction([-1, 3], [1, 2, 3, 4])

    assert fraction == pytest.approx([1.0, 1.0, 0.5, 0.5])


def test_padded_ratio_holds_each_trajectory_at_its_last_measured_energy():
    # The recording stops for the whole batch when one trajectory fails, so the
    # failed one stays infinite and its healthy batch-mates keep their energy ---
    # which is what bounded_fraction says about the same steps.
    trace = np.array([[1.0, 1.0], [1.02, np.inf]])

    padded = make_figures.padded_ratio(trace, 4)

    assert padded.shape == (4, 2)
    assert padded[:, 0] == pytest.approx([1.0, 1.02, 1.02, 1.02])
    assert np.isinf(padded[1:, 1]).all()


def test_padded_ratio_trims_a_trace_longer_than_the_horizon():
    assert make_figures.padded_ratio(np.ones((6, 2)), 4).shape == (4, 2)


def test_free_horizon_is_the_longest_rollout_any_arm_recorded():
    traces = {
        "FNO": {1: (np.ones((4, 2)), np.array([-1, -1]))},
        "PINO": {1: (np.ones((7, 2)), np.array([-1, -1]))},
    }

    assert make_figures.free_horizon(traces) == 7
    assert make_figures.free_horizon({}) == 0


def test_training_logs_name_the_arm_from_the_run_directory(tmp_path):
    # The log's own file name carries the architecture, so the arm can only come
    # from the directory it sits in.
    write_training_log(tmp_path, "FNO_proj_cont", 1, [0.3, 0.2], [0.4, np.nan])

    logs = make_figures.training_logs(tmp_path)

    assert set(logs) == {"FNO_proj_cont"}
    assert list(logs["FNO_proj_cont"][1]["train_l2"]) == [0.3, 0.2]


def test_selected_epochs_read_the_checkpoint_the_evaluation_used(tmp_path):
    write_metrics(tmp_path, "FNO", 1, 271)
    write_metrics(tmp_path, "FNO", 2, 264)
    # A second evaluation of the same checkpoint under another cutoff must not
    # become an arm of its own.
    write_metrics(tmp_path, "FNO", 1, 271, suffix="_k16")

    epochs = make_figures.selected_epochs(tmp_path)

    assert epochs == {"FNO": {1: 271, 2: 264}}


def test_band_is_the_median_inside_the_interquartile_range():
    values = np.array([[1.0], [2.0], [3.0], [4.0], [100.0]])

    middle, lower, upper = make_figures.band(values)

    # The diverged fifth seed moves the upper quartile and leaves the median put,
    # which is the whole reason the figures use this spread and not mean +/- std.
    assert middle == pytest.approx([3.0])
    assert lower == pytest.approx([2.0])
    assert upper == pytest.approx([4.0])


def test_seed_range_keeps_the_seed_a_quartile_band_would_hide():
    values = np.array([[1.0], [2.0], [3.0], [4.0], [100.0]])

    middle, lower, upper = make_figures.seed_range(values)

    assert middle == pytest.approx([3.0])
    assert lower == pytest.approx([1.0])
    assert upper == pytest.approx([100.0])


def test_log_marks_reach_both_ends_without_repeating_an_index():
    marks = make_figures.log_marks(1000, marks=8)

    assert marks[0] == 0
    assert marks[-1] == 999
    assert marks == sorted(set(marks))


def test_ordered_walks_the_arms_in_the_configured_order_not_the_campaigns():
    available = {"PINO": 1, "FNO": 2, "unlisted": 3}

    assert list(make_figures.ordered(available, ["FNO", "PINO", "absent"])) == ["FNO", "PINO"]


def test_format_spread_stays_fixed_for_moderate_numbers():
    assert make_figures.format_spread(0.125, 0.004) == r"\shortstack{$0.125$\\$\pm0.004$}"


def test_format_spread_carries_the_mean_to_the_decade_of_its_spread():
    # Two significant figures alone print 1.109 and 1.066 both as 1.1, which is
    # the whole ordering of an accuracy column at the top of a decade.
    assert make_figures.format_spread(1.109, 0.013) == r"\shortstack{$1.11$\\$\pm0.01$}"
    assert make_figures.format_spread(1.066, 0.010) == r"\shortstack{$1.07$\\$\pm0.01$}"


def test_column_places_is_the_widest_any_entry_needs():
    # 0.0349 +/- 0.0004 needs four places and 0.0340 +/- 0.0007 needs three; the
    # column takes four so the two means sit digit above digit.
    assert make_figures.column_places([0.0349, 0.0340], [0.0004, 0.0007]) == 4


def test_column_places_ignores_the_entries_that_do_not_share_a_decimal_place():
    # An entry in exponent notation and a missing one carry no decimal place of
    # their own, so neither may widen or narrow the column.
    assert make_figures.column_places([2.6e-5, 0.50, None], [1.4e-7, 0.01, None]) == 2
    assert make_figures.column_places([2.6e-5, None], [1.4e-7, None]) is None


def test_a_column_prints_every_entry_at_the_place_the_column_settled_on():
    summary = pd.DataFrame(
        {
            "model": ["FNO", "PINO"],
            "l2_step32_mean": [0.0349, 0.0340],
            "l2_step32_std": [0.0004, 0.0007],
        }
    )

    cells = make_figures.table_cells(summary, ["FNO", "PINO"], ["l2_step32"])

    assert "0.0349" in cells["l2_step32"][0]
    assert "0.0340" in cells["l2_step32"][1]


def test_format_spread_never_slips_into_exponent_notation_inside_that_window():
    # 518.15 is inside the fixed window, where a "%g" format would have printed
    # 5.2e+02 and broken the column.
    assert make_figures.format_spread(518.15, 655.29) == r"\shortstack{$518$\\$\pm655$}"


def test_format_spread_shares_one_power_of_ten_outside_that_window():
    assert make_figures.format_spread(1.04e10, 2.32e10) == (
        r"\shortstack{$1.0{\pm}2.3$\\$\times 10^{10}$}"
    )


def test_format_spread_keeps_a_spread_that_would_round_away():
    # 1.4e-7 is 0.014 of 2.6e-5: reported at one decimal it would read as an
    # exact zero, which is a claim about constraint satisfaction, not a rounding.
    assert make_figures.format_spread(2.6e-5, 1.4e-7) == (
        r"\shortstack{$2.6{\pm}0.01$\\$\times 10^{-5}$}"
    )


MISSING_CELL = r"\shortstack{--\\\strut}"


def test_format_spread_reports_a_missing_mean_as_missing():
    assert make_figures.format_spread(None, None) == MISSING_CELL
    assert make_figures.format_spread(float("nan"), 1.0) == MISSING_CELL


def test_format_spread_prints_a_mean_alone_when_its_spread_is_unavailable():
    # Never "+/- 0.0", which reads as seeds that agreed exactly.
    assert make_figures.format_spread(1.5, float("nan")) == r"\shortstack{$1.5$\\\strut}"
    assert make_figures.format_spread(1.5, None) == r"\shortstack{$1.5$\\\strut}"


def test_format_spread_tells_an_overflow_apart_from_a_metric_never_recorded():
    assert make_figures.format_spread(float("inf"), 1.0) == r"\shortstack{$\infty$\\\strut}"
    assert make_figures.format_spread(float("nan"), 1.0) == MISSING_CELL


def test_format_spread_adds_places_rather_than_report_a_spread_as_zero():
    assert make_figures.format_spread(0.020, 9.4e-05) == r"\shortstack{$0.0200$\\$\pm0.0001$}"


def test_format_spread_emboldens_only_the_line_that_carries_the_digits():
    cell = make_figures.format_spread(1.04e10, 2.32e10, bold=True)

    assert cell == r"\shortstack{$\mathbf{1.0{\pm}2.3}$\\$\times 10^{10}$}"


def test_format_ratio_is_the_multiple_of_the_baseline():
    assert make_figures.format_ratio(84.8, 77.2) == r"\shortstack{$1.10\times$\\\strut}"
    assert make_figures.format_ratio(66.47, 0.0188) == r"\shortstack{$3536\times$\\\strut}"
    assert make_figures.format_ratio(1.0, None) == MISSING_CELL


def test_tto_inference_uses_the_optimisation_wall_clock():
    summary = pd.DataFrame(
        {
            "model": ["FNO", "PINO_TTO_tto"],
            "inference_s_per_trajectory_mean": [0.02, 0.09],
            "tto_s_per_trajectory_mean": [np.nan, 66.0],
        }
    )

    cells = make_figures.table_cells(
        summary, ["FNO", "PINO_TTO_tto"], ["inference_s_per_trajectory"]
    )

    assert cells["inference_s_per_trajectory"] == [
        r"\shortstack{$1.00\times$\\\strut}",
        r"\shortstack{$3300\times$\\\strut}",
    ]


def test_every_cell_is_two_lines_so_a_row_reads_on_one_baseline():
    # \shortstack sets a cell's baseline on its last line, so a one-line cell in a
    # row of stacks drops to the spread's line and the row stops reading straight.
    cells = make_figures.table_cells(SUMMARY, ["FNO", "PINO"], ["seconds_per_epoch"])

    assert all(cell.startswith(r"\shortstack") for cell in cells["seconds_per_epoch"])


def test_the_strict_rule_marks_the_leader_alone():
    # 0.300 +/- 0.350 against 0.004 +/- 0.001: the gap is inside the wider spread,
    # and the strict rule marks the smaller mean regardless.
    cells = make_figures.table_cells(SUMMARY, ["FNO", "PINO"], ["l2_rollout_mean"])

    assert r"\mathbf{" not in cells["l2_rollout_mean"][0]
    assert r"\mathbf{" in cells["l2_rollout_mean"][1]


def test_the_lenient_rule_marks_every_arm_level_with_the_leader():
    cells = make_figures.table_cells(SUMMARY, ["FNO", "PINO"], ["l2_rollout_mean"], strict=False)

    assert all(r"\mathbf{" in cell for cell in cells["l2_rollout_mean"])


def test_the_lenient_rule_marks_only_the_leader_when_the_spreads_separate_the_arms():
    separated = SUMMARY.assign(l2_rollout_mean_std=[0.001, 0.001])

    cells = make_figures.table_cells(separated, ["FNO", "PINO"], ["l2_rollout_mean"], strict=False)

    assert r"\mathbf{" not in cells["l2_rollout_mean"][0]
    assert r"\mathbf{" in cells["l2_rollout_mean"][1]


def test_arms_that_print_the_same_digits_are_marked_together():
    # 2.607e-5 against 2.608e-5: both print as 2.6, so bolding one of them would
    # read as a typesetting slip rather than as a result.
    summary = pd.DataFrame(
        {
            "model": ["FNO", "PINO"],
            "div_max_mean": [2.60694e-5, 2.60788e-5],
            "div_max_std": [1.0e-7, 2.0e-7],
        }
    )

    cells = make_figures.table_cells(summary, ["FNO", "PINO"], ["div_max"])

    assert all(r"\mathbf{" in cell for cell in cells["div_max"])


def test_printed_value_rounds_to_what_the_cell_shows():
    assert make_figures.printed_value(1.109, 0.013) == 1.11
    assert make_figures.printed_value(2.60694e-5, 1e-7) == pytest.approx(2.6e-5)
    assert make_figures.printed_value(None, None) is None


def test_the_stability_column_is_led_by_the_largest_entry():
    summary = pd.DataFrame(
        {
            "model": ["FNO", "PINO"],
            "never_blown_frac_mean": [0.18, 0.89],
            "never_blown_frac_std": [0.30, 0.16],
        }
    )

    cells = make_figures.table_cells(summary, ["FNO", "PINO"], ["never_blown_frac"])

    assert r"\mathbf{" not in cells["never_blown_frac"][0]
    assert r"\mathbf{" in cells["never_blown_frac"][1]


def test_an_excluded_arm_leads_no_accuracy_column_but_still_leads_the_others():
    summary = pd.DataFrame(
        {
            "model": ["FNO", "probe"],
            "l2_rollout_mean_mean": [0.30, 0.10],
            "l2_rollout_mean_std": [0.01, 0.01],
            "div_max_mean": [0.50, 1e-7],
            "div_max_std": [0.01, 1e-9],
        }
    )

    cells = make_figures.table_cells(
        summary, ["FNO", "probe"], ["l2_rollout_mean", "div_max"], excluded=("probe",)
    )

    assert r"\mathbf{" in cells["l2_rollout_mean"][0]  # the leader among the eligible
    assert r"\mathbf{" not in cells["l2_rollout_mean"][1]
    assert r"\mathbf{" in cells["div_max"][1]  # a constraint column it does compete in


def test_nothing_is_marked_in_a_column_no_arm_measured():
    cells = make_figures.table_cells(SUMMARY, ["FNO", "PINO"], ["div_max"])

    assert cells["div_max"] == [MISSING_CELL, MISSING_CELL]


def test_a_stability_header_names_the_horizon_it_was_counted_over():
    header = make_figures.column_header("never_blown_frac", 448)

    assert "448 steps" in header
    # The header names the criterion too: a campaign summary carries a
    # `blowup_never_frac` measured against the truth that means something else.
    assert "energy" in header
    assert make_figures.column_header("l2_step64", 448) == "step 64"


def test_stability_rows_average_the_never_blown_fraction_over_seeds():
    traces = {
        "FNO": {
            1: (np.ones((4, 2)), np.array([-1, -1])),
            2: (np.ones((4, 2)), np.array([2, -1])),
        }
    }

    counted = make_figures.stability_rows(traces, 4)

    assert counted.loc[0, "model"] == "FNO"
    assert counted.loc[0, "never_blown_frac_mean"] == pytest.approx(0.75)
    assert counted.loc[0, "never_blown_frac_std"] == pytest.approx(np.std([1.0, 0.5], ddof=1))


def test_stability_rows_report_no_spread_for_a_single_seed():
    # Never "+/- 0.00", which reads as seeds that agreed rather than as one seed.
    traces = {"FNO": {1: (np.ones((4, 2)), np.array([-1, 2]))}}

    counted = make_figures.stability_rows(traces, 4)

    assert counted.loc[0, "never_blown_frac_mean"] == pytest.approx(0.5)
    assert np.isnan(counted.loc[0, "never_blown_frac_std"])


def test_with_stability_leaves_an_arm_with_no_free_rollout_unmeasured():
    traces = {"FNO": {1: (np.ones((4, 2)), np.array([-1, -1]))}}

    merged = make_figures.with_stability(SUMMARY, traces, 4).set_index("model")

    assert merged.loc["FNO", "never_blown_frac_mean"] == pytest.approx(1.0)
    assert np.isnan(merged.loc["PINO", "never_blown_frac_mean"])


def test_with_stability_passes_a_campaign_with_no_free_rollouts_through():
    assert make_figures.with_stability(SUMMARY, {}, 0) is SUMMARY
    assert make_figures.with_stability(None, {}, 0) is None


def table_body(source):
    """The model rows of an emitted table, without its comments and rules."""
    return [line for line in source.splitlines() if r"\shortstack[l]{" in line]


def test_table_reports_an_experiment_that_has_not_run_as_missing():
    source = make_figures.latex_table(TABLE_SPEC, {"E1": SUMMARY, "E3": None}, *TABLE_LABELS)

    assert [row.count(make_figures.MISSING) for row in table_body(source)] == [2, 2]
    assert r"\multicolumn{2}{c}{E2}" in source


def test_table_quotes_cost_against_the_baseline_arm_of_the_same_experiment():
    source = make_figures.latex_table(TABLE_SPEC, {"E1": SUMMARY, "E3": None}, *TABLE_LABELS)

    assert r"{$1.00\times$" in source  # the baseline against itself, never emphasised
    assert r"1.10\times" in source


def test_an_overhead_column_marks_no_arm_at_all():
    # The baseline's own entry is one by construction, so a mark could only land
    # on a larger number beside a smaller unmarked one.
    cells = make_figures.table_cells(SUMMARY, ["FNO", "PINO"], ["seconds_per_epoch"])

    assert cells["seconds_per_epoch"] == [
        r"\shortstack{$1.00\times$\\\strut}",
        r"\shortstack{$1.10\times$\\\strut}",
    ]


def test_table_marks_an_excluded_arm_and_says_what_the_mark_means():
    spec = TABLE_SPEC | {"exclude_from_leading": ["PINO"]}

    source = make_figures.latex_table(spec, {"E1": SUMMARY, "E3": None}, *TABLE_LABELS)

    assert r"\shortstack[l]{PINO$^{\dagger}$" in source
    assert make_figures.EXCLUDED_NOTE in source


def test_table_sets_no_note_when_no_arm_is_excluded():
    source = make_figures.latex_table(TABLE_SPEC, {"E1": SUMMARY, "E3": None}, *TABLE_LABELS)

    assert r"\dagger" not in source


def test_table_column_specification_matches_the_number_of_columns():
    source = make_figures.latex_table(TABLE_SPEC, {"E1": SUMMARY, "E3": None}, *TABLE_LABELS)
    header = next(line for line in source.splitlines() if line.startswith(r"\begin{tabular}"))
    body = table_body(source)[0]

    assert header.count("c") == 4
    assert body.count("&") == 4
