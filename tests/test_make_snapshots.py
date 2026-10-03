"""Vorticity snapshot figure: archive reading, the shared colour scale, the panel grid."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "figures"))

import make_snapshots


def write_archive(path, steps):
    """An archive shaped like the one test_operator_AR_2d.py writes, level t holding t."""
    field = np.arange(steps + 1, dtype=np.float32) * np.ones((4, 4, steps + 1), dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, vorticity_pred=field, vorticity_truth=-field, step_indices=np.arange(steps + 1))


def write_run(runs_dir, run, stem, steps=8):
    """One archive under one run directory, named by its own stem."""
    write_archive(Path(runs_dir) / run / "saved_plots" / f"{stem}.npz", steps)


def test_archive_path_matches_the_evaluation_naming(tmp_path):
    write_run(tmp_path, "PINO_cont_seed1", "PINO_cont_seed1_vorticity_sample0_k16_T448")
    path = make_snapshots.archive_path(tmp_path, "PINO_cont", 1, "_k16_T448")
    assert path.name == "PINO_cont_seed1_vorticity_sample0_k16_T448.npz"
    assert path.parent.parent.name == "PINO_cont_seed1"


def test_archive_path_finds_an_adapted_run_under_the_checkpoint_it_adapted(tmp_path):
    # The evaluation writes PINO_TTO's adapted rollout into PINO's run directory,
    # and marks the file rather than the directory.
    write_run(tmp_path, "PINO_seed1", "PINO_TTO_seed1_vorticity_sample0_k16_T448_tto")
    path = make_snapshots.archive_path(tmp_path, "PINO_TTO_tto", 1, "_k16_T448")
    assert path.name == "PINO_TTO_seed1_vorticity_sample0_k16_T448_tto.npz"
    assert path.parent.parent.name == "PINO_seed1"


def test_archive_path_reports_an_arm_the_campaign_never_archived(tmp_path):
    write_run(tmp_path, "PINO_seed1", "PINO_seed1_vorticity_sample0")
    assert make_snapshots.archive_path(tmp_path, "PINO_TTO_tto", 1, "") is None


def test_available_arms_keeps_the_arms_archived_at_every_step(tmp_path):
    write_run(tmp_path, "FNO_seed1", "FNO_seed1_vorticity_sample0", steps=8)
    write_run(tmp_path, "FNO_seed1", "FNO_seed1_vorticity_sample0_long", steps=8)
    write_run(tmp_path, "PINO_seed1", "PINO_seed1_vorticity_sample0", steps=8)
    present, missing = make_snapshots.available_arms(
        tmp_path, ["FNO", "PINO"], 1, [(4, ""), (8, "_long")]
    )
    assert present == ["FNO"]
    assert missing == ["PINO"]


def test_build_rows_puts_the_truth_first_in_every_row(tmp_path):
    write_run(tmp_path, "FNO_seed1", "FNO_seed1_vorticity_sample0", steps=8)
    write_run(tmp_path, "PINO_seed1", "PINO_TTO_seed1_vorticity_sample0_tto", steps=8)
    rows = make_snapshots.build_rows(
        tmp_path, ["FNO", "PINO_TTO_tto"], {"PINO_TTO_tto": "PINO + TTO"}, 1, [(4, ""), (8, "")]
    )
    assert [label for label, _ in rows] == ["step 4", "step 8"]
    for _, panels in rows:
        assert [label for label, _ in panels] == ["truth", "FNO", "PINO + TTO"]
    assert np.allclose(rows[0][1][0][1], -4.0)
    assert np.allclose(rows[1][1][2][1], 8.0)


def test_read_snapshot_takes_the_named_step(tmp_path):
    path = tmp_path / "v.npz"
    write_archive(path, steps=8)
    assert np.allclose(make_snapshots.read_snapshot(path, 5, "vorticity_pred"), 5.0)
    assert np.allclose(make_snapshots.read_snapshot(path, 8, "vorticity_truth"), -8.0)


def test_read_snapshot_rejects_a_step_the_rollout_never_reached(tmp_path):
    path = tmp_path / "v.npz"
    write_archive(path, steps=8)
    with pytest.raises(ValueError, match="0-8"):
        make_snapshots.read_snapshot(path, 9, "vorticity_pred")


def test_symmetric_limit_is_the_largest_finite_magnitude():
    fields = [np.array([[-3.0, 1.0], [0.5, 2.0]]), np.array([[np.nan, np.inf], [0.0, -1.0]])]
    assert make_snapshots.symmetric_limit(fields) == pytest.approx(3.0)


def test_symmetric_limit_clips_the_tail_at_a_quantile():
    field = np.concatenate([np.zeros(99), [100.0]]).reshape(10, 10)
    assert make_snapshots.symmetric_limit([field]) == pytest.approx(100.0)
    assert make_snapshots.symmetric_limit([field], quantile=0.98) == pytest.approx(0.0)


def test_symmetric_limit_needs_a_finite_value():
    with pytest.raises(ValueError):
        make_snapshots.symmetric_limit([np.array([[np.nan, np.inf]])])


def test_wrap_label_gives_every_part_of_an_arm_name_its_own_line():
    # Nine panels share the text block, so the longest label -- family, qualifier and
    # the adapted marker -- has to break at both joins or it reaches into its
    # neighbour's panel.
    assert make_snapshots.wrap_label("PINO (cont only) + TTO") == "PINO\n(cont only)\n+ TTO"
    assert make_snapshots.wrap_label("PINO + TTO") == "PINO\n+ TTO"
    assert make_snapshots.wrap_label("FNO+proj (cont)") == "FNO+proj\n(cont)"
    assert make_snapshots.wrap_label("FNO") == "FNO"


def test_draw_snapshots_draws_one_panel_per_field_and_one_colourbar(tmp_path):
    rows = [
        ("step 64", [("truth", np.zeros((8, 8))), ("FNO", np.ones((8, 8)))]),
        ("step 448", [("truth", np.zeros((8, 8))), ("FNO", np.full((8, 8), np.nan))]),
    ]
    path = tmp_path / "snapshots.pdf"
    figure = make_snapshots.draw_snapshots(rows, path, width_in=5.5, limit=1.0)
    assert path.exists()
    assert len(figure.axes) == 5


def test_draw_snapshots_leaves_every_extreme_to_the_caption(tmp_path):
    # A lost rollout and an arm damped towards rest are both far off the shared scale,
    # and neither writes its extreme over its panel: the numbers are in the caption.
    # The only words on the figure are the panel titles, the row label and the colour
    # bar's label.
    rows = [
        (
            "step 448",
            [
                ("truth", np.zeros((4, 4))),
                ("FNO", np.full((4, 4), np.inf)),
                ("PINO + TTO", np.full((4, 4), 1e-3)),
            ],
        )
    ]
    figure = make_snapshots.draw_snapshots(rows, tmp_path / "s.pdf", width_in=5.5, limit=1.0)
    assert [text.get_text() for axis in figure.axes for text in axis.texts] == []
    assert [axis.get_title() for axis in figure.axes[:3]] == ["truth", "FNO", "PINO\n+ TTO"]
    assert figure.axes[0].get_ylabel() == "step 448"
    assert figure.axes[-1].get_xlabel() == "vorticity $\\omega$"
