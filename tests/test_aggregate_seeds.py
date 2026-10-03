"""Seed aggregation, on synthetic per-step metrics CSVs.

Covers both formulations: a velocity run carries divergence, domain-mean
velocity and kinetic energy, a vorticity run carries none of them.
"""

import json

import numpy as np
import pandas as pd
import pytest

from experiments import aggregate_seeds
from experiments.aggregate_seeds import main


def write_vorticity_run(directory, model, seed, l2_values, pred_residual, truth_residual):
    """One run's per-step metrics for a one-channel state."""
    frame = pd.DataFrame(
        {
            "step": np.arange(1, len(l2_values) + 1),
            "l2_mean": l2_values,
            "l2_std": np.zeros(len(l2_values)),
            "pred_cont_rel_mean": np.full(len(l2_values), pred_residual),
            "truth_cont_rel_mean": np.full(len(l2_values), truth_residual),
        }
    )
    path = directory / f"{model}_seed{seed}_per_step_metrics.csv"
    frame.to_csv(path, index=False)
    return path


def write_velocity_run(runs_dir, model, seed, l2, div_max, suffix="", seconds_per_epoch=None):
    """One run's evaluation artefacts: the per-step frame and the metrics row."""
    run_dir = runs_dir / f"{model}_seed{seed}"
    results = run_dir / "evaluation_metrics"
    results.mkdir(parents=True, exist_ok=True)
    if seconds_per_epoch is not None:
        (run_dir / "train_timing.json").write_text(
            json.dumps({"seconds_per_epoch": seconds_per_epoch})
        )

    pd.DataFrame(
        [
            {
                "model": f"{model}{suffix}",
                "seed": seed,
                "n_trajectories": 4,
                "non_finite_total": 0,
                "rollout_seconds_per_trajectory": 0.5,
                "tto_seconds_per_trajectory": 12.0 if suffix else float("nan"),
            }
        ]
    ).to_csv(results / f"{model}_seed{seed}_metrics{suffix}.csv", index=False)

    pd.DataFrame(
        {
            "step": [1, 2],
            "l2_mean": [l2, l2],
            "pred_div_max_mean": [div_max, div_max],
            "truth_div_max_mean": [1e-5, 1e-5],
            "pred_domain_mean_speed_mean": [1e-3, 1e-3],
            "truth_domain_mean_speed_mean": [1e-9, 1e-9],
            "pred_cont_rel_mean": [0.1, 0.1],
            "truth_cont_rel_mean": [0.01, 0.01],
            "pred_momx_rel_mean": [0.2, 0.2],
            "truth_momx_rel_mean": [0.02, 0.02],
            "pred_momy_rel_mean": [0.3, 0.3],
            "truth_momy_rel_mean": [0.03, 0.03],
            "pred_energy_mean": [1.0, 1.1],
            "truth_energy_mean": [1.0, 1.05],
        }
    ).to_csv(results / f"{model}_seed{seed}_per_step_metrics{suffix}.csv", index=False)
    # A per-sample artefact shares the directory and is not a run row.
    np.savez(results / f"{model}_seed{seed}_per_sample_metrics{suffix}.npz", l2=np.zeros(2))
    return run_dir


@pytest.fixture
def campaign(tmp_path):
    """Two vorticity models, three seeds each, over a four-step horizon."""
    paths = []
    for seed in (1, 2, 3):
        paths.append(
            write_vorticity_run(
                tmp_path, "FNO_vort", seed, [0.1, 0.2, 0.3, 0.4 + 0.1 * seed], 0.9, 0.05
            )
        )
        paths.append(
            write_vorticity_run(
                tmp_path, "PINO_vort", seed, [0.1, 0.2, 0.3, 0.2 + 0.1 * seed], 0.5, 0.05
            )
        )
    return tmp_path, [str(path) for path in paths]


@pytest.fixture
def velocity_campaign(tmp_path):
    """Two velocity models over two seeds, laid out as a campaign directory."""
    runs_dir = tmp_path / "runs"
    write_velocity_run(runs_dir, "FNO", 1, l2=0.10, div_max=1e-2, seconds_per_epoch=100.0)
    write_velocity_run(runs_dir, "FNO", 2, l2=0.20, div_max=1e-2, seconds_per_epoch=100.0)
    write_velocity_run(runs_dir, "FNO_proj", 1, l2=0.12, div_max=1e-14)
    write_velocity_run(runs_dir, "FNO_proj", 2, l2=0.18, div_max=1e-14)
    return runs_dir


def test_the_summary_averages_over_the_seeds(campaign):
    tmp_path, paths = campaign
    out = tmp_path / "out"
    main([*paths, "--output_dir", str(out), "--steps", "1", "4"])

    summary = pd.read_csv(out / "summary.csv").set_index("model")
    assert list(summary.index) == ["FNO_vort", "PINO_vort"]
    assert (summary["n_seeds"] == 3).all()

    # FNO rollout means are (0.5+0.6+0.7)/4 per seed: 0.275, 0.3, 0.325.
    assert summary.loc["FNO_vort", "l2_rollout_mean_mean"] == pytest.approx(0.3)
    assert summary.loc["FNO_vort", "l2_rollout_mean_std"] == pytest.approx(0.025)
    assert summary.loc["PINO_vort", "l2_step4_mean"] == pytest.approx(0.4)
    assert summary.loc["PINO_vort", "l2_step1_std"] == pytest.approx(0.0)
    assert summary.loc["PINO_vort", "residual_rel_pred_mean"] == pytest.approx(0.5)
    assert summary.loc["FNO_vort", "residual_rel_truth_mean"] == pytest.approx(0.05)


def test_a_vorticity_run_reports_the_velocity_metrics_as_missing(campaign):
    tmp_path, paths = campaign
    out = tmp_path / "out"
    main([*paths, "--output_dir", str(out), "--steps", "1", "4"])

    runs = pd.read_csv(out / "runs.csv")
    assert runs["div_max"].isna().all()
    assert runs["energy_error_final"].isna().all()
    # A missing metric renders as a dash, not as a number nobody measured.
    assert "| max|div u| (worst step) | -- | -- |" in (out / "summary.md").read_text()


def test_the_runs_table_keeps_one_row_per_run(campaign):
    tmp_path, paths = campaign
    out = tmp_path / "out"
    main([*paths, "--output_dir", str(out)])

    runs = pd.read_csv(out / "runs.csv")
    assert len(runs) == 6
    assert sorted(runs["seed"].unique()) == [1, 2, 3]
    assert (runs["n_steps"] == 4).all()
    # A step beyond the horizon is missing, not invented.
    assert runs["l2_step16"].isna().all()


def test_the_paired_comparison_is_per_seed(campaign):
    tmp_path, paths = campaign
    out = tmp_path / "out"
    main([*paths, "--output_dir", str(out), "--steps", "1", "4"])

    report = (out / "summary.md").read_text()
    assert "PINO_vort - FNO_vort" in report
    # PINO beats FNO by 0.05 in the rollout mean for every seed.
    per_seed = report.split("### Rollout relative L2 per seed")[1].split("## ")[0]
    assert per_seed.count("-0.05") == 3
    assert "0.3000 ± 0.0250" in report


def test_an_unrecognised_file_name_is_rejected(tmp_path):
    path = tmp_path / "FNO_vort_per_step_metrics.csv"
    path.write_text("step,l2_mean\n1,0.1\n")
    with pytest.raises(ValueError, match="seed"):
        main([str(path), "--output_dir", str(tmp_path / "out")])


def test_a_campaign_directory_is_walked(velocity_campaign):
    paths = aggregate_seeds.find_metric_files(str(velocity_campaign))
    assert len(paths) == 4
    assert all(path.endswith("_per_step_metrics.csv") for path in paths)


def test_the_velocity_metrics_and_the_training_cost_are_carried(velocity_campaign, tmp_path):
    out = tmp_path / "out"
    main(["--runs", str(velocity_campaign), "--out", str(out), "--steps", "1,2"])

    summary = pd.read_csv(out / "summary.csv").set_index("model")
    assert summary.loc["FNO_proj", "div_max_mean"] == pytest.approx(1e-14)
    assert summary.loc["FNO", "seconds_per_epoch_mean"] == pytest.approx(100.0)
    # FNO_proj wrote no timing file, so it reports no training cost.
    assert np.isnan(summary.loc["FNO_proj", "seconds_per_epoch_mean"])
    assert summary.loc["FNO", "inference_s_per_trajectory_mean"] == pytest.approx(0.5)


def test_the_energy_error_and_the_drift_answer_different_questions(velocity_campaign, tmp_path):
    out = tmp_path / "out"
    main(["--runs", str(velocity_campaign), "--out", str(out), "--steps", "1,2"])
    runs = pd.read_csv(out / "runs.csv")
    # Energy runs 1.0 -> 1.1 in the prediction and 1.0 -> 1.05 in the truth.
    np.testing.assert_allclose(runs["energy_error_final"], (1.1 - 1.05) / 1.05, rtol=1e-9)
    np.testing.assert_allclose(runs["energy_drift"], 0.1, rtol=1e-9)
    np.testing.assert_allclose(runs["truth_energy_drift"], 0.05, rtol=1e-9)


def test_the_difference_is_paired_seed_by_seed(velocity_campaign, tmp_path):
    out = tmp_path / "out"
    main(["--runs", str(velocity_campaign), "--out", str(out), "--baseline", "FNO", "--steps", "2"])
    summary = pd.read_csv(out / "summary.csv").set_index("model")
    # Differences are +0.02 and -0.02: a mean of zero the seeds do not share.
    assert summary.loc["FNO_proj", "l2_rollout_mean_paired_mean"] == pytest.approx(0.0)
    assert summary.loc["FNO_proj", "l2_rollout_mean_paired_std"] == pytest.approx(0.02 * 2**0.5)


def test_a_suffixed_rollout_is_its_own_variant(velocity_campaign, tmp_path):
    write_velocity_run(velocity_campaign, "FNO", 1, l2=0.08, div_max=1e-2, suffix="_tto")
    out = tmp_path / "out"
    main(["--runs", str(velocity_campaign), "--out", str(out), "--steps", "1,2"])

    summary = pd.read_csv(out / "summary.csv").set_index("model")
    assert "FNO_tto" in summary.index
    assert summary.loc["FNO_tto", "tto_s_per_trajectory_mean"] == pytest.approx(12.0)


def test_a_diverged_seed_is_counted_rather_than_dropped(velocity_campaign, tmp_path):
    write_velocity_run(velocity_campaign, "FNO", 3, l2=42.0, div_max=1e-2)
    out = tmp_path / "out"
    main(["--runs", str(velocity_campaign), "--out", str(out), "--steps", "1,2"])

    summary = pd.read_csv(out / "summary.csv").set_index("model")
    assert int(summary.loc["FNO", "n_bounded"]) == 2
    assert int(summary.loc["FNO", "n_seeds"]) == 3
    # The mean carries the diverged seed; the median does not.
    assert summary.loc["FNO", "l2_rollout_mean_mean"] == pytest.approx((0.10 + 0.20 + 42.0) / 3)
    assert summary.loc["FNO", "l2_rollout_mean_median"] == pytest.approx(0.20)
    report = (out / "summary.md").read_text()
    assert "seeds bounded" in report
    assert "Median over seeds" in report


def test_duplicate_model_seed_pairs_are_refused(tmp_path):
    """A stale smoke directory that repeats a (model, seed) pair must abort the aggregation."""
    import pytest

    from experiments.aggregate_seeds import main

    def per_step(runs_dir):
        write_velocity_run(runs_dir, "FNO", 1, l2=0.1, div_max=1e-5)
        return runs_dir / "FNO_seed1" / "evaluation_metrics" / "FNO_seed1_per_step_metrics.csv"

    fresh = per_step(tmp_path / "runs")
    stale = per_step(tmp_path / "runs_smoke")
    with pytest.raises(SystemExit, match="duplicate"):
        main([str(fresh), str(stale), "--out", str(tmp_path / "out")])
