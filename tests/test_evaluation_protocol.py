"""Evaluation protocol: baseline diagnostics, checkpoint choice, strict aggregation."""

import math
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from experiments import test_operator_AR_2d as evaluator
from utils.criterion import build_forcing_for_data, physics_from_config

S = 16
DATA_CONFIG = {
    "nu": 0.002,
    "alpha": 1.0e-8,
    "dt": 0.015625,
    "domain_length": 2 * math.pi,
    "nx": S,
    "ny": S,
    "formulation": "velocity",
    "forcing": {
        "type": "kolmogorov_diag",
        "amplitude": 2 * math.sqrt(2),
        "wavenumber": 4,
        "phase": 5 * math.pi / 4,
    },
}


def _rollout_statistics(pred_seq=None, energy_balance_params=None):
    """Rollout statistics for two tiny trajectories of two steps each."""
    torch.manual_seed(0)
    physics = physics_from_config(DATA_CONFIG)
    forcing = build_forcing_for_data(physics, (S, S), device=torch.device("cpu"))
    truth = torch.randn(2, S, S, 3, 3)
    if pred_seq is None:
        pred_seq = truth + 0.01 * torch.randn_like(truth)
    return evaluator.compute_rollout_statistics(
        truth[..., 0],
        pred_seq,
        truth[..., 1:],
        norm_mean=torch.zeros(3, 1, 1),
        norm_std=torch.ones(3, 1, 1),
        device=torch.device("cpu"),
        forcing=forcing,
        physics=physics,
        constraint_domain_lengths=None,
        energy_balance_params=energy_balance_params,
    )


class BaselineDiagnosticsTest(unittest.TestCase):
    def test_divergence_is_reported_without_a_constraint_layer(self):
        stats = _rollout_statistics()
        div_max = stats["pred_div_max"]
        self.assertIsNotNone(div_max)
        self.assertEqual(div_max.shape, (2, 2))
        self.assertTrue(np.isfinite(div_max).all())
        self.assertGreater(div_max.min(), 0.0)

    def test_domain_mean_velocity_is_reported_for_prediction_and_truth(self):
        stats = _rollout_statistics()
        self.assertEqual(stats["pred_domain_mean_velocity"]["mean_ux"].shape, (2, 2))
        self.assertEqual(stats["truth_domain_mean_velocity"]["mean_ux"].shape, (2, 2))

    def test_the_residual_scales_come_from_the_truth_trajectory(self):
        stats = _rollout_statistics()
        scales = stats["residual_scales"]
        self.assertGreater(scales.cont, 0.0)
        self.assertGreater(scales.momx, 0.0)


class StrictAggregationTest(unittest.TestCase):
    def test_a_non_finite_sample_is_counted_and_not_averaged_away(self):
        torch.manual_seed(0)
        truth = torch.randn(2, S, S, 3, 3)
        pred = truth + 0.01 * torch.randn_like(truth)
        pred[1, 0, 0, 0, 2] = float("nan")
        stats = _rollout_statistics(pred_seq=pred)

        self.assertGreater(stats["non_finite_counts"]["l2"], 0)
        self.assertTrue(np.isnan(stats["l2_mean"][-1]))
        self.assertEqual(stats["n_trajectories"], 2)

    def test_metric_mean_keeps_a_non_finite_entry(self):
        values = np.array([[1.0, np.nan], [1.0, 1.0]])
        self.assertTrue(np.isnan(evaluator.metric_mean(values, axis=0)[1]))
        self.assertEqual(evaluator.count_non_finite({"values": values}), {"values": 1})


class CheckpointSelectionTest(unittest.TestCase):
    def test_best_is_the_default_and_last_is_available(self):
        best = evaluator.resolve_checkpoint_path("/runs/FNO", "FNO2d.pt", seed=42, which="best")
        last = evaluator.resolve_checkpoint_path("/runs/FNO", "FNO2d.pt", seed=42, which="last")
        self.assertEqual(best, "/runs/FNO/FNO2d_seed42_best.pt")
        self.assertEqual(last, "/runs/FNO/FNO2d_seed42.pt")

    def test_an_unknown_choice_is_rejected(self):
        with self.assertRaises(ValueError):
            evaluator.resolve_checkpoint_path("/runs/FNO", "FNO2d.pt", seed=42, which="latest")


class EnergyDiagnosticTest(unittest.TestCase):
    """Kinetic energy is reported per predicted step, for every velocity model."""

    def test_the_per_step_energy_aligns_with_the_predicted_states(self):
        domain_lengths = (2 * math.pi, 2 * math.pi)
        stats = _rollout_statistics(energy_balance_params={"domain_lengths": domain_lengths})
        with tempfile.TemporaryDirectory() as save_dir:
            evaluator.save_rollout_artifacts(stats, save_dir, "FNO", seed=7)
            frame = pd.read_csv(
                Path(save_dir) / "evaluation_metrics" / "FNO_seed7_per_step_metrics.csv"
            )
        n_steps = stats["l2_per_step"].shape[1]
        self.assertEqual(len(frame), n_steps)

        # Energy of the predicted state at each step, computed independently.
        cell_area = domain_lengths[0] * domain_lengths[1] / (S * S)
        velocity = stats["pred_phys"][..., :2, 1:]
        expected = 0.5 * cell_area * velocity.square().sum(dim=(1, 2, 3)).mean(dim=0)
        np.testing.assert_allclose(
            frame["pred_energy_mean"].to_numpy(), expected.numpy(), rtol=1e-5
        )

    def test_an_unconstrained_model_gets_the_energy_columns(self):
        stats = _rollout_statistics(
            energy_balance_params={"domain_lengths": (2 * math.pi, 2 * math.pi)}
        )
        self.assertIsNotNone(stats["pred_energy_balance"])
        self.assertIsNotNone(stats["truth_energy_balance"])


VORTICITY_DATA_CONFIG = {
    **DATA_CONFIG,
    "formulation": "vorticity",
    "forcing": {"type": "pino_cos4y", "amplitude": 1.0, "wavenumber": 4},
    "residual_upsample": 2,
}


def _vorticity_rollout_statistics():
    """Rollout statistics for two tiny one-channel trajectories of two steps each."""
    torch.manual_seed(0)
    physics = physics_from_config(VORTICITY_DATA_CONFIG)
    forcing = build_forcing_for_data(physics, (S, S), device=torch.device("cpu"))
    truth = torch.randn(2, S, S, 1, 3)
    pred_seq = truth + 0.01 * torch.randn_like(truth)
    return evaluator.compute_rollout_statistics(
        truth[..., 0],
        pred_seq,
        truth[..., 1:],
        norm_mean=torch.zeros(1, 1, 1),
        norm_std=torch.ones(1, 1, 1),
        device=torch.device("cpu"),
        forcing=forcing,
        physics=physics,
        constraint_domain_lengths=None,
        velocity_channels=None,
    )


class VorticityFormulationTest(unittest.TestCase):
    """A one-channel state carries no velocity: those diagnostics stand down."""

    def test_the_rollout_metrics_are_finite(self):
        stats = _vorticity_rollout_statistics()
        self.assertEqual(stats["l2_per_step"].shape, (2, 2))
        self.assertTrue(np.isfinite(stats["l2_per_step"]).all())
        self.assertTrue(np.isfinite(stats["pde_pred"]["loss_cont_rel"]).all())
        self.assertTrue(np.isfinite(stats["pde_truth"]["loss_cont_rel"]).all())
        self.assertEqual(sum(stats["non_finite_counts"].values()), 0)

    def test_the_velocity_only_diagnostics_are_skipped(self):
        stats = _vorticity_rollout_statistics()
        self.assertIsNone(stats["pred_div_max"])
        self.assertIsNone(stats["truth_div_max"])
        self.assertIsNone(stats["pred_domain_mean_velocity"])
        self.assertIsNone(stats["residual_scales"])

    def test_the_artifacts_are_written_without_the_velocity_columns(self):
        stats = _vorticity_rollout_statistics()
        with tempfile.TemporaryDirectory() as save_dir:
            evaluator.print_rollout_statistics(stats)
            evaluator.save_rollout_artifacts(stats, save_dir, "PINO_vort", seed=1)

            metrics_dir = Path(save_dir) / "evaluation_metrics"
            per_step = pd.read_csv(metrics_dir / "PINO_vort_seed1_per_step_metrics.csv")
            self.assertIn("pred_cont_rel_mean", per_step.columns)
            self.assertNotIn("pred_div_max_mean", per_step.columns)
            self.assertEqual(len(per_step), 2)

            # The spectra come from the velocity the vorticity implies.
            spectra = np.load(Path(save_dir) / "saved_plots" / "PINO_vort_seed1_spectra_t1.npz")
            self.assertEqual(spectra["Ek_pred"].shape[0], 2)
            self.assertTrue(np.isfinite(spectra["Ek_pred"]).all())

            vorticity = np.load(
                Path(save_dir) / "saved_plots" / "PINO_vort_seed1_vorticity_sample0.npz"
            )
            self.assertTrue(
                np.allclose(
                    vorticity["vorticity_pred"], stats["pred_phys"][0, ..., 0, :].numpy(), atol=0
                )
            )


class RolloutShapeTest(unittest.TestCase):
    """A one-channel state survives the rollout: its channel axis is not squeezed."""

    class ZeroResidual(torch.nn.Module):
        def forward(self, x_in):
            return x_in[..., :1] * 0.0

    def test_a_one_channel_state_rolls_out_as_a_residual(self):
        initial_condition = torch.randn(2, S, S, 1)
        grid = torch.zeros(S, S, 4)

        rollout = evaluator.autoregressive_rollout(
            self.ZeroResidual(), initial_condition, grid, rollout_steps=3, use_residual=True
        )

        self.assertEqual(rollout.shape, (2, S, S, 1, 4))
        # A zero residual repeats the initial condition at every step.
        self.assertTrue(torch.allclose(rollout[..., -1], initial_condition))

    def test_a_trailing_singleton_time_axis_is_still_dropped(self):
        prev = torch.randn(2, S, S, 3)
        self.assertEqual(
            evaluator.align_prediction(torch.randn(2, S, S, 3, 1), prev).shape, (2, S, S, 3)
        )
        self.assertEqual(
            evaluator.align_prediction(torch.randn(2, S, S, 1, 3), prev).shape, (2, S, S, 3)
        )
        self.assertEqual(
            evaluator.align_prediction(torch.randn(2, S, S, 1), torch.randn(2, S, S, 1)).shape,
            (2, S, S, 1),
        )


class EvaluationVariantTest(unittest.TestCase):
    """A non-default split or horizon must name its own artefacts.

    ``save_rollout_artifacts`` names every file after the experiment and the seed
    alone, so a second evaluation of the same checkpoint overwrites the first
    unless it carries a suffix.
    """

    def test_the_default_evaluation_keeps_its_bare_name(self):
        self.assertEqual(evaluator.evaluation_variant_suffix("test", None), "")

    def test_another_split_names_itself(self):
        self.assertEqual(evaluator.evaluation_variant_suffix("test_time", None), "_test_time")

    def test_a_longer_horizon_names_itself(self):
        self.assertEqual(evaluator.evaluation_variant_suffix("test", 448), "_T448")

    def test_a_split_and_a_horizon_both_appear(self):
        self.assertEqual(evaluator.evaluation_variant_suffix("test_time", 128), "_test_time_T128")


class SplitAvailabilityTest(unittest.TestCase):
    """A mistyped split must raise, not silently evaluate another one.

    ``NSLoader2D`` falls back to the first ``X``/``y`` key in the file when
    ``X_<split>`` is missing, and in every dataset this repository builds that
    first key is ``X_train``.
    """

    def _dataset(self, directory):
        path = Path(directory) / "d.npz"
        np.savez(
            path,
            X_train=np.zeros((2, 3, 4, 4), dtype=np.float32),
            y_train=np.zeros((2, 3, 4, 4), dtype=np.float32),
            X_test=np.zeros((2, 3, 4, 4), dtype=np.float32),
            y_test=np.zeros((2, 3, 4, 4), dtype=np.float32),
        )
        return path

    def test_a_split_the_file_carries_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            self._dataset(directory)
            evaluator.check_split_available(directory, "d.npz", "test")

    def test_a_split_the_file_lacks_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            self._dataset(directory)
            with self.assertRaises(KeyError) as raised:
                evaluator.check_split_available(directory, "d.npz", "test_tme")
            self.assertIn("test_tme", str(raised.exception))


class RolloutHorizonTest(unittest.TestCase):
    """A horizon must divide every contiguous run, not merely the frame count.

    ``transform_rollout`` reshapes the flat frame list into ``(n, steps)`` and
    checks divisibility alone, so a horizon that divides the total but not the
    length of each run stitches independent stretches of the flow into one
    rollout and reports it as though it were continuous.
    """

    def _dataset(self, directory, runs, dt=0.4):
        """An npz whose ``times_test`` describes contiguous runs of given lengths."""
        rows = []
        start = 0.0
        for length in runs:
            for _ in range(length):
                rows.append((start, start + dt))
                start += dt
            start += 100.0  # a gap: the next run does not continue this one
        times = np.array(rows, dtype=np.float64)
        path = Path(directory) / "d.npz"
        np.savez(path, X_test=np.zeros((len(rows), 3, 4, 4), dtype=np.float32), times_test=times)
        return path

    def test_a_horizon_dividing_every_run_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            self._dataset(directory, runs=[2688])
            evaluator.check_rollout_horizon(directory, "d.npz", "test", 448)

    def test_a_horizon_that_would_stitch_two_runs_is_refused(self):
        # 200 runs of 64: 128 divides the 12800 total but not the run length.
        with tempfile.TemporaryDirectory() as directory:
            self._dataset(directory, runs=[64] * 200)
            evaluator.check_rollout_horizon(directory, "d.npz", "test", 64)
            with self.assertRaises(ValueError) as raised:
                evaluator.check_rollout_horizon(directory, "d.npz", "test", 128)
            self.assertIn("128", str(raised.exception))

    def test_runs_of_unequal_length_are_all_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            self._dataset(directory, runs=[384, 384, 384])
            evaluator.check_rollout_horizon(directory, "d.npz", "test", 64)
            with self.assertRaises(ValueError):
                evaluator.check_rollout_horizon(directory, "d.npz", "test", 288)

    def test_a_dataset_without_times_is_not_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "d.npz"
            np.savez(path, X_test=np.zeros((10, 3, 4, 4), dtype=np.float32))
            evaluator.check_rollout_horizon(directory, "d.npz", "test", 3)


class BlowupStepTest(unittest.TestCase):
    """The stability column: where a trajectory first loses all predictive skill."""

    CURVES = np.array(
        [
            [0.1, 0.5, 2.0, 3.0],  # crosses at step 3
            [0.1, 0.2, 0.3, 0.4],  # never crosses
            [0.1, np.nan, 0.3, 0.4],  # non-finite at step 2 counts as a blow-up
            [2.0, 2.0, 2.0, 2.0],  # already gone at step 1
        ]
    )

    def test_the_first_crossing_is_reported_one_based(self):
        steps = evaluator.compute_blowup_step(self.CURVES)
        self.assertEqual(list(steps[[0, 2, 3]]), [3.0, 2.0, 1.0])

    def test_a_trajectory_that_never_blows_up_is_infinite(self):
        self.assertEqual(evaluator.compute_blowup_step(self.CURVES)[1], np.inf)

    def test_the_quantiles_report_a_step_that_was_reached(self):
        steps = evaluator.compute_blowup_step(self.CURVES)
        self.assertEqual(evaluator.blowup_quantile(steps, 0.5), 2.0)
        self.assertEqual(evaluator.blowup_quantile(steps, 0.25), 1.0)

    def test_a_mostly_surviving_run_reports_an_infinite_median(self):
        surviving = np.array([[0.1, 0.2]] * 3 + [[0.1, 2.0]])
        steps = evaluator.compute_blowup_step(surviving)
        self.assertEqual(evaluator.blowup_quantile(steps, 0.5), np.inf)
        self.assertEqual(float(np.mean(~np.isfinite(steps))), 0.75)


class HighBandFractionTest(unittest.TestCase):
    """The share of the kinetic energy in the shells the spectral layers cannot write."""

    @staticmethod
    def _field(side, low_k, high_k):
        """(1, S, S, 2, 1) with u a single low mode and v a single high mode."""
        axis = torch.arange(side, dtype=torch.float32) * (2 * math.pi / side)
        u = torch.cos(low_k * axis)[None, :].expand(side, side)
        v = torch.cos(high_k * axis)[None, :].expand(side, side)
        return torch.stack([u, v], dim=-1)[None, ..., None]

    def test_two_equal_modes_split_the_energy_at_the_cutoff(self):
        field = self._field(64, low_k=2, high_k=20)
        fraction = evaluator.compute_high_band_fraction_per_step(field, (0, 1), 16)
        self.assertEqual(fraction.shape, (1, 1))
        self.assertAlmostEqual(float(fraction[0, 0]), 0.5, places=6)

    def test_a_field_entirely_below_the_cutoff_reports_nothing_above_it(self):
        field = self._field(64, low_k=2, high_k=4)
        fraction = evaluator.compute_high_band_fraction_per_step(field, (0, 1), 16)
        self.assertAlmostEqual(float(fraction[0, 0]), 0.0, places=12)

    def test_a_field_entirely_above_the_cutoff_reports_all_of_it(self):
        field = self._field(64, low_k=18, high_k=20)
        fraction = evaluator.compute_high_band_fraction_per_step(field, (0, 1), 16)
        self.assertAlmostEqual(float(fraction[0, 0]), 1.0, places=9)

    def test_the_energy_is_summed_beyond_the_range_of_float32(self):
        # A diverging rollout overflows float32; in float64 the band is still a number.
        field = self._field(64, low_k=2, high_k=20) * 1e30
        fraction = evaluator.compute_high_band_fraction_per_step(field, (0, 1), 16)
        self.assertAlmostEqual(float(fraction[0, 0]), 0.5, places=6)


class TestTimeOptimisationHorizonTest(unittest.TestCase):
    """The horizon TTO adapts on, and the free rollout that follows the adapted weights."""

    SIDE = 8
    STEPS = 6

    def _harness(self):
        """A tiny FNO2d and the arguments ``run_test_time_optimization`` needs."""
        from torch.utils.data import DataLoader, TensorDataset

        from models.fno import FNO2d
        from utils.criterion import residual_scales_from_truth
        from utils.utilities import torch2dgrid_2d

        torch.manual_seed(0)
        data_config = dict(DATA_CONFIG, nx=self.SIDE, ny=self.SIDE)
        physics = physics_from_config(data_config)
        device = torch.device("cpu")
        forcing = build_forcing_for_data(physics, (self.SIDE, self.SIDE), device=device)
        grid = torch2dgrid_2d(self.SIDE, self.SIDE, form="periodic", device=device)
        model = FNO2d(
            in_dim=3 + grid.shape[-1],
            out_dim=3,
            modes1=[2, 2],
            modes2=[2, 2],
            fc_dim=8,
            layers=[4, 4, 4],
            act="gelu",
        )
        # (B, S, S, T, C): the loader hands one trajectory at a time.
        sequence = 0.1 * torch.randn(2, self.SIDE, self.SIDE, self.STEPS, 3)
        loader = DataLoader(TensorDataset(sequence, sequence), batch_size=1, shuffle=False)
        truth = torch.cat([sequence[..., :1, :], sequence], dim=-2).permute(0, 1, 2, 4, 3)
        scales = residual_scales_from_truth(
            truth.permute(0, 3, 1, 2, 4), physics, physics.dt * self.STEPS
        )
        return {
            "model": model,
            "reference_state": evaluator.clone_model_state(model),
            "test_loader": loader,
            "device": device,
            "grid": grid,
            "forcing": forcing,
            "norm_mean": torch.zeros(3, 1, 1),
            "norm_std": torch.ones(3, 1, 1),
            "use_residual": False,
            "model_name": "fno2d",
            "cont_weight": 1.0,
            "momx_weight": 0.0,
            "momy_weight": 0.0,
            "physics": physics,
            "scales": scales,
        }

    def _recorded_horizons(self, tto_cfg):
        """Every ``rollout_steps`` ``run_test_time_optimization`` rolls out with."""
        harness = self._harness()
        horizons = []
        original = evaluator.autoregressive_rollout

        def recording(model, initial_condition, grid, rollout_steps, use_residual=False):
            horizons.append(rollout_steps)
            return original(model, initial_condition, grid, rollout_steps, use_residual)

        with unittest.mock.patch.object(evaluator, "autoregressive_rollout", recording):
            result = evaluator.run_test_time_optimization(tto_cfg=tto_cfg, **harness)
        return horizons, result

    def test_adapt_steps_shortens_the_adapted_horizon_but_not_the_evaluated_one(self):
        horizons, result = self._recorded_horizons(
            {"num_iter": 2, "base_lr": 1e-4, "adapt_steps": 2}
        )
        # Two adaptation iterations at the shortened horizon, then the full evaluation,
        # for each of the two trajectories.
        self.assertEqual(horizons, [2, 2, self.STEPS, 2, 2, self.STEPS])
        self.assertEqual(result[1].shape[-1], self.STEPS + 1)

    def test_without_adapt_steps_the_whole_horizon_is_adapted_on(self):
        horizons, _ = self._recorded_horizons({"num_iter": 2, "base_lr": 1e-4})
        self.assertEqual(horizons, [self.STEPS] * 6)

    def test_the_free_rollout_trace_covers_every_trajectory_and_step(self):
        harness = self._harness()
        _, _, _, _, free_trace = evaluator.run_test_time_optimization(
            tto_cfg={"num_iter": 1, "base_lr": 1e-4, "free_rollout_steps": 5},
            mode_cutoff=2,
            **harness,
        )
        self.assertEqual(free_trace.shape, (5, 2, 2))

    def test_the_free_rollout_band_can_be_set_apart_from_the_metric_band(self):
        # The T = 64 metrics are reported above the model's own cutoff, while the free
        # rollout has to be measured over the band the plain arms were measured over.
        harness = self._harness()
        traces = {}
        for band in (1, 3):
            _, _, _, _, traces[band] = evaluator.run_test_time_optimization(
                tto_cfg={
                    "num_iter": 1,
                    "base_lr": 1e-4,
                    "free_rollout_steps": 4,
                    "free_rollout_band_cutoff": band,
                },
                mode_cutoff=2,
                **harness,
            )
        self.assertFalse(np.allclose(traces[1][..., 1], traces[3][..., 1]))

    def test_a_trajectory_that_goes_non_finite_is_padded_to_the_full_length(self):
        # free_rollout stops the moment the state goes non-finite, so without the padding
        # the traces of two trajectories would not share a time axis. The padding sits
        # after the first crossing, so blowup_steps reports the same step either way.
        short = np.array([[[1.0, 0.1]], [[np.inf, 0.1]]])
        with unittest.mock.patch.object(evaluator, "free_rollout", lambda *args, **kwargs: short):
            _, _, _, _, trace = evaluator.run_test_time_optimization(
                tto_cfg={"num_iter": 1, "base_lr": 1e-4, "free_rollout_steps": 5},
                mode_cutoff=2,
                **self._harness(),
            )
        self.assertEqual(trace.shape, (5, 2, 2))
        self.assertTrue(np.isinf(trace[2:, :, 0]).all())
        self.assertEqual(
            evaluator.blowup_steps(trace[..., 0]), evaluator.blowup_steps(short[..., 0]) * 2
        )

    def test_the_free_rollout_refuses_a_band_it_was_never_given(self):
        with self.assertRaises(ValueError):
            evaluator.run_test_time_optimization(
                tto_cfg={"num_iter": 1, "base_lr": 1e-4, "free_rollout_steps": 5},
                **self._harness(),
            )

    def test_no_free_rollout_is_run_unless_it_is_asked_for(self):
        harness = self._harness()
        result = evaluator.run_test_time_optimization(
            tto_cfg={"num_iter": 1, "base_lr": 1e-4}, **harness
        )
        self.assertIsNone(result[4])


class TrajectoryLimitTest(unittest.TestCase):
    """The evaluation can be cut to the first few trajectories of a split.

    A figure needs one trajectory of one seed where the campaign evaluates the whole
    split, and the split is ordered, so the first N are the trajectories the full
    evaluation begins with. A short run therefore covers the same data --- which is
    the same fields only where the rollout is deterministic, and not under TTO.
    """

    def test_no_limit_leaves_the_split_whole(self):
        dataset = ["a", "b", "c"]
        self.assertIs(evaluator.limit_trajectories(dataset, None), dataset)

    def test_a_limit_keeps_the_first_trajectories_in_order(self):
        limited = evaluator.limit_trajectories(["a", "b", "c", "d"], 2)
        self.assertEqual([limited[index] for index in range(len(limited))], ["a", "b"])

    def test_a_limit_past_the_end_keeps_every_trajectory(self):
        limited = evaluator.limit_trajectories(["a", "b"], 10)
        self.assertEqual([limited[index] for index in range(len(limited))], ["a", "b"])
