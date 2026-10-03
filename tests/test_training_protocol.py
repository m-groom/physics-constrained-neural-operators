"""Training protocol: seeded reproducibility, per-run logs, baseline diagnostics."""

import csv
import math
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from experiments import test_operator_AR_2d as tester
from experiments import train_operator_AR_PINO_2d as trainer
from experiments.run_protocol import loader_kwargs, seed_everything
from models.fno import FNO2d
from utils.criterion import (
    build_forcing_for_data,
    physics_from_config,
    residual_scales_from_truth,
)
from utils.utilities import save_checkpoint, torch2dgrid_2d

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


def _train_config(save_dir, epochs=2):
    return {
        "train": {
            "epochs": epochs,
            "grad_clip": 1.0,
            "noise_std": 0.0,
            "patience": 0,
            "save_dir": str(save_dir),
            "save_name": "tiny.pt",
        }
    }


def _run_two_epochs(save_dir, seed, resume=False, output_constraint=None, curl_weight=0.0):
    """Train a tiny FNO on synthetic pairs; return the per-epoch train L2 values."""
    seed_everything(seed)
    physics = physics_from_config(DATA_CONFIG)
    device = torch.device("cpu")
    forcing = build_forcing_for_data(physics, (S, S), device=device)
    x = torch.randn(8, S, S, 3)
    y = x + 0.01 * torch.randn(8, S, S, 3)
    loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=True, **loader_kwargs(seed))
    model = FNO2d(
        in_dim=7,
        out_dim=3,
        modes1=[4],
        modes2=[4],
        fc_dim=8,
        layers=[8, 8],
        act="gelu",
        output_constraint=output_constraint,
        physics=physics,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1], gamma=0.5)
    config = _train_config(save_dir)
    grid = torch2dgrid_2d(S, S, form="periodic", device=device, dtype=torch.float32)
    truth = torch.stack([x, y], dim=-1).permute(0, 3, 1, 2, 4)
    scales = residual_scales_from_truth(truth, physics, physics.dt)
    if output_constraint is not None:
        model.set_output_normalizer(torch.zeros(3, 1, 1), torch.ones(3, 1, 1))

    trainer.train_step_ahead(
        model,
        loader,
        optimizer,
        scheduler,
        config,
        device,
        grid,
        test_loader=None,
        use_tqdm=False,
        weight_dict={
            "data_weight": 1.0,
            "cont_weight": 0.23,
            "ic_weight": 0.0,
            "momx_weight": 0.0153,
            "momy_weight": 0.0153,
            "curl_weight": curl_weight,
        },
        forcing=forcing,
        physics=physics,
        scales=scales,
        denorm_mean=torch.zeros(1, 1, 3),
        denorm_std=torch.ones(1, 1, 3),
        use_residual=True,
        resume=resume,
    )
    with open(trainer.get_training_log_path(config), newline="") as handle:
        return list(csv.DictReader(handle))


class TrainingDeterminismTest(unittest.TestCase):
    def test_the_same_seed_reproduces_the_training_losses(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            a = [row["train_l2"] for row in _run_two_epochs(first, seed=7)]
            b = [row["train_l2"] for row in _run_two_epochs(second, seed=7)]
        self.assertEqual(a, b)

    def test_a_different_seed_changes_the_training_losses(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            a = [row["train_l2"] for row in _run_two_epochs(first, seed=7)]
            b = [row["train_l2"] for row in _run_two_epochs(second, seed=8)]
        self.assertNotEqual(a, b)


class TrainingLogTest(unittest.TestCase):
    def test_a_fresh_run_truncates_the_log(self):
        with tempfile.TemporaryDirectory() as save_dir:
            _run_two_epochs(save_dir, seed=7)
            rows = _run_two_epochs(save_dir, seed=7)
        self.assertEqual([row["epoch"] for row in rows], ["1", "2"])

    def test_a_resumed_run_appends_to_the_log(self):
        with tempfile.TemporaryDirectory() as save_dir:
            _run_two_epochs(save_dir, seed=7)
            rows = _run_two_epochs(save_dir, seed=7, resume=True)
        self.assertEqual([row["epoch"] for row in rows], ["1", "2", "1", "2"])


class BaselineDiagnosticsTest(unittest.TestCase):
    def test_an_unconstrained_model_still_logs_divergence_and_mean_velocity(self):
        with tempfile.TemporaryDirectory() as save_dir:
            rows = _run_two_epochs(save_dir, seed=7)
        self.assertTrue(math.isfinite(float(rows[-1]["train_div_max"])))
        self.assertTrue(math.isfinite(float(rows[-1]["train_mean_vel"])))
        self.assertGreater(float(rows[-1]["train_div_max"]), 0.0)


class CurlLossTermTest(unittest.TestCase):
    """The curl-matching term reaches the optimiser and the log (#28).

    Validation stays the one-step data relative L2, so `_best.pt` is selected exactly
    as in #23 whether or not the term is on.
    """

    def test_the_term_is_logged_as_zero_when_its_weight_is_zero(self):
        with tempfile.TemporaryDirectory() as save_dir:
            rows = _run_two_epochs(save_dir, seed=7)
        self.assertEqual(float(rows[-1]["train_curl_rel"]), 0.0)

    def test_the_term_is_logged_and_positive_when_it_is_switched_on(self):
        with tempfile.TemporaryDirectory() as save_dir:
            rows = _run_two_epochs(save_dir, seed=7, curl_weight=1.0)
        self.assertGreater(float(rows[-1]["train_curl_rel"]), 0.0)

    def test_switching_the_term_on_changes_the_trained_weights(self):
        with tempfile.TemporaryDirectory() as off, tempfile.TemporaryDirectory() as on:
            without = [row["train_l2"] for row in _run_two_epochs(off, seed=7)]
            with_term = [row["train_l2"] for row in _run_two_epochs(on, seed=7, curl_weight=1.0)]
        # `train_l2` is the mean over batches, and the curl gradient is applied from
        # the first batch onwards, so both epochs move.
        self.assertNotEqual(without, with_term)

    def test_a_formulation_without_velocity_refuses_the_term(self):
        physics = physics_from_config({**DATA_CONFIG, "formulation": "vorticity"})
        with self.assertRaises(ValueError):
            trainer.compute_curl_loss(
                FNO2d(
                    in_dim=5,
                    out_dim=1,
                    modes1=[4],
                    modes2=[4],
                    fc_dim=8,
                    layers=[8, 8],
                    act="gelu",
                ),
                torch.randn(2, S, S, 1),
                torch.randn(2, S, S, 1),
                torch.zeros(1, 1, 1),
                torch.ones(1, 1, 1),
                physics,
            )

    def test_a_run_without_normalisation_statistics_refuses_the_term(self):
        physics = physics_from_config(DATA_CONFIG)
        model = FNO2d(
            in_dim=7, out_dim=3, modes1=[4], modes2=[4], fc_dim=8, layers=[8, 8], act="gelu"
        )
        with self.assertRaises(ValueError):
            trainer.compute_curl_loss(
                model, torch.randn(2, S, S, 3), torch.randn(2, S, S, 3), None, None, physics
            )

    def test_the_term_carries_a_gradient_back_to_the_model(self):
        physics = physics_from_config(DATA_CONFIG)
        model = FNO2d(
            in_dim=7, out_dim=3, modes1=[4], modes2=[4], fc_dim=8, layers=[8, 8], act="gelu"
        )
        grid = torch2dgrid_2d(S, S, form="periodic", device=torch.device("cpu"))
        x = torch.randn(2, S, S, 3)
        pred = model(torch.cat((x, grid.unsqueeze(0).expand(2, -1, -1, -1)), dim=-1))
        loss = trainer.compute_curl_loss(
            model,
            pred,
            torch.randn(2, S, S, 3),
            torch.zeros(1, 1, 3),
            torch.ones(1, 1, 3),
            physics,
        )
        loss.backward()
        self.assertTrue(
            any(
                parameter.grad is not None and torch.any(parameter.grad != 0)
                for parameter in model.parameters()
            )
        )


class UnforcedVorticityDiagnosticsTest(unittest.TestCase):
    """An unforced vorticity run reports its residual as missing, and still runs.

    E3's coarse 64^2 field carries no forcing at all (#14), so the vorticity control of
    #28 has nothing to score its single residual against -- that residual is relative to
    the forcing, following PINO. Reporting it as missing is what lets the run happen;
    raising aborts a run that never asked for the residual in the first place.
    """

    def _evaluate(self, forcing):
        config = {**DATA_CONFIG, "formulation": "vorticity", "forcing": forcing}
        physics = physics_from_config(config)
        device = torch.device("cpu")
        forcing_field = build_forcing_for_data(physics, (S, S), device=device)
        x = torch.randn(4, S, S, 1)
        y = x + 0.01 * torch.randn(4, S, S, 1)
        loader = DataLoader(TensorDataset(x, y), batch_size=4)
        model = FNO2d(
            in_dim=5, out_dim=1, modes1=[4], modes2=[4], fc_dim=8, layers=[8, 8], act="gelu"
        )
        grid = torch2dgrid_2d(S, S, form="periodic", device=device, dtype=torch.float32)
        return trainer.evaluate_step_ahead(
            model, loader, device, grid, forcing_field, physics, None, use_residual=True
        )

    def test_an_unforced_run_evaluates_and_reports_the_residual_as_missing(self):
        result = self._evaluate({"type": "none"})
        self.assertTrue(math.isfinite(result[0]))
        for index in range(1, 8):
            self.assertTrue(math.isnan(result[index]), f"entry {index} should be missing")

    def test_a_forced_run_still_reports_a_residual(self):
        result = self._evaluate({"type": "pino_cos4y", "amplitude": 1.0, "wavenumber": 4})
        self.assertTrue(math.isfinite(result[0]))
        self.assertTrue(math.isfinite(result[2]))


class CheckpointPathTest(unittest.TestCase):
    """A checkpoint lands inside its run directory, trailing slash or not."""

    def test_a_save_dir_without_a_trailing_slash_still_writes_inside_it(self):
        with tempfile.TemporaryDirectory() as save_dir:
            run_dir = Path(save_dir) / "FNO_vort_seed1"
            with unittest.mock.patch.object(torch.cuda, "is_available", return_value=True):
                save_checkpoint(str(run_dir), "FNO2d.pt", torch.nn.Linear(2, 2), epoch=0)

            self.assertTrue((run_dir / "FNO2d.pt").exists())
            self.assertFalse(Path(f"{run_dir}FNO2d.pt").exists())


def _run_with_scripted_validation(save_dir, noise_std, validation_losses):
    """Train with a scripted validation curve; return the epoch ``_best.pt`` holds.

    The validation loss is supplied rather than measured, so the test states the situation
    the guard is for: the clean loss is at its lowest before the noise ramp has finished.
    """
    seed_everything(3)
    physics = physics_from_config(DATA_CONFIG)
    device = torch.device("cpu")
    forcing = build_forcing_for_data(physics, (S, S), device=device)
    x = torch.randn(4, S, S, 3)
    y = x + 0.01 * torch.randn(4, S, S, 3)
    loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)
    model = FNO2d(
        in_dim=7,
        out_dim=3,
        modes1=[4],
        modes2=[4],
        fc_dim=8,
        layers=[8, 8],
        act="gelu",
        physics=physics,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[99], gamma=0.5)
    config = _train_config(save_dir, epochs=len(validation_losses))
    config["train"]["noise_std"] = noise_std
    config["train"]["noise_warmup_frac"] = 0.2
    config["train"]["eval_step"] = 1
    grid = torch2dgrid_2d(S, S, form="periodic", device=device, dtype=torch.float32)
    truth = torch.stack([x, y], dim=-1).permute(0, 3, 1, 2, 4)
    scales = residual_scales_from_truth(truth, physics, physics.dt)

    scripted = iter(validation_losses)
    saved = {}

    def fake_evaluation(*args, **kwargs):
        return (next(scripted), *([0.0] * 9), None, None)

    def record_checkpoint(save_dir, name, model, epoch, optimizer=None, scheduler=None):
        # Stands in for save_checkpoint, which writes nothing without CUDA. Recording the
        # epoch is what the guard is about, and it keeps the test off the GPU path: with
        # torch.cuda.is_available patched to True the optimizer step calls into the CUDA
        # graph API, which fails on a machine that has no driver.
        saved[name] = epoch

    with (
        unittest.mock.patch.object(trainer, "evaluate_step_ahead", fake_evaluation),
        unittest.mock.patch.object(trainer, "save_checkpoint", record_checkpoint),
    ):
        trainer.train_step_ahead(
            model,
            loader,
            optimizer,
            scheduler,
            config,
            device,
            grid,
            test_loader=loader,
            use_tqdm=False,
            weight_dict={
                "data_weight": 1.0,
                "cont_weight": 0.0,
                "ic_weight": 0.0,
                "momx_weight": 0.0,
                "momy_weight": 0.0,
            },
            forcing=forcing,
            physics=physics,
            scales=scales,
            denorm_mean=torch.zeros(1, 1, 3),
            denorm_std=torch.ones(1, 1, 3),
            use_residual=True,
        )
    return saved.get("tiny_best.pt")


class NoiseRampTest(unittest.TestCase):
    """The input-noise ramp: off during the warmup, full amplitude at the final epoch."""

    def test_no_noise_before_the_warmup_fraction(self):
        self.assertEqual(trainer.noise_amplitude(0, 100, 2e-2, 0.2), 0.0)
        self.assertEqual(trainer.noise_amplitude(19, 100, 2e-2, 0.2), 0.0)

    def test_the_ramp_reaches_the_full_amplitude_at_the_final_epoch(self):
        self.assertGreater(trainer.noise_amplitude(50, 100, 2e-2, 0.2), 0.0)
        self.assertLess(trainer.noise_amplitude(50, 100, 2e-2, 0.2), 2e-2)
        self.assertEqual(trainer.noise_amplitude(99, 100, 2e-2, 0.2), 2e-2)

    def test_a_zero_standard_deviation_switches_the_noise_off(self):
        self.assertEqual(trainer.noise_amplitude(99, 100, 0.0, 0.2), 0.0)


class NoiseGuardedCheckpointTest(unittest.TestCase):
    """``_best.pt`` may not come from an epoch whose noise has not reached full amplitude.

    Noise injection raises the clean validation loss while it ramps, so the unguarded
    selection keeps a pre-noise epoch and silently discards the stabilisation (#42).
    """

    LOSSES = (0.10, 0.20, 0.30, 0.40, 0.50)

    def test_without_noise_the_best_validation_epoch_is_selected(self):
        with tempfile.TemporaryDirectory() as save_dir:
            epoch = _run_with_scripted_validation(save_dir, 0.0, self.LOSSES)
        self.assertEqual(epoch, 0)

    def test_with_noise_the_selected_epoch_is_one_trained_at_full_amplitude(self):
        with tempfile.TemporaryDirectory() as save_dir:
            epoch = _run_with_scripted_validation(save_dir, 2e-2, self.LOSSES)
        self.assertEqual(epoch, len(self.LOSSES) - 1)

    def test_a_noise_level_yaml_read_as_a_string_still_trains(self):
        # YAML 1.1 reads "2e-2" (an exponent without a decimal point) as a string.
        with tempfile.TemporaryDirectory() as save_dir:
            epoch = _run_with_scripted_validation(save_dir, "2e-2", self.LOSSES)
        self.assertEqual(epoch, len(self.LOSSES) - 1)


def _tiny_model_and_grid(seed=0):
    """A small FNO and its grid, for the pushforward tests."""
    seed_everything(seed)
    physics = physics_from_config(DATA_CONFIG)
    model = FNO2d(
        in_dim=7,
        out_dim=3,
        modes1=[4],
        modes2=[4],
        fc_dim=8,
        layers=[8, 8],
        act="gelu",
        physics=physics,
    )
    grid = torch2dgrid_2d(S, S, form="periodic", device=torch.device("cpu"), dtype=torch.float32)
    return model, grid


class RolloutPairsTest(unittest.TestCase):
    """Re-targeting one-step pairs ``steps`` ahead, for pushforward training."""

    @staticmethod
    def _chained(n, channels=3, side=4):
        """A dataset stand-in whose pairs chain: ``y[i]`` is ``X[i + 1]``."""
        frames = torch.randn(n + 1, side, side, channels)
        pairs = unittest.mock.Mock()
        pairs.X_data = frames[:-1]
        pairs.y_data = frames[1:]
        return pairs

    def test_one_step_keeps_every_pair_unchanged(self):
        pairs = self._chained(6)
        dataset = trainer.RolloutPairs(pairs, steps=1)
        self.assertEqual(len(dataset), 6)
        x, y = dataset[3]
        self.assertTrue(torch.equal(x, pairs.X_data[3]))
        self.assertTrue(torch.equal(y, pairs.y_data[3]))

    def test_two_steps_target_the_second_frame_ahead(self):
        pairs = self._chained(6)
        dataset = trainer.RolloutPairs(pairs, steps=2)
        # Five starts survive: the last pair has nothing to chain into.
        self.assertEqual(len(dataset), 5)
        x, y = dataset[3]
        self.assertTrue(torch.equal(x, pairs.X_data[3]))
        self.assertTrue(torch.equal(y, pairs.y_data[4]))

    def test_a_break_in_the_chain_drops_the_starts_that_would_cross_it(self):
        pairs = self._chained(6)
        # A realisation boundary: pair 2's target is not pair 3's input.
        pairs.X_data = pairs.X_data.clone()
        pairs.X_data[3] = torch.randn_like(pairs.X_data[3])
        dataset = trainer.RolloutPairs(pairs, steps=2)
        starts = [int(s) for s in dataset.starts]
        self.assertNotIn(2, starts)
        self.assertEqual(starts, [0, 1, 3, 4])

    def test_three_steps_drop_every_start_that_crosses_the_break(self):
        pairs = self._chained(6)
        pairs.X_data = pairs.X_data.clone()
        pairs.X_data[3] = torch.randn_like(pairs.X_data[3])
        dataset = trainer.RolloutPairs(pairs, steps=3)
        self.assertEqual([int(s) for s in dataset.starts], [0, 3])
        x, y = dataset[1]
        self.assertTrue(torch.equal(x, pairs.X_data[3]))
        self.assertTrue(torch.equal(y, pairs.y_data[5]))


class PushforwardTest(unittest.TestCase):
    """The pushforward trick: roll the input forward without gradients, train one step."""

    def test_zero_steps_returns_the_input_untouched(self):
        model, grid = _tiny_model_and_grid()
        x = torch.randn(2, S, S, 3)
        self.assertIs(trainer.pushforward_input(model, x, grid, 0, use_residual=True), x)

    def test_one_step_returns_the_model_step_with_no_gradient_attached(self):
        model, grid = _tiny_model_and_grid()
        x = torch.randn(2, S, S, 3)
        rolled = trainer.pushforward_input(model, x, grid, 1, use_residual=True)
        with torch.no_grad():
            expected = trainer.model_step(model, x, grid, use_residual=True)
        self.assertTrue(torch.allclose(rolled, expected))
        self.assertFalse(rolled.requires_grad)
        self.assertIsNone(rolled.grad_fn)

    def test_the_detached_chain_gives_different_gradients_from_a_joined_one(self):
        model, grid = _tiny_model_and_grid()
        x = torch.randn(2, S, S, 3)

        rolled = trainer.pushforward_input(model, x, grid, 1, use_residual=True)
        trainer.model_step(model, rolled, grid, use_residual=True).pow(2).mean().backward()
        detached = [p.grad.clone() for p in model.parameters() if p.grad is not None]

        model.zero_grad()
        joined = trainer.model_step(model, x, grid, use_residual=True)
        trainer.model_step(model, joined, grid, use_residual=True).pow(2).mean().backward()
        through = [p.grad.clone() for p in model.parameters() if p.grad is not None]

        self.assertEqual(len(detached), len(through))
        self.assertTrue(
            any(not torch.allclose(a, b) for a, b in zip(detached, through, strict=True))
        )


class ResolvePushforwardStepsTest(unittest.TestCase):
    """`rollout_steps` and `pushforward` have to agree, and be refused when they do not."""

    def test_the_default_configuration_takes_no_detached_step(self):
        self.assertEqual(trainer.resolve_pushforward_steps({}), 0)
        self.assertEqual(trainer.resolve_pushforward_steps({"rollout_steps": 1}), 0)

    def test_a_two_step_rollout_takes_one_detached_step(self):
        config = {"rollout_steps": 2, "pushforward": True}
        self.assertEqual(trainer.resolve_pushforward_steps(config), 1)

    def test_a_horizon_without_the_switch_is_refused(self):
        with self.assertRaises(ValueError):
            trainer.resolve_pushforward_steps({"rollout_steps": 2})

    def test_the_switch_without_a_horizon_is_refused(self):
        # Silently training one-step under `pushforward: true` would report a run as
        # stabilised that took no detached step at all.
        with self.assertRaises(ValueError):
            trainer.resolve_pushforward_steps({"pushforward": True})
        with self.assertRaises(ValueError):
            trainer.resolve_pushforward_steps({"pushforward": True, "rollout_steps": 1})


class EvaluationVariantSuffixTest(unittest.TestCase):
    """The band cutoff names the file, so two bands cannot collide under one name."""

    def test_the_default_evaluation_keeps_its_bare_name(self):
        self.assertEqual(tester.evaluation_variant_suffix("test", None), "")

    def test_an_overridden_band_names_the_file(self):
        self.assertEqual(tester.evaluation_variant_suffix("test", None, 16), "_k16")

    def test_the_band_composes_with_the_split_and_the_horizon(self):
        self.assertEqual(
            tester.evaluation_variant_suffix("test_time", 448, 16), "_test_time_k16_T448"
        )
