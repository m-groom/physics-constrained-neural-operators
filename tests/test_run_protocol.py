"""Run-protocol helpers: seeding, per-run logs, manifests, run-directory guards."""

import csv
import json
import math
import random
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from experiments.run_protocol import (
    best_logged_eval_loss,
    check_run_directory,
    loader_kwargs,
    seed_everything,
    start_training_log,
    write_run_manifest,
)


def _draws():
    """One draw from each of the three generators a run depends on."""
    return (random.random(), float(np.random.rand()), float(torch.rand(1).item()))


class SeedEverythingTest(unittest.TestCase):
    def test_same_seed_gives_the_same_draws(self):
        seed_everything(1234)
        first = _draws()
        seed_everything(1234)
        self.assertEqual(_draws(), first)

    def test_different_seeds_give_different_draws(self):
        seed_everything(1234)
        first = _draws()
        seed_everything(5678)
        self.assertNotEqual(_draws(), first)

    def test_shuffled_dataloader_order_follows_the_seed(self):
        dataset = TensorDataset(torch.arange(32).unsqueeze(1))

        def order(seed):
            seed_everything(seed)
            loader = DataLoader(dataset, batch_size=4, shuffle=True, **loader_kwargs(seed))
            return [int(value) for (batch,) in loader for value in batch]

        self.assertEqual(order(1234), order(1234))
        self.assertNotEqual(order(1234), order(5678))


COLUMNS = ["epoch", "eval_l2"]


class TrainingLogFileTest(unittest.TestCase):
    def _log(self, directory):
        return str(Path(directory) / "run" / "model_training_log.csv")

    def _write_rows(self, path, rows):
        with open(path, "a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            for row in rows:
                writer.writerow(row)

    def test_a_fresh_run_starts_a_new_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._log(directory)
            start_training_log(path, COLUMNS)
            self._write_rows(path, [{"epoch": 1, "eval_l2": 0.5}])
            start_training_log(path, COLUMNS)
            with open(path, newline="") as handle:
                self.assertEqual(list(csv.DictReader(handle)), [])

    def test_a_resume_keeps_the_existing_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._log(directory)
            start_training_log(path, COLUMNS)
            self._write_rows(path, [{"epoch": 1, "eval_l2": 0.5}])
            start_training_log(path, COLUMNS, resume=True)
            with open(path, newline="") as handle:
                self.assertEqual([row["epoch"] for row in csv.DictReader(handle)], ["1"])

    def test_best_logged_eval_loss_ignores_the_unevaluated_epochs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._log(directory)
            start_training_log(path, COLUMNS)
            self._write_rows(
                path,
                [
                    {"epoch": 1, "eval_l2": 0.5},
                    {"epoch": 2, "eval_l2": float("nan")},
                    {"epoch": 3, "eval_l2": 0.25},
                ],
            )
            self.assertEqual(best_logged_eval_loss(path), 0.25)

    def test_best_logged_eval_loss_of_a_missing_log_is_infinite(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(best_logged_eval_loss(self._log(directory)), math.inf)


class RunManifestTest(unittest.TestCase):
    def test_the_manifest_records_the_run(self):
        config = {"train": {"epochs": 2}, "data": {"nu": 0.002}}
        with tempfile.TemporaryDirectory() as directory:
            path = write_run_manifest(
                directory, config, seed=7, checkpoint_paths=["a.pt", "a_best.pt"]
            )
            self.assertEqual(path, str(Path(directory) / "run_manifest.json"))
            manifest = json.loads(Path(path).read_text())

        self.assertEqual(manifest["config"], config)
        self.assertEqual(manifest["seed"], 7)
        self.assertEqual(manifest["checkpoint_paths"], ["a.pt", "a_best.pt"])
        self.assertEqual(len(manifest["git_commit"]), 40)
        self.assertIn("start_time", manifest)
        self.assertIn("slurm_job_id", manifest)


class RunDirectoryGuardTest(unittest.TestCase):
    def test_an_existing_checkpoint_stops_a_fresh_run(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            checkpoint.write_text("weights")
            with self.assertRaises(FileExistsError):
                check_run_directory([str(checkpoint)])
            check_run_directory([str(checkpoint)], overwrite=True)
            check_run_directory([str(checkpoint)], resume=True)

    def test_an_empty_run_directory_is_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            check_run_directory([str(Path(directory) / "model.pt")])
