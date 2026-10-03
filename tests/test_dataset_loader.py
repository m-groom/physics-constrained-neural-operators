"""NSLoader2D reads the dataset file it is given, not one fixed name."""

import numpy as np
import pytest

from experiments.data_utils.datasets_dedalus import NSLoader2D


def write_dataset(directory, name):
    """A four-sample one-channel dataset under an arbitrary file name."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal((4, 1, 8, 8)).astype(np.float32)
    path = directory / name
    np.savez(path, X_train=data, y_train=data + 1.0)
    return path


def test_the_dataset_file_can_be_named(tmp_path):
    write_dataset(tmp_path, "pino_kf_vorticity.npz")

    loader = NSLoader2D(
        str(tmp_path),
        state="train",
        velocity_channels=None,
        filename="pino_kf_vorticity.npz",
    )

    assert loader.X_data.shape == (4, 8, 8, 1)  # (N, H, W, C)
    assert loader.mean.shape == (1, 1, 1)


def test_the_default_name_is_unchanged(tmp_path):
    write_dataset(tmp_path, "pino_kf_vorticity.npz")

    with pytest.raises(FileNotFoundError, match="kolmogorov_dataset"):
        NSLoader2D(str(tmp_path), state="train", velocity_channels=None)
