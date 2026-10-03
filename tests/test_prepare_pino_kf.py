import numpy as np
import torch

from experiments.data_utils.datasets_dedalus import NSLoader2D
from experiments.data_utils.prepare_pino_kf import (
    build_dataset,
    invert_vorticity,
    split_trajectories,
)

N = 64
L = 2 * np.pi


def grid(n=N):
    """Physical grid with axis -2 = x and axis -1 = y, matching utils/criterion.py."""
    c = np.linspace(0.0, L, n, endpoint=False)
    return np.meshgrid(c, c, indexing="ij")


def test_invert_vorticity_matches_the_taylor_green_solution():
    # w = 2 sin(x) sin(y) with the criterion.py conventions
    #   Delta psi = -w, u = d psi / dy, v = -d psi / dx
    # gives psi = sin(x) sin(y), u = sin(x) cos(y), v = -cos(x) sin(y) and,
    # from Delta p = -div((u.grad)u), p = (1/4)(cos(2x) + cos(2y)).
    x, y = grid()
    w = 2.0 * np.sin(x) * np.sin(y)

    psi, u, v, p = invert_vorticity(w[None])

    assert psi.shape == u.shape == v.shape == p.shape == (1, N, N)
    np.testing.assert_allclose(psi[0], np.sin(x) * np.sin(y), atol=1e-12)
    np.testing.assert_allclose(u[0], np.sin(x) * np.cos(y), atol=1e-12)
    np.testing.assert_allclose(v[0], -np.cos(x) * np.sin(y), atol=1e-12)
    np.testing.assert_allclose(p[0], 0.25 * (np.cos(2 * x) + np.cos(2 * y)), atol=1e-12)


def band_limited_vorticity(rng, n=N, k_cut=20):
    """A random real field whose spectral support stops well below the grid Nyquist."""
    kx = np.fft.fftfreq(n, d=1.0 / n)
    ky = np.fft.rfftfreq(n, d=1.0 / n)
    k2 = kx[:, None] ** 2 + ky[None, :] ** 2
    spec = rng.normal(size=(n, n // 2 + 1)) + 1j * rng.normal(size=(n, n // 2 + 1))
    spec[(k2 > k_cut**2) | (k2 == 0.0)] = 0.0
    return np.fft.irfft2(spec, s=(n, n))


def ddx(f):
    kx = np.fft.fftfreq(f.shape[-2], d=1.0 / f.shape[-2])
    return np.fft.irfft2(1j * kx[:, None] * np.fft.rfft2(f), s=f.shape[-2:])


def ddy(f):
    ky = np.fft.rfftfreq(f.shape[-1], d=1.0 / f.shape[-1])
    return np.fft.irfft2(1j * ky[None, :] * np.fft.rfft2(f), s=f.shape[-2:])


def laplacian(f):
    """Spectral Laplacian, applied as the single even operator -|k|^2 (Nyquist-safe)."""
    kx = np.fft.fftfreq(f.shape[-2], d=1.0 / f.shape[-2])
    ky = np.fft.rfftfreq(f.shape[-1], d=1.0 / f.shape[-1])
    k2 = kx[:, None] ** 2 + ky[None, :] ** 2
    return np.fft.irfft2(-k2 * np.fft.rfft2(f), s=f.shape[-2:])


def test_inverted_velocity_is_divergence_free_and_reproduces_the_vorticity():
    rng = np.random.default_rng(0)
    w = np.stack([band_limited_vorticity(rng) for _ in range(3)])

    _, u, v, _ = invert_vorticity(w)

    scale = np.sqrt(np.mean(u**2 + v**2))
    assert np.abs(ddx(u) + ddy(v)).max() < 1e-12 * scale
    np.testing.assert_allclose(ddx(v) - ddy(u), w, atol=1e-10 * np.abs(w).max())


def test_pressure_satisfies_the_incompressible_poisson_equation():
    rng = np.random.default_rng(1)
    w = np.stack([band_limited_vorticity(rng) for _ in range(3)])

    _, u, v, p = invert_vorticity(w)

    lap_p = laplacian(p)
    div_adv = ddx(u) ** 2 + 2.0 * ddy(u) * ddx(v) + ddy(v) ** 2
    np.testing.assert_allclose(lap_p, -div_adv, atol=1e-9 * np.abs(div_adv).max())
    np.testing.assert_allclose(p.mean(axis=(-2, -1)), 0.0, atol=1e-12)


def test_split_trajectories_is_disjoint_sized_and_reproducible():

    split = split_trajectories(4000, n_val=200, n_test=200, seed=1234)

    assert [len(split[k]) for k in ("train", "val", "test")] == [3600, 200, 200]
    everything = np.concatenate([split["train"], split["val"], split["test"]])
    assert len(np.unique(everything)) == 4000
    np.testing.assert_array_equal(np.sort(everything), np.arange(4000))
    np.testing.assert_array_equal(
        split["test"], split_trajectories(4000, n_val=200, n_test=200, seed=1234)["test"]
    )
    assert not np.array_equal(
        split["test"], split_trajectories(4000, n_val=200, n_test=200, seed=99)["test"]
    )


def synthetic_source(tmp_path, n_traj=6, n_time=5, n=16):
    """A small band-limited vorticity file with the source layout (n_traj, n_time, x, y)."""
    rng = np.random.default_rng(7)
    w = np.stack(
        [
            np.stack([band_limited_vorticity(rng, n=n, k_cut=4) for _ in range(n_time)])
            for _ in range(n_traj)
        ]
    )
    path = tmp_path / "source.npy"
    np.save(path, w)
    return path, w


def test_vorticity_dataset_has_the_loader_layout_and_consecutive_pairs(tmp_path):

    source, w = synthetic_source(tmp_path)
    summary = build_dataset(source, tmp_path, "vorticity", seed=3, n_val=2, n_test=2)

    data = np.load(summary["dataset_path"])
    n_pairs = w.shape[1] - 1
    assert data["X_train"].shape == (2 * n_pairs, 1, 16, 16)
    assert data["X_train"].dtype == np.float32
    for state in ("train", "val", "test"):
        X, y = data[f"X_{state}"], data[f"y_{state}"]
        traj = data[f"{state}_realisations"]
        # trajectory-major ordering: index = local trajectory * n_pairs + step
        expected = w[traj].astype(np.float32)
        np.testing.assert_array_equal(X.reshape(len(traj), n_pairs, 16, 16), expected[:, :-1])
        np.testing.assert_array_equal(y.reshape(len(traj), n_pairs, 16, 16), expected[:, 1:])
        # pairs are consecutive in time, and the recorded times say so
        times = data[f"times_{state}"]
        assert times.shape == (len(traj) * n_pairs, 2)
        np.testing.assert_allclose(times[:, 1] - times[:, 0], float(data["dt"]))
        np.testing.assert_array_equal(data[f"realisation_ids_{state}"], np.repeat(traj, n_pairs))
    assert bytes(data["field"]) == b"vorticity"
    assert float(data["nu"]) == 1 / 500
    assert float(data["domain_length"]) == 2 * np.pi
    assert data["x_coords"].shape == (16,) and data["y_coords"].shape == (16,)


def test_velocity_dataset_carries_uvp_and_the_streamfunction(tmp_path):

    source, w = synthetic_source(tmp_path)
    summary = build_dataset(source, tmp_path, "velocity", seed=3, n_val=2, n_test=2)

    data = np.load(summary["dataset_path"])
    n_pairs = w.shape[1] - 1
    assert data["X_val"].shape == (2 * n_pairs, 3, 16, 16)
    assert data["psi_X_val"].shape == (2 * n_pairs, 1, 16, 16)
    traj = data["val_realisations"]
    psi, u, v, p = invert_vorticity(w[traj][:, :-1].reshape(-1, 16, 16))
    got = data["X_val"]
    np.testing.assert_allclose(got[:, 0], u.astype(np.float32), atol=1e-6)
    np.testing.assert_allclose(got[:, 1], v.astype(np.float32), atol=1e-6)
    np.testing.assert_allclose(got[:, 2], p.astype(np.float32), atol=1e-6)
    np.testing.assert_allclose(data["psi_X_val"][:, 0], psi.astype(np.float32), atol=1e-6)
    assert [bytes(c) for c in data["channels"]] == [b"u", b"v", b"p"]


def test_normaliser_matches_what_the_loader_computes_and_the_npz_round_trips(tmp_path):

    source, _ = synthetic_source(tmp_path)
    for field, velocity_channels in (("vorticity", None), ("velocity", (0, 1))):
        out = tmp_path / field
        summary = build_dataset(source, out, field, seed=3, n_val=2, n_test=2)

        stats = torch.load(summary["normaliser_path"], weights_only=False)
        loader = NSLoader2D(
            str(summary["loader_dir"]), state="train", velocity_channels=velocity_channels
        )
        torch.testing.assert_close(stats["mean"], loader.mean, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(stats["std"], loader.std, rtol=1e-5, atol=1e-6)
        assert stats["normalization"] == "global_channel_rms"
        assert stats["velocity_channels"] == velocity_channels


def test_test_split_rebuilds_rollout_trajectories_in_order(tmp_path):

    source, w = synthetic_source(tmp_path)
    summary = build_dataset(source, tmp_path, "vorticity", seed=3, n_val=2, n_test=2)
    n_pairs = w.shape[1] - 1

    loader = NSLoader2D(
        str(summary["loader_dir"]),
        state="test",
        normalizer_path=str(summary["normaliser_path"]),
        velocity_channels=None,
    )
    X, _ = loader.transform_rollout(T=n_pairs)  # (n_traj, H, W, T, C)

    traj = np.load(summary["dataset_path"])["test_realisations"]
    assert X.shape == (len(traj), 16, 16, n_pairs, 1)
    restored = X.numpy()[..., 0].transpose(0, 3, 1, 2) * loader.std[0, 0, 0].item()
    restored = restored + loader.mean[0, 0, 0].item()
    np.testing.assert_allclose(restored, w[traj][:, :-1], rtol=1e-4, atol=1e-4)


def test_the_two_datasets_describe_the_same_flow(tmp_path):
    # A full-spectrum source: unlike the band-limited fixtures above it carries energy in the
    # grid-Nyquist row and column, which no real grid velocity field can reproduce.
    rng = np.random.default_rng(11)
    w = rng.normal(size=(4, 3, 16, 16))
    source = tmp_path / "full.npy"
    np.save(source, w)

    velocity = build_dataset(source, tmp_path / "e1", "velocity", seed=5, n_val=1, n_test=1)
    vorticity = build_dataset(source, tmp_path / "e0", "vorticity", seed=5, n_val=1, n_test=1)

    vel = np.load(velocity["dataset_path"])["X_train"]
    vort = np.load(vorticity["dataset_path"])["X_train"]
    np.testing.assert_allclose(
        ddx(vel[:, 1]) - ddy(vel[:, 0]), vort[:, 0], atol=1e-4 * np.abs(vort).max()
    )
