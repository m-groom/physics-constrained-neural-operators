import json

import h5py
import numpy as np
import pytest
import torch

from experiments.data_utils.datasets_dedalus import NSLoader2D
from experiments.data_utils.prepare_ns2d_stochastic import (
    build_dataset,
    derive_vorticity_and_streamfunction,
    main,
    read_run,
    recommend_stride,
    spectral_truncate,
    split_along_time,
)

L = 2 * np.pi


def grid(n):
    """Physical grid with axis -2 = x and axis -1 = y, matching utils/criterion.py."""
    c = np.linspace(0.0, L, n, endpoint=False)
    return np.meshgrid(c, c, indexing="ij")


def test_truncation_keeps_the_retained_modes_exactly_and_drops_the_rest():
    # At n_out = 16 the retained band is |kx|, |ky| < 8, so (3, 5) survives untouched and
    # both the out-of-band (9, 0) mode and the coarse-Nyquist (8, 0) mode go.
    x, y = grid(32)
    fine = np.cos(3 * x) * np.sin(5 * y) + 0.7 * np.cos(9 * x) + 0.3 * np.cos(8 * x)

    coarse = spectral_truncate(fine, 16)

    xc, yc = grid(16)
    assert coarse.shape == (16, 16)
    np.testing.assert_allclose(coarse, np.cos(3 * xc) * np.sin(5 * yc), atol=1e-13)


def test_vorticity_and_streamfunction_follow_the_repo_sign_convention():
    # psi = sin(x) sin(y) gives u = dpsi/dy, v = -dpsi/dx and omega = -Delta psi.
    x, y = grid(32)
    u, v = np.sin(x) * np.cos(y), -np.cos(x) * np.sin(y)

    omega, psi = derive_vorticity_and_streamfunction(u[None], v[None])

    np.testing.assert_allclose(omega[0], 2 * np.sin(x) * np.sin(y), atol=1e-12)
    np.testing.assert_allclose(psi[0], np.sin(x) * np.sin(y), atol=1e-12)


def test_time_split_is_ordered_disjoint_and_separated_by_the_gap():
    split = split_along_time(1000, gap=20)

    assert [split[s] for s in ("train", "val", "test")] == [(0, 672), (692, 836), (856, 1000)]
    sizes = [stop - start for start, stop in split.values()]
    assert sum(sizes) == 1000 - 2 * 20
    assert sizes == [672, 144, 144]  # 70 / 15 / 15 of the 960 usable snapshots


def test_time_split_refuses_a_run_too_short_to_carry_the_gaps():
    with pytest.raises(ValueError, match="too short"):
        split_along_time(40, gap=20)


def band_limited(rng, n, k_cut, shape=()):
    """A random real field whose spectral support stops well below the grid Nyquist."""
    kx = np.fft.fftfreq(n, d=1.0 / n)
    ky = np.fft.rfftfreq(n, d=1.0 / n)
    k2 = kx[:, None] ** 2 + ky[None, :] ** 2
    spec = rng.normal(size=(*shape, n, n // 2 + 1)) + 1j * rng.normal(size=(*shape, n, n // 2 + 1))
    spec[..., (k2 > k_cut**2) | (k2 == 0.0)] = 0.0
    return np.fft.irfft2(spec, s=(n, n))


def synthetic_fields(n_time, n, seed=0, k_cut=6):
    """A consistent (u, v, p, omega, psi, forcing) set with the repo's sign convention."""
    rng = np.random.default_rng(seed)
    psi = band_limited(rng, n, k_cut, (n_time,))
    u, v = _velocity_from(psi)
    omega, _ = derive_vorticity_and_streamfunction(u, v)
    p = band_limited(rng, n, k_cut, (n_time,))
    forcing = np.stack([band_limited(rng, n, k_cut, (n_time,)) for _ in range(2)], axis=1)
    return {"u": u, "v": v, "p": p, "omega": omega, "psi": psi, "forcing": forcing}


def _velocity_from(psi):
    """u = dpsi/dy, v = -dpsi/dx, spectrally."""
    n = psi.shape[-1]
    kx = np.fft.fftfreq(n, d=1.0 / n)[:, None]
    ky = np.fft.rfftfreq(n, d=1.0 / n)[None, :]
    psi_h = np.fft.rfft2(psi, axes=(-2, -1))
    u = np.fft.irfft2(1j * ky * psi_h, s=(n, n), axes=(-2, -1))
    v = np.fft.irfft2(-1j * kx * psi_h, s=(n, n), axes=(-2, -1))
    return u, v


CANONICAL = {
    "velocity": "velocity",
    "pressure": "pressure",
    "vorticity": "vorticity",
    "streamfunction": "streamfunction",
    "forcing": "forcing",
}
FIELD_KEY = {"pressure": "p", "vorticity": "omega", "streamfunction": "psi", "forcing": "forcing"}


def write_h5_set(path, fields, times, names=CANONICAL, attrs=None, set_number=1):
    """Write one Dedalus-shaped `*_s{N}.h5` set with configurable task names."""
    n = fields["u"].shape[-1]
    coords = np.linspace(0.0, L, n, endpoint=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["handler_name"] = "snapshots"
        f.attrs["set_number"] = set_number
        f.attrs["writes"] = len(times)
        for key, value in (attrs or {}).items():
            f.attrs[key] = value
        scales = f.create_group("scales")
        scales["sim_time"] = np.asarray(times, dtype=float)
        scales["write_number"] = np.arange(1, len(times) + 1)
        scales["x_hash_deadbeef"] = coords
        scales["y_hash_deadbeef"] = coords
        tasks = f.create_group("tasks")
        for canonical, name in names.items():
            if canonical == "velocity":
                if isinstance(name, str):
                    tasks[name] = np.stack([fields["u"], fields["v"]], axis=1)
                else:
                    tasks[name[0]], tasks[name[1]] = fields["u"], fields["v"]
            elif name is not None:
                tasks[name] = fields[FIELD_KEY[canonical]]


def write_run(tmp_path, fields, times, handler="snapshots", **kwargs):
    """Write a realisation directory holding one snapshot set."""
    run_dir = tmp_path / "realisation_0000"
    write_h5_set(run_dir / handler / f"{handler}_s1.h5", fields, times, **kwargs)
    return run_dir


def test_reader_finds_the_tasks_grid_and_times_and_truncates_to_the_coarse_grid(tmp_path):
    fields = synthetic_fields(4, 32)
    run_dir = write_run(tmp_path, fields, [0.0, 0.5, 1.0, 1.5])

    run = read_run(run_dir, n_coarse=16)

    assert run["u"].shape == (4, 16, 16)
    np.testing.assert_allclose(run["times"], [0.0, 0.5, 1.0, 1.5])
    np.testing.assert_allclose(run["x"], np.linspace(0.0, L, 16, endpoint=False))
    np.testing.assert_allclose(run["u"], spectral_truncate(fields["u"], 16), atol=1e-6)
    np.testing.assert_allclose(run["p"], spectral_truncate(fields["p"], 16), atol=1e-6)
    # omega and psi are derived from the truncated velocity, not read back from the file
    np.testing.assert_allclose(run["omega"], spectral_truncate(fields["omega"], 16), atol=1e-6)
    np.testing.assert_allclose(run["psi"], spectral_truncate(fields["psi"], 16), atol=1e-6)


def test_reader_tolerates_alias_task_names_a_vector_velocity_and_missing_attrs(tmp_path):
    fields = synthetic_fields(3, 32)
    names = {
        "velocity": ("u", "v"),
        "pressure": "p",
        "vorticity": None,
        "streamfunction": None,
        "forcing": None,
    }
    run_dir = write_run(tmp_path, fields, [0.0, 0.5, 1.0], names=names)

    run = read_run(run_dir, n_coarse=16)

    np.testing.assert_allclose(run["u"], spectral_truncate(fields["u"], 16), atol=1e-6)
    np.testing.assert_allclose(run["v"], spectral_truncate(fields["v"], 16), atol=1e-6)
    assert run["forcing"] is None


def test_reader_drops_the_frame_a_restart_repeats_and_orders_the_sets_by_time(tmp_path):
    fields = synthetic_fields(6, 32)
    run_dir = tmp_path / "realisation_0000"
    # a restart re-runs from t = 1.0, so that frame appears in both sets
    write_h5_set(run_dir / "coarse" / "coarse_s1.h5", _take(fields, [0, 1, 2]), [0.0, 0.5, 1.0])
    write_h5_set(
        run_dir / "coarse" / "coarse_s2.h5", _take(fields, [2, 3, 4]), [1.0, 1.5, 2.0], set_number=2
    )

    run = read_run(run_dir, n_coarse=16)

    np.testing.assert_allclose(run["times"], [0.0, 0.5, 1.0, 1.5, 2.0])
    assert run["handler"] == "coarse"
    np.testing.assert_allclose(
        run["u"], spectral_truncate(fields["u"][[0, 1, 2, 3, 4]], 16), atol=1e-6
    )


def _take(fields, index):
    return {k: v[index] for k, v in fields.items()}


def test_reader_rejects_data_written_with_the_opposite_vorticity_sign(tmp_path):
    fields = synthetic_fields(2, 32)
    fields["omega"] = -fields["omega"]
    run_dir = write_run(tmp_path, fields, [0.0, 0.5])

    with pytest.raises(ValueError, match="disagrees with the assumed convention"):
        read_run(run_dir, n_coarse=16)


def long_run(tmp_path, n_time=40, n=32, n_real=1, dt=0.25, seed=0):
    """A run directory with `n_real` realisations of `n_time` consecutive snapshots."""
    times = np.arange(n_time) * dt
    for r in range(n_real):
        fields = synthetic_fields(n_time, n, seed=seed + r)
        write_h5_set(tmp_path / f"realisation_{r:04d}" / "coarse" / "coarse_s1.h5", fields, times)
    return tmp_path, times


def test_pairs_are_consecutive_snapshots_of_the_split_segment(tmp_path):
    run_dir, times = long_run(tmp_path)
    fields = synthetic_fields(40, 32)
    coarse = {k: spectral_truncate(fields[k], 16) for k in ("u", "v", "p")}

    summary = build_dataset(run_dir, tmp_path / "out", n_coarse=16, gap=2, skip=4)

    data = np.load(summary["datasets"]["velocity"])
    assert data["X_train"].shape == (21, 3, 16, 16)
    assert data["X_train"].dtype == np.float32
    stack = np.stack([coarse["u"], coarse["v"], coarse["p"]], axis=1).astype(np.float32)
    np.testing.assert_allclose(data["X_train"], stack[4:25], atol=1e-6)
    np.testing.assert_allclose(data["y_train"], stack[5:26], atol=1e-6)
    np.testing.assert_allclose(data["times_train"][:, 0], times[4:25])
    np.testing.assert_allclose(data["times_train"][:, 1] - data["times_train"][:, 0], 0.25)


def test_splits_are_ordered_and_separated_by_the_gap_after_the_spin_up(tmp_path):
    run_dir, times = long_run(tmp_path)

    summary = build_dataset(run_dir, tmp_path / "out", n_coarse=16, gap=2, skip=4)

    data = np.load(summary["datasets"]["vorticity"])
    assert [data[f"X_{s}"].shape[0] for s in ("train", "val", "test")] == [21, 4, 4]
    assert data["X_train"].shape[1] == 1
    ends = {
        s: (data[f"times_{s}"][0, 0], data[f"times_{s}"][-1, 1]) for s in ("train", "val", "test")
    }
    assert ends["train"][0] == times[4]
    # two discarded snapshots between the segments: the next segment starts 3 dt later
    assert ends["val"][0] - ends["train"][1] == pytest.approx(3 * 0.25)
    assert ends["test"][0] - ends["val"][1] == pytest.approx(3 * 0.25)
    assert float(data["t_start"]) == times[4]
    assert int(data["gap"]) == 2
    assert int(data["cutoff"]) == 16


def test_further_realisations_become_an_extra_test_split(tmp_path):
    run_dir, _ = long_run(tmp_path, n_real=3)

    summary = build_dataset(run_dir, tmp_path / "out", n_coarse=16, gap=2, skip=4)

    data = np.load(summary["datasets"]["velocity"])
    # both extra realisations contribute all 36 post-spin-up snapshots, so 35 pairs each
    assert data["X_test_extra"].shape == (70, 3, 16, 16)
    assert data["psi_X_test_extra"].shape == (70, 1, 16, 16)
    np.testing.assert_array_equal(data["test_extra_realisations"], [1, 2])
    np.testing.assert_array_equal(
        data["realisation_ids_test_extra"], np.repeat([1, 2], 35).astype(np.int32)
    )
    # realisation-major ordering, so a rollout of 35 steps is one realisation
    assert data["X_test_extra"].shape[0] % 35 == 0


def test_a_forcing_band_above_the_cutoff_must_truncate_to_nothing(tmp_path):
    run_dir, _ = long_run(tmp_path)
    x, _ = grid(32)
    _add_forcing(run_dir, np.broadcast_to(np.cos(10 * x), (40, 32, 32)))

    summary = build_dataset(
        run_dir,
        tmp_path / "out",
        n_coarse=8,
        gap=2,
        skip=4,
        physics={"kmin": 10, "kmax": 12},
    )

    check = json.loads(summary["diagnostics"]["json"].read_text())["forcing"]
    assert check["resolved_at_n_coarse"] is False
    assert check["asserted_empty"] is True
    assert check["retained_fraction"] < 1e-12


def test_a_forcing_that_survives_the_cutoff_is_an_error_when_the_band_says_it_should_not(
    tmp_path,
):
    run_dir, _ = long_run(tmp_path)
    x, _ = grid(32)
    _add_forcing(run_dir, np.broadcast_to(np.cos(2 * x), (40, 32, 32)))

    with pytest.raises(ValueError, match="lies above the cut-off"):
        build_dataset(
            run_dir, tmp_path / "out", n_coarse=8, gap=2, skip=4, physics={"kmin": 10, "kmax": 12}
        )

    # A leak far below the velocity scale is still a leak: measured against the coarse
    # velocity this forcing is 1e-10, but the truncation retains all of its own RMS.
    _add_forcing(run_dir, 1e-10 * np.broadcast_to(np.cos(2 * x), (40, 32, 32)))
    with pytest.raises(ValueError, match="retains"):
        build_dataset(
            run_dir, tmp_path / "out2", n_coarse=8, gap=2, skip=4, physics={"kmin": 10, "kmax": 12}
        )


def test_an_already_coarse_source_measures_the_forcing_against_the_velocity(tmp_path):
    # The in-situ coarse handler is the truncated field, so truncating it again is a no-op:
    # it retains all of its own RMS whatever that RMS is, and the fraction retained can say
    # nothing. What says the forcing vanished on the coarse grid is its size beside the
    # coarse velocity, which is what the check asserts on.
    n_time, n = 12, 16
    run_dir, _ = long_run(tmp_path, n_time=n_time, n=n)
    # round-off dust, which is what the production run's coarse forcing is
    _add_forcing(run_dir, 1e-17 * np.random.default_rng(1).normal(size=(n_time, n, n)))

    summary = build_dataset(
        run_dir, tmp_path / "out", n_coarse=n, gap=1, skip=2, physics={"kmin": 20, "kmax": 24}
    )

    check = json.loads(summary["diagnostics"]["json"].read_text())["forcing"]
    # the no-op truncation keeps the dust, so the fraction retained stays of order one
    assert check["retained_fraction"] > 0.5
    assert check["rms_relative_to_velocity"] < 1e-12
    assert check["asserted_empty"] is True

    # A forcing that really is on the coarse grid stays an error, in-situ source or not.
    x, _ = grid(n)
    _add_forcing(run_dir, np.broadcast_to(np.cos(2 * x), (n_time, n, n)))
    with pytest.raises(ValueError, match="lies above the cut-off"):
        build_dataset(
            run_dir, tmp_path / "out2", n_coarse=n, gap=1, skip=2, physics={"kmin": 20, "kmax": 24}
        )


def _add_forcing(run_dir, component):
    """Overwrite the forcing task of every realisation with `(component, 0)`."""
    for path in sorted(run_dir.glob("realisation_*/*/*_s*.h5")):
        with h5py.File(path, "r+") as f:
            del f["tasks"]["forcing"]
            f["tasks"]["forcing"] = np.stack([component, np.zeros_like(component)], axis=1)


def test_normaliser_comes_from_the_training_split_and_the_npz_round_trips(tmp_path):
    run_dir, _ = long_run(tmp_path)

    summary = build_dataset(run_dir, tmp_path / "out", n_coarse=16, gap=2, skip=4)

    for field, velocity_channels in (("velocity", (0, 1)), ("vorticity", None)):
        stats = torch.load(summary["normalisers"][field], weights_only=False)
        loader = NSLoader2D(
            str(summary["loader_dirs"][field]),
            state="train",
            velocity_channels=velocity_channels,
        )
        torch.testing.assert_close(stats["mean"], loader.mean, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(stats["std"], loader.std, rtol=1e-5, atol=1e-6)
        # the test split has its own spread, so the saved statistics are the training ones
        test_std = np.load(summary["datasets"][field])["X_test"].std(axis=(0, 2, 3))
        assert not np.allclose(test_std, stats["std"].numpy().ravel(), rtol=1e-3)


def test_the_test_split_loads_and_rolls_out_through_the_loader(tmp_path):
    run_dir, _ = long_run(tmp_path)

    summary = build_dataset(run_dir, tmp_path / "out", n_coarse=16, gap=2, skip=4)

    loader = NSLoader2D(
        str(summary["loader_dirs"]["vorticity"]),
        state="test",
        normalizer_path=str(summary["normalisers"]["vorticity"]),
        velocity_channels=None,
    )
    X, y = loader.transform_rollout(T=4)

    assert X.shape == (1, 16, 16, 4, 1)
    data = np.load(summary["datasets"]["vorticity"])
    restored = X.numpy()[0, ..., 0].transpose(2, 0, 1) * loader.std.item() + loader.mean.item()
    np.testing.assert_allclose(restored, data["X_test"][:, 0], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(y.numpy()[0, ..., 0].transpose(2, 0, 1).shape, (4, 16, 16))


def test_reading_the_in_situ_coarse_output_matches_truncating_the_full_resolution_output(
    tmp_path,
):
    # Dedalus's `add_task(..., scales=1/16)` is an exact spectral truncation, so a coarse
    # handler holding the truncated field must give the same dataset as truncating offline.
    fields = synthetic_fields(4, 32)
    coarse_fields = {k: spectral_truncate(v, 16) for k, v in fields.items() if k != "forcing"}
    coarse_fields["forcing"] = spectral_truncate(fields["forcing"], 16)
    run_dir = tmp_path / "realisation_0000"
    write_h5_set(run_dir / "snapshots" / "snapshots_s1.h5", fields, [0.0, 0.5, 1.0, 1.5])
    write_h5_set(run_dir / "coarse" / "coarse_s1.h5", coarse_fields, [0.0, 0.5, 1.0, 1.5])

    insitu = read_run(run_dir, n_coarse=16)
    offline = read_run(run_dir, n_coarse=16, handler="snapshots")

    assert insitu["handler"] == "coarse" and offline["handler"] == "snapshots"
    for name in ("u", "v", "p", "omega", "psi"):
        np.testing.assert_allclose(insitu[name], offline[name], rtol=1e-5, atol=1e-12)


def test_time_split_keeps_a_pair_in_every_segment_of_a_short_run():
    # 11 snapshots less two gaps of one leave 9: 15 % of that rounds to one snapshot, which
    # would carry no pair at all, so the small splits are floored at two snapshots.
    split = split_along_time(11, gap=1)

    assert [split[s] for s in ("train", "val", "test")] == [(0, 5), (6, 8), (9, 11)]


def test_the_snapshot_cadence_jitter_is_recorded_and_a_gross_one_is_refused(tmp_path):
    # Dedalus writes at the first step past each target time, so the interval carries a
    # jitter of one timestep: it is recorded, and only a gross one is an error.
    run_dir, times = long_run(tmp_path)
    jittered = times.copy()
    jittered[10] += 0.02  # one snapshot written 8 % of an interval late
    _rewrite_times(run_dir, jittered)

    summary = build_dataset(run_dir, tmp_path / "out", n_coarse=16, gap=2, skip=4)

    data = np.load(summary["datasets"]["vorticity"])
    assert float(data["dt_max_deviation"]) == pytest.approx(0.08, rel=1e-6)
    assert float(data["dt"]) == pytest.approx(0.25, rel=1e-3)

    gross = times.copy()
    gross[10] += 0.05
    _rewrite_times(run_dir, gross)
    with pytest.raises(ValueError, match="cadence is not uniform"):
        build_dataset(run_dir, tmp_path / "out2", n_coarse=16, gap=2, skip=4)


def _rewrite_times(run_dir, times):
    for path in sorted(run_dir.glob("realisation_*/*/*_s*.h5")):
        with h5py.File(path, "r+") as f:
            f["scales"]["sim_time"][...] = times


def test_the_forcing_diagnostic_measures_the_fraction_of_the_band_the_cutoff_keeps(tmp_path):
    fields = synthetic_fields(3, 32)
    x, _ = grid(32)
    run_dir = write_run(tmp_path, fields, [0.0, 0.5, 1.0])
    _add_forcing(tmp_path, np.broadcast_to(np.cos(2 * x), (3, 32, 32)))

    # the whole band is below the cut-off, so all of the forcing RMS survives
    assert read_run(run_dir, n_coarse=16)["forcing"]["retained_fraction"] == pytest.approx(1.0)
    # at a cut-off of |k| < 2 the k = 2 forcing is gone altogether
    assert read_run(run_dir, n_coarse=4)["forcing"]["retained_fraction"] == pytest.approx(0.0)


def test_the_two_thirds_cutoff_drops_the_modes_the_dealiasing_filter_would():
    # NS2D's rectangular 2/3 filter keeps |kx|, |ky| <= n // 3, so on a 24-point grid the
    # mode 8 survives and 9 does not, while the sharp cut-off at |k| < 12 keeps both.
    x, y = grid(48)
    field = np.cos(8 * x) + 0.5 * np.cos(9 * y)
    xc, yc = grid(24)

    np.testing.assert_allclose(
        spectral_truncate(field, 24, "two-thirds"), np.cos(8 * xc), atol=1e-13
    )
    np.testing.assert_allclose(
        spectral_truncate(field, 24, "sharp"), np.cos(8 * xc) + 0.5 * np.cos(9 * yc), atol=1e-13
    )


def test_the_cutoff_rule_reaches_the_dataset_and_is_recorded(tmp_path):
    run_dir, _ = long_run(tmp_path)

    summary = build_dataset(
        run_dir, tmp_path / "out", n_coarse=16, gap=2, skip=4, cutoff="two-thirds"
    )

    data = np.load(summary["datasets"]["velocity"])
    assert bytes(data["cutoff_rule"]) == b"two-thirds"
    assert int(data["cutoff_kmax"]) == 5  # 16 // 3
    fields = synthetic_fields(40, 32)
    expected = spectral_truncate(fields["u"], 16, "two-thirds").astype(np.float32)
    np.testing.assert_allclose(data["X_train"][:, 0], expected[4:25], atol=1e-6)


def test_the_forcing_band_check_uses_the_band_the_cutoff_rule_actually_keeps(tmp_path):
    # At n_coarse = 24 the sharp rule keeps |k| <= 11 and the two-thirds rule only |k| <= 8,
    # so a forcing at k = 10 is resolved under one and truncated away under the other.
    run_dir, _ = long_run(tmp_path, n_time=20, n=48)
    x, _ = grid(48)
    _add_forcing(run_dir, np.broadcast_to(np.cos(10 * x), (20, 48, 48)))
    physics = {"kmin": 10, "kmax": 10}

    sharp = build_dataset(
        run_dir, tmp_path / "sharp", n_coarse=24, gap=1, physics=physics, cutoff="sharp"
    )
    two_thirds = build_dataset(
        run_dir, tmp_path / "thirds", n_coarse=24, gap=1, physics=physics, cutoff="two-thirds"
    )

    assert _forcing(sharp)["resolved_at_n_coarse"] is True
    assert _forcing(sharp)["retained_fraction"] == pytest.approx(1.0)
    assert _forcing(two_thirds)["resolved_at_n_coarse"] is False
    assert _forcing(two_thirds)["asserted_empty"] is True
    assert _forcing(two_thirds)["retained_fraction"] < 1e-12


def _forcing(summary):
    return json.loads(summary["diagnostics"]["json"].read_text())["forcing"]


def test_the_derived_fields_use_the_box_the_data_lives_on(tmp_path):
    # NS2D takes --Lx, so the box is not always 2 pi. On a 4 pi box a builder that assumed
    # 2 pi would double every wavenumber and return four times the vorticity.
    n = 32
    c = np.linspace(0.0, 4 * np.pi, n, endpoint=False)
    x, y = np.meshgrid(c, c, indexing="ij")
    u, v = np.sin(x) * np.cos(y), -np.cos(x) * np.sin(y)

    omega, psi = derive_vorticity_and_streamfunction(u[None], v[None], 4 * np.pi)

    np.testing.assert_allclose(omega[0], 2 * np.sin(x) * np.sin(y), atol=1e-12)
    np.testing.assert_allclose(psi[0], np.sin(x) * np.sin(y), atol=1e-12)


NS2D_ATTRS = {
    "vorticity_convention": "omega = dx(v) - dy(u)",
    "streamfunction_convention": "lap(psi) = -omega, u = (dy(psi), -dx(psi))",
}


def test_the_reader_reads_the_sign_convention_the_solver_recorded(tmp_path):
    fields = synthetic_fields(2, 32)
    run_dir = write_run(tmp_path, fields, [0.0, 0.5], attrs=NS2D_ATTRS)

    conventions = read_run(run_dir, n_coarse=16)["conventions"]

    assert conventions["vorticity_convention"] == NS2D_ATTRS["vorticity_convention"]
    assert conventions["streamfunction_convention"] == NS2D_ATTRS["streamfunction_convention"]


def test_the_reader_refuses_a_run_whose_recorded_convention_is_the_opposite_one(tmp_path):
    fields = synthetic_fields(2, 32)
    run_dir = write_run(
        tmp_path, fields, [0.0, 0.5], attrs={"vorticity_convention": "omega = dy(u) - dx(v)"}
    )

    with pytest.raises(ValueError, match="records the opposite"):
        read_run(run_dir, n_coarse=16)


def test_the_reader_says_so_when_there_is_nothing_to_check_the_convention_against(tmp_path):
    fields = synthetic_fields(2, 32)
    names = dict(CANONICAL, vorticity=None, streamfunction=None)
    run_dir = write_run(tmp_path, fields, [0.0, 0.5], names=names)

    conventions = read_run(run_dir, n_coarse=16)["conventions"]

    assert "omega" not in conventions and "psi" not in conventions
    assert "not possible" in conventions["numerical_check"]


def test_the_forcing_check_says_so_when_the_run_wrote_no_forcing_to_measure(tmp_path):
    fields = synthetic_fields(20, 32)
    times = np.arange(20) * 0.25
    names = dict(CANONICAL, forcing=None)
    write_h5_set(tmp_path / "realisation_0000" / "coarse" / "coarse_s1.h5", fields, times, names)

    summary = build_dataset(
        tmp_path, tmp_path / "out", n_coarse=8, gap=1, physics={"kmin": 10, "kmax": 12}
    )

    check = json.loads(summary["diagnostics"]["json"].read_text())["forcing"]
    assert check["resolved_at_n_coarse"] is False
    assert check["asserted_empty"] is False
    assert "not measurable" in check["assertion"]


def test_an_extra_realisation_on_a_different_cadence_is_refused(tmp_path):
    run_dir, times = long_run(tmp_path, n_real=2)
    with h5py.File(run_dir / "realisation_0001" / "coarse" / "coarse_s1.h5", "r+") as f:
        f["scales"]["sim_time"][...] = times * 2.0

    with pytest.raises(ValueError, match="cadence"):
        build_dataset(run_dir, tmp_path / "out", n_coarse=16, gap=2, skip=4)


def test_a_stride_subsamples_the_run_in_time_and_is_recorded(tmp_path):
    run_dir, times = long_run(tmp_path)
    fields = synthetic_fields(40, 32)
    coarse = spectral_truncate(fields["u"], 16).astype(np.float32)

    summary = build_dataset(run_dir, tmp_path / "out", n_coarse=16, gap=1, skip=1, stride=3)

    data = np.load(summary["datasets"]["velocity"])
    # every third snapshot from the start, so a pair spans three snapshot intervals
    assert float(data["dt"]) == pytest.approx(0.75)
    assert int(data["stride"]) == 3
    np.testing.assert_allclose(data["times_train"][:, 1] - data["times_train"][:, 0], 0.75)
    kept = coarse[::3]
    np.testing.assert_allclose(data["X_train"][:, 0], kept[1 : 1 + len(data["X_train"])], atol=1e-6)
    np.testing.assert_allclose(data["times_train"][0, 0], times[3])


def write_spectra(path, k, energy, times=(0.0, 1.0)):
    """A `spectra.h5` in the solver's layout: one `k_E_Z_t{t}` dataset per output time."""
    with h5py.File(path, "w") as f:
        for t in times:
            f[f"k_E_Z_t{t:.6f}"] = np.stack([k, energy, energy * k**2], axis=1)


def test_the_stride_recommendation_comes_from_the_eddy_turnover_time(tmp_path):
    # tau(k) = (k^3 E(k))^(-1/2) is one everywhere for E(k) = k^-3, so at dt = 0.08 the
    # snapshot interval reaches one turnover time after twelve snapshots.
    k = np.arange(64, dtype=float)
    energy = np.divide(1.0, np.maximum(k, 1.0) ** 3)
    run_dir = tmp_path / "realisation_0000"
    run_dir.mkdir(parents=True)
    write_spectra(run_dir / "spectra.h5", k, energy)

    advice = recommend_stride(run_dir, dt=0.08)

    assert advice["tau_min"] == pytest.approx(1.0)
    assert advice["k_range"] == [20, 32]
    assert advice["stride"] == 12
    assert advice["snapshots"] == 2


def test_no_spectra_file_means_no_recommendation(tmp_path):
    assert recommend_stride(tmp_path, dt=0.08) is None


def seeded_runs(tmp_path, seeds, n_time=40, n=32, dt=0.25):
    """Runs in the E3 production layout: one seed per run directory, one realisation in each.

    Each realisation directory carries its own `run_config.jsonl`, which is where the seed
    that identifies the realisation is recorded.
    """
    times = np.arange(n_time) * dt
    runs = []
    for index, seed in enumerate(seeds):
        run_dir = tmp_path / f"seed{seed}" / "Nx32_Ny32_nu5e-03"
        realisation = run_dir / "realisation_0000"
        fields = synthetic_fields(n_time, n, seed=index)
        write_h5_set(realisation / "coarse" / "coarse_s1.h5", fields, times)
        (realisation / "run_config.jsonl").write_text(
            json.dumps({"args": {"seed": seed, "nu": 5e-3, "alpha": 3e-3}}) + "\n"
        )
        runs.append(run_dir)
    return runs, times


E3_SPLIT = {"train": [1234, 1235, 1236], "val": [1237], "test": [1238]}


def build_by_realisation(runs, out_dir, **kwargs):
    """Build the five-seed dataset with the split agreed for E3."""
    return build_dataset(
        runs,
        out_dir,
        n_coarse=16,
        gap=2,
        skip=4,
        split="realisation",
        realisations=E3_SPLIT,
        **kwargs,
    )


def test_the_realisation_split_gives_each_split_its_own_realisations(tmp_path):
    runs, _ = seeded_runs(tmp_path, [1234, 1235, 1236, 1237, 1238])

    summary = build_by_realisation(runs, tmp_path / "out")

    data = np.load(summary["datasets"]["velocity"])
    for state, ids in E3_SPLIT.items():
        np.testing.assert_array_equal(data[f"{state}_realisations"], ids)
        assert set(data[f"realisation_ids_{state}"].tolist()) == set(ids)
    # every realisation belongs to exactly one split
    assert not set(data["realisation_ids_train"]) & set(data["realisation_ids_val"])
    assert not set(data["realisation_ids_train"]) & set(data["realisation_ids_test"])
    assert not set(data["realisation_ids_val"]) & set(data["realisation_ids_test"])


def test_the_time_test_tail_comes_out_of_the_training_realisations_behind_the_gap(tmp_path):
    runs, times = seeded_runs(tmp_path, [1234, 1235, 1236, 1237, 1238])

    summary = build_by_realisation(runs, tmp_path / "out")

    data = np.load(summary["datasets"]["velocity"])
    # 36 post-spin-up snapshots, of which the last 15 % is five: training stops two snapshots
    # earlier still, so 29 training snapshots and 28 pairs from each of the three realisations
    assert data["X_train"].shape == (84, 3, 16, 16)
    assert data["X_test_time"].shape == (12, 3, 16, 16)
    np.testing.assert_array_equal(data["test_time_realisations"], [1234, 1235, 1236])
    # the tail is behind the gap, so no training snapshot is within two of it
    assert data["times_train"][:, 1].max() == pytest.approx(times[32])
    assert data["times_test_time"][:, 0].min() == pytest.approx(times[35])
    assert data["times_test_time"][:, 1].max() == pytest.approx(times[39])
    # and the val and test realisations keep every post-spin-up snapshot
    assert data["X_val"].shape == (35, 3, 16, 16)
    assert data["X_test"].shape == (35, 3, 16, 16)


def test_no_pair_and_no_rollout_of_the_realisation_split_crosses_a_realisation(tmp_path):
    runs, _ = seeded_runs(tmp_path, [1234, 1235, 1236, 1237, 1238])

    summary = build_by_realisation(runs, tmp_path / "out")

    data = np.load(summary["datasets"]["vorticity"])
    for state in ("train", "val", "test", "test_time"):
        # a pair that straddled two realisations would step backwards in time
        steps = data[f"times_{state}"][:, 1] - data[f"times_{state}"][:, 0]
        np.testing.assert_allclose(steps, 0.25)
    ids = data["realisation_ids_train"]
    # realisation-major and equal in length, so a rollout of 28 steps is one realisation
    assert len(ids) % 28 == 0
    assert [set(block.tolist()) for block in ids.reshape(-1, 28)] == [{1234}, {1235}, {1236}]
    loader = NSLoader2D(
        str(summary["loader_dirs"]["vorticity"]),
        state="train",
        velocity_channels=None,
    )
    X, _ = loader.transform_rollout(T=28)
    assert X.shape == (3, 16, 16, 28, 1)
    # the first trajectory is realisation 1234, and its frames are that realisation's own
    fields = synthetic_fields(40, 32, seed=0)
    expected = spectral_truncate(fields["omega"], 16)[4:32]
    restored = X.numpy()[0, ..., 0].transpose(2, 0, 1) * loader.std.item() + loader.mean.item()
    np.testing.assert_allclose(restored, expected, rtol=1e-3, atol=1e-3)


def test_realisations_of_unequal_length_are_trimmed_to_a_common_window(tmp_path):
    runs, _ = seeded_runs(tmp_path, [1234, 1235, 1236, 1237])
    short, _ = seeded_runs(tmp_path / "short", [1238], n_time=36)

    summary = build_by_realisation([*runs, *short], tmp_path / "out")

    data = np.load(summary["datasets"]["vorticity"])
    # the shortest realisation sets the window: 32 usable snapshots, a tail of five and a
    # gap of two leave 25 training snapshots, so 24 pairs from each training realisation
    assert data["X_train"].shape[0] == 3 * 24
    assert data["X_val"].shape[0] == 31 and data["X_test"].shape[0] == 31
    np.testing.assert_array_equal(data["split_bounds"][0], [4, 29])


def test_the_realisation_split_round_trips_through_the_loader(tmp_path):
    runs, _ = seeded_runs(tmp_path, [1234, 1235, 1236, 1237, 1238])

    summary = build_by_realisation(runs, tmp_path / "out")

    data = np.load(summary["datasets"]["velocity"])
    for state, expected in (("train", 84), ("test", 35), ("test_time", 12)):
        loader = NSLoader2D(
            str(summary["loader_dirs"]["velocity"]),
            state=state,
            normalizer_path=str(summary["normalisers"]["velocity"]),
        )
        assert len(loader) == expected
        assert loader[0][0].shape == (16, 16, 3)
    # the normaliser is the training realisations', which the held-out ones do not share
    stats = torch.load(summary["normalisers"]["velocity"], weights_only=False)
    test_std = data["X_test"].std(axis=(0, 2, 3))
    assert not np.allclose(test_std, stats["std"].numpy().ravel(), rtol=1e-3)
    # the rollout length of a split is its own `split_bounds`, which is what keeps a
    # trajectory inside one realisation
    states = [bytes(state).decode() for state in data["split_states"]]
    bounds = dict(zip(states, data["split_bounds"], strict=True))
    for state, realisations in (("test", 1), ("test_time", 3)):
        start, stop = bounds[state]
        loader = NSLoader2D(
            str(summary["loader_dirs"]["velocity"]),
            state=state,
            normalizer_path=str(summary["normalisers"]["velocity"]),
        )
        X, y = loader.transform_rollout(T=stop - start - 1)
        assert X.shape == (realisations, 16, 16, stop - start - 1, 3)
        assert y.shape == X.shape


def test_the_realisation_split_records_how_it_was_built(tmp_path):
    runs, _ = seeded_runs(tmp_path, [1234, 1235, 1236, 1237, 1238])

    summary = build_by_realisation(runs, tmp_path / "out", stride=1, decorrelation_frames=7)

    data = np.load(summary["datasets"]["velocity"])
    assert bytes(data["split_mode"]) == b"realisation"
    assert [bytes(s) for s in data["split_states"]] == [b"train", b"val", b"test", b"test_time"]
    assert int(data["gap"]) == 2 and int(data["skip"]) == 4 and int(data["stride"]) == 1
    assert float(data["time_test_fraction"]) == pytest.approx(0.15)
    assert float(data["decorrelation_frames"]) == 7.0
    # the provenance covers every realisation the datasets draw on, not only the first
    assert len(data["source_realisations"]) == 5
    assert len(data["source_files"]) == 5
    record = json.loads(summary["diagnostics"]["json"].read_text())
    assert record["split_mode"] == "realisation"
    assert record["splits"]["train"]["realisations"] == [1234, 1235, 1236]
    assert record["splits"]["train"]["pairs"] == 84
    assert record["splits"]["test_time"]["realisations"] == [1234, 1235, 1236]


def test_the_default_split_still_records_the_time_mode_and_one_realisation(tmp_path):
    run_dir, _ = long_run(tmp_path)

    summary = build_dataset(run_dir, tmp_path / "out", n_coarse=16, gap=2, skip=4)

    data = np.load(summary["datasets"]["velocity"])
    assert bytes(data["split_mode"]) == b"time"
    assert [bytes(s) for s in data["split_states"]] == [b"train", b"val", b"test"]
    np.testing.assert_array_equal(data["train_realisations"], [0])
    assert np.isnan(float(data["time_test_fraction"]))
    assert (
        "realisations"
        not in json.loads(summary["diagnostics"]["json"].read_text())["splits"]["train"]
    )


def test_a_realisation_split_refuses_an_unknown_id_and_one_id_in_two_splits(tmp_path):
    runs, _ = seeded_runs(tmp_path, [1234, 1235])

    with pytest.raises(ValueError, match="no realisation 9999"):
        build_dataset(
            runs,
            tmp_path / "out",
            n_coarse=16,
            gap=2,
            skip=4,
            split="realisation",
            realisations={"train": [1234], "val": [1235], "test": [9999]},
        )
    with pytest.raises(ValueError, match="in both the train and the test split"):
        build_dataset(
            runs,
            tmp_path / "out",
            n_coarse=16,
            gap=2,
            skip=4,
            split="realisation",
            realisations={"train": [1234], "val": [1235], "test": [1234]},
        )


def test_the_command_line_builds_the_realisation_split(tmp_path):
    runs, _ = seeded_runs(tmp_path, [1234, 1235, 1236, 1237, 1238])
    argv = [
        "--out-dir",
        str(tmp_path / "out"),
        "--n-coarse",
        "16",
        "--field",
        "vorticity",
        "--gap",
        "2",
        "--skip",
        "4",
        "--split",
        "realisation",
        "--train-realisations",
        "1234,1235,1236",
        "--val-realisations",
        "1237",
        "--test-realisations",
        "1238",
        "--decorrelation-frames",
        "7",
    ]
    for run in runs:
        argv += ["--run-dir", str(run)]

    summary = main(argv)

    data = np.load(summary["datasets"]["vorticity"])
    np.testing.assert_array_equal(data["train_realisations"], [1234, 1235, 1236])
    np.testing.assert_array_equal(data["test_realisations"], [1238])
    assert data["X_test_time"].shape[0] == 12
    assert float(data["decorrelation_frames"]) == 7.0
