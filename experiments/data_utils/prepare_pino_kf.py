"""Build the E0 (vorticity) and E1 (velocity) datasets from the PINO-paper Kolmogorov data.

The source is `NS_fft_Re500_T4000.npy`, shape `(4000, 65, 64, 64)` float64: 4000 trajectories
of 65 vorticity snapshots on `(0, 2pi)^2` at 64^2, Re = 500 (`nu = 1/500`), forced by
`-4 cos(4 y)` in the vorticity equation. The snapshot interval is `dt = 1/64 = 0.015625`
(fitted 0.01554 +/- 3e-5 from the truth momentum residual, issue #20); the `T = 0.5`
stated in the PINO paper does not match the data.

Axis conventions follow `utils/criterion.py` (`FDM_NS_vorticity`): the first spatial axis is
`x` and the second is `y`, so the `cos(4 .)` signature of the forcing lies along the second
spatial axis. Sign conventions, also from `FDM_NS_vorticity`:

    omega = dv/dx - du/dy,   Delta psi = -omega,   u = dpsi/dy,   v = -dpsi/dx,

i.e. `psi_hat = omega_hat / |k|^2`. The pressure follows the incompressible Poisson equation

    Delta p = -div((u . grad) u) = -((du/dx)^2 + 2 (du/dy)(dv/dx) + (dv/dy)^2),

with zero spatial mean. The Kolmogorov forcing is `F = (sin(4 y), 0)` in velocity form (its
curl is `-4 cos(4 y)`), which is divergence-free, as is the viscous term for a solenoidal
velocity, so neither contributes to the pressure Poisson equation.

Both datasets are built from a vorticity field projected onto the range of the discrete curl
(`project_onto_curl_range`), which drops the mean mode and the grid-Nyquist row and column.
No periodic velocity field can carry either: the mean vorticity of a periodic flow is zero,
and the first derivative of a Nyquist mode is not a real grid function. On the source file
that removes 1.0e-2 of the vorticity in relative L2, all of it at the 64^2 cutoff where the
data is unresolved anyway, and in exchange E0 and E1 describe exactly the same flow.

Each npz holds `X_<state>`, `y_<state>` of shape `(N, C, nx, ny)` float32 and `times_<state>`
of shape `(N, 2)`, for `state` in `train`, `val`, `test`; the velocity dataset adds
`psi_X_<state>` and `psi_y_<state>` with a single streamfunction channel. Samples run
trajectory-major, so `NSLoader2D.transform_rollout` rebuilds the rollouts. The remaining keys
are metadata: `field`, `channels`, `nu`, `dt`, `t_start`, `t_end`, `domain_length`, `forcing`,
`axis_order`, `sign_convention`, `vorticity_projection`, `source`, `seed`, `x_coords`,
`y_coords`,
`<state>_realisations` and `realisation_ids_<state>`.
"""

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap

DOMAIN_LENGTH = 2.0 * np.pi


def wavenumbers(n, length=DOMAIN_LENGTH):
    """Return the `rfft2` wavenumber grids for an `n x n` periodic box of side `length`.

    Three grids come back: `kx` and `ky` for first derivatives, and `inv_k2` for the Poisson
    solve. The Nyquist entry of a first-derivative wavenumber is zeroed because the derivative
    of a Nyquist mode is not representable as a real grid function; `inv_k2` keeps it, since
    `|k|^2` is an even operator. The mean mode of `inv_k2` is zeroed, which fixes the additive
    constant of `psi` and `p` by requiring zero spatial mean.

    Args:
        n: Number of grid points per direction.
        length: Side length of the periodic box.

    Returns:
        Tuple `(kx, ky, inv_k2)` of arrays broadcastable against an `(..., n, n // 2 + 1)`
        half-spectrum, with axis -2 the `x` axis and axis -1 the `y` axis.
    """
    scale = 2.0 * np.pi / length
    kx = np.fft.fftfreq(n, d=1.0 / n) * scale
    ky = np.fft.rfftfreq(n, d=1.0 / n) * scale
    k2 = kx[:, None] ** 2 + ky[None, :] ** 2
    inv_k2 = np.zeros_like(k2)
    np.divide(1.0, k2, out=inv_k2, where=k2 > 0.0)
    kx = kx.copy()
    ky = ky.copy()
    kx[n // 2] = 0.0
    ky[n // 2] = 0.0
    return kx[:, None], ky[None, :], inv_k2


def project_onto_curl_range(w):
    """Drop the vorticity modes no periodic grid velocity field can produce.

    Those are the mean mode, since the mean vorticity of a periodic flow is zero, and the
    grid-Nyquist row and column, since `d/dx` and `d/dy` annihilate the Nyquist mode of the
    direction they act on. What remains round-trips exactly through `invert_vorticity`.

    Args:
        w: Vorticity, shape `(..., n, n)`.

    Returns:
        The projected field, with the same shape as `w`.
    """
    n = w.shape[-1]
    w_h = np.fft.rfft2(w, axes=(-2, -1))
    w_h[..., 0, 0] = 0.0
    w_h[..., n // 2, :] = 0.0
    w_h[..., :, n // 2] = 0.0
    return np.fft.irfft2(w_h, s=(n, n), axes=(-2, -1))


def invert_vorticity(w):
    """Recover the streamfunction, velocity and pressure from vorticity, spectrally.

    Args:
        w: Vorticity, shape `(..., n, n)`, with axis -2 the `x` axis and axis -1 the `y` axis.

    Returns:
        Tuple `(psi, u, v, p)` of arrays with the same shape as `w`. `psi` and `p` have zero
        spatial mean and `u`, `v` are divergence-free to machine precision.
    """
    n = w.shape[-1]
    kx, ky, inv_k2 = wavenumbers(n)
    axes = (-2, -1)

    def to_grid(field_h):
        return np.fft.irfft2(field_h, s=(n, n), axes=axes)

    psi_h = np.fft.rfft2(w, axes=axes) * inv_k2
    u_h = 1j * ky * psi_h
    v_h = -1j * kx * psi_h
    psi = to_grid(psi_h)
    u = to_grid(u_h)
    v = to_grid(v_h)

    # div((u.grad)u) for a solenoidal velocity, evaluated on the grid (no dealiasing: the
    # products are formed at 64^2 so the Poisson solve below is exact on this grid).
    ux = to_grid(1j * kx * u_h)
    uy = to_grid(1j * ky * u_h)
    vx = to_grid(1j * kx * v_h)
    vy = to_grid(1j * ky * v_h)
    source = -(ux * ux + 2.0 * uy * vx + vy * vy)
    p = to_grid(-np.fft.rfft2(source, axes=axes) * inv_k2)

    return psi, u, v, p


def loader_dir(dataset_path):
    """Give a dataset a sibling directory holding the file name `NSLoader2D` hardcodes.

    NSLoader2D hardcodes the file name `kolmogorov_dataset.npz`, so the directory holds a
    symlink to the dataset and the loader can be pointed at it unchanged.

    Args:
        dataset_path: Path of the written npz.

    Returns:
        The directory to hand `NSLoader2D`.
    """
    directory = dataset_path.parent / dataset_path.stem
    directory.mkdir(exist_ok=True)
    link = directory / "kolmogorov_dataset.npz"
    link.unlink(missing_ok=True)
    link.symlink_to(Path("..") / dataset_path.name)
    return directory


def split_trajectories(n_traj, n_val, n_test, seed):
    """Partition trajectory indices into train/val/test by a seeded permutation.

    Args:
        n_traj: Total number of trajectories in the source file.
        n_val: Number of validation trajectories.
        n_test: Number of test trajectories.
        seed: Seed for the permutation.

    Returns:
        Dict with keys `train`, `val` and `test`, each a sorted array of trajectory indices.
    """
    order = np.random.default_rng(seed).permutation(n_traj)
    val, test, train = order[:n_val], order[n_val : n_val + n_test], order[n_val + n_test :]
    return {"train": np.sort(train), "val": np.sort(val), "test": np.sort(test)}


CHANNELS = {"vorticity": ("omega",), "velocity": ("u", "v", "p")}
# Channels NSLoader2D scales isotropically (a shared RMS) so the velocity direction survives.
VELOCITY_CHANNELS = {"vorticity": None, "velocity": (0, 1)}
NU = 1.0 / 500.0
T_INTERVAL = 1.0  # data satisfy NS at dt = 1/64, not 0.5/64 (issue #20)
FORCING = "-4*cos(4*y) (vorticity form); F = (sin(4*y), 0) in velocity form"


def _channels_and_psi(w, field):
    """Return the `(n, C, nx, ny)` channel stack and, for velocity, the streamfunction."""
    if field == "vorticity":
        return w[:, None, :, :], None
    psi, u, v, p = invert_vorticity(w)
    return np.stack([u, v, p], axis=1), psi[:, None, :, :]


def build_dataset(source_path, out_dir, field, seed=1234, n_val=200, n_test=200, chunk=100):
    """Convert the PINO-paper vorticity file into one-step pairs and write them as an npz.

    Samples are ordered trajectory-major (`index = local trajectory * n_pairs + step`), which
    is what `NSLoader2D.transform_rollout` assumes when it rebuilds rollout trajectories.
    Arrays are staged as memory-mapped `.npy` files and then stored into the npz, so peak
    memory stays at one chunk of trajectories rather than the whole split.

    Args:
        source_path: Path to `NS_fft_Re500_T4000.npy`, shape `(n_traj, n_time, nx, ny)`.
        out_dir: Directory to write `pino_kf_<field>.npz` into.
        field: Either `vorticity` (channel `omega`) or `velocity` (channels `u`, `v`, `p`).
        seed: Seed for the trajectory split.
        n_val: Number of validation trajectories.
        n_test: Number of test trajectories.
        chunk: Number of trajectories converted at a time.

    Returns:
        Dict with the dataset path, the loader-compatible directory, the per-split shapes and
        the per-channel RMS of the training inputs.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src = np.load(source_path, mmap_mode="r")
    n_traj, n_time, nx, ny = src.shape
    n_pairs = n_time - 1
    dt = T_INTERVAL / n_pairs
    channels = CHANNELS[field]
    split = split_trajectories(n_traj, n_val, n_test, seed)

    stage = Path(tempfile.mkdtemp(dir=out_dir, prefix=".stage_"))
    members = {}
    shapes = {}
    sums = np.zeros(len(channels))
    sumsq = np.zeros(len(channels))
    count = 0

    def stage_array(name, shape):
        members[name] = stage / f"{name}.npy"
        return open_memmap(members[name], mode="w+", dtype=np.float32, shape=shape)

    try:
        for state, ids in split.items():
            n = len(ids) * n_pairs
            shapes[state] = (n, len(channels), nx, ny)
            arrays = {
                "X": stage_array(f"X_{state}", (n, len(channels), nx, ny)),
                "y": stage_array(f"y_{state}", (n, len(channels), nx, ny)),
            }
            if field == "velocity":
                arrays["psi_X"] = stage_array(f"psi_X_{state}", (n, 1, nx, ny))
                arrays["psi_y"] = stage_array(f"psi_y_{state}", (n, 1, nx, ny))
            for start in range(0, len(ids), chunk):
                block = ids[start : start + chunk]
                raw = np.asarray(src[block], dtype=np.float64).reshape(-1, nx, ny)
                w = project_onto_curl_range(raw)
                stack, psi = _channels_and_psi(w, field)
                stack = stack.reshape(len(block), n_time, len(channels), nx, ny)
                lo, hi = start * n_pairs, (start + len(block)) * n_pairs
                arrays["X"][lo:hi] = stack[:, :-1].reshape(-1, len(channels), nx, ny)
                arrays["y"][lo:hi] = stack[:, 1:].reshape(-1, len(channels), nx, ny)
                if psi is not None:
                    psi = psi.reshape(len(block), n_time, 1, nx, ny)
                    arrays["psi_X"][lo:hi] = psi[:, :-1].reshape(-1, 1, nx, ny)
                    arrays["psi_y"][lo:hi] = psi[:, 1:].reshape(-1, 1, nx, ny)
                if state == "train":
                    # Normaliser statistics come from the training inputs only.
                    written = np.asarray(arrays["X"][lo:hi], dtype=np.float64)
                    sums += written.sum(axis=(0, 2, 3))
                    sumsq += (written**2).sum(axis=(0, 2, 3))
                    count += written.shape[0] * nx * ny
            for array in arrays.values():
                array.flush()
            arrays.clear()

        coords = np.linspace(0.0, DOMAIN_LENGTH, nx, endpoint=False)
        times = np.stack([np.arange(n_pairs) * dt, (np.arange(n_pairs) + 1) * dt], axis=1)
        metadata = {
            "field": np.asarray(field.encode()),
            "channels": np.asarray([c.encode() for c in channels]),
            "nu": np.asarray(NU),
            "dt": np.asarray(dt),
            "t_start": np.asarray(0.0),
            "t_end": np.asarray(T_INTERVAL),
            "domain_length": np.asarray(DOMAIN_LENGTH),
            "forcing": np.asarray(FORCING.encode()),
            "axis_order": np.asarray(b"(N, C, x, y); axis -2 is x, axis -1 is y"),
            "sign_convention": np.asarray(
                b"omega = dv/dx - du/dy; Delta psi = -omega; u = dpsi/dy; v = -dpsi/dx"
            ),
            "vorticity_projection": np.asarray(
                b"mean mode and grid-Nyquist row/column removed, so omega lies in the "
                b"range of the discrete curl and E0/E1 describe the same flow"
            ),
            "source": np.asarray(str(source_path).encode()),
            "seed": np.asarray(seed),
            "x_coords": coords,
            "y_coords": coords.copy(),
        }
        for state, ids in split.items():
            metadata[f"{state}_realisations"] = ids
            metadata[f"realisation_ids_{state}"] = np.repeat(ids, n_pairs).astype(np.int32)
            metadata[f"times_{state}"] = np.tile(times, (len(ids), 1))
        for name, value in metadata.items():
            members[name] = stage / f"{name}.npy"
            np.save(members[name], value)

        dataset_path = out_dir / f"pino_kf_{field}.npz"
        with zipfile.ZipFile(dataset_path, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
            for name, path in members.items():
                zf.write(path, arcname=f"{name}.npy")
    finally:
        shutil.rmtree(stage, ignore_errors=True)

    mean, mean_square = sums / count, sumsq / count
    stats = normaliser_stats(mean, mean_square - mean**2, VELOCITY_CHANNELS[field])
    normaliser_path = out_dir / f"pino_kf_{field}_normstat.pt"
    torch.save(stats, normaliser_path)

    return {
        "dataset_path": dataset_path,
        "loader_dir": loader_dir(dataset_path),
        "normaliser_path": normaliser_path,
        "shapes": shapes,
        "channels": channels,
        "rms": dict(zip(channels, np.sqrt(mean_square), strict=True)),
        "mean": dict(zip(channels, mean, strict=True)),
    }


def normaliser_stats(mean, var, velocity_channels, eps=1e-8):
    """Build the per-channel normaliser `NSLoader2D` consumes, from streamed moments.

    This reproduces `NSLoader2D.normalize`: every channel is centred on its global mean and
    divided by its global RMS deviation, except that the velocity channels share one
    isotropic scale. The shared scale picks up `eps` twice, once when it is first formed and
    once when the loader re-imposes isotropy; both are kept here so the two agree.

    Args:
        mean: Per-channel mean of the training inputs, shape `(C,)`.
        var: Per-channel mean squared deviation of the training inputs, shape `(C,)`.
        velocity_channels: Channel indices to scale isotropically, or `None`.
        eps: Floor added inside each square root, matching the loader.

    Returns:
        Dict with `mean` and `std` tensors of shape `(C, 1, 1)` plus the loader's own
        `velocity_channels` and `normalization` tags.
    """
    std = np.sqrt(var + eps)
    if velocity_channels is not None and len(velocity_channels) >= 2:
        shared = np.sqrt(var[list(velocity_channels)].mean() + eps)
        std[list(velocity_channels)] = np.sqrt(shared**2 + eps)
    return {
        "mean": torch.tensor(mean, dtype=torch.float32).reshape(-1, 1, 1),
        "std": torch.tensor(std, dtype=torch.float32).reshape(-1, 1, 1),
        "velocity_channels": velocity_channels,
        "normalization": "global_channel_rms",
    }


SOURCE = "DATA_ROOT/pino/NS_fft_Re500_T4000.npy"


def main(argv=None):
    """Run the conversion from the command line and print a summary of what was written."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default=SOURCE, help="PINO-paper vorticity .npy")
    parser.add_argument("--out-dir", default="DATA_ROOT/pino_kf")
    parser.add_argument("--field", choices=sorted(CHANNELS), action="append", default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--n-val", type=int, default=200)
    parser.add_argument("--n-test", type=int, default=200)
    parser.add_argument("--chunk", type=int, default=100, help="trajectories per conversion step")
    args = parser.parse_args(argv)

    for field in args.field or sorted(CHANNELS):
        summary = build_dataset(
            args.source, args.out_dir, field, args.seed, args.n_val, args.n_test, args.chunk
        )
        print(f"[{field}] wrote {summary['dataset_path']}")
        print(f"[{field}] loader dir {summary['loader_dir']}")
        print(f"[{field}] normaliser {summary['normaliser_path']}")
        for state, shape in summary["shapes"].items():
            print(f"[{field}]   {state}: X/y {shape}")
        for name in summary["channels"]:
            print(
                f"[{field}]   {name}: train mean {summary['mean'][name]:+.6e}, "
                f"RMS {summary['rms'][name]:.6e}"
            )


if __name__ == "__main__":
    main()
