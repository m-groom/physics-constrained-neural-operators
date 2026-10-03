"""Build the E3 dataset from a stochastically forced 2D turbulence run (m-groom/NS2D).

Two sources are read down the same path. The production run writes an in-situ coarse handler
(`add_task(..., scales=1/16)`, an exact spectral truncation of the 1024^2 state onto 64^2);
where that is unavailable the full-resolution `snapshots` handler is truncated offline, keeping
`|kx|, |ky| < n_coarse / 2` and zeroing the coarse grid's Nyquist row and column. The two agree
to round-off, which `read_run` is tested against. `--cutoff two-thirds` stops the retained band
at NS2D's rectangular dealiasing filter instead, `|kx|, |ky| <= n_coarse // 3`, for a dataset
whose nonlinear products do not alias on the coarse grid; the rule is recorded as
`cutoff_rule`.

`u`, `v` and `p` come from the file; `omega` and `psi` are always derived spectrally from the
truncated velocity, so both lie in the range of the discrete curl. Deriving them is not a
fallback: the production 1024^2 output writes only `velocity`, `pressure`, `vorticity` and
`forcing`, and where a solver does write its own `vorticity` or `streamfunction` they are used
to cross-check the convention, never as the stored field. The conventions are those of
`utils/criterion.py` and the NS2D solver, the opposite of the more common streamfunction sign:

    omega = dv/dx - du/dy,   Delta psi = -omega,   u = dpsi/dy,   v = -dpsi/dx.

Where the solver wrote its own `vorticity` or `streamfunction`, `read_run` checks the derived
fields against them and refuses data that follows a different convention; where it stamped a
`vorticity_convention` or `streamfunction_convention` attribute, that is read too, recorded, and
refused if it states the opposite. A run offering neither says so in the diagnostics rather than
passing quietly.

Physics metadata comes from the solver's `run_config.jsonl` where the run wrote one, and from
`--nu`, `--alpha`, `--kmin` and `--kmax` where it did not. The 2026 production run predates that
file and needs `--nu 9.3e-6 --alpha 3e-3 --kmin 56 --kmax 72`; its band sits well above the 64^2
cut-off of 32, so the truncated forcing is identically zero. The 2026 production runs stamp
their own `run_config.jsonl` and need none of those overrides.

`--stride` keeps every n-th snapshot, so one pair spans that many snapshot intervals; the
production cadence of `coarse_dt = 0.08` is far finer than the flow evolves. `recommend_stride`
reads the run's own `spectra.h5` and reports the stride that would make a pair interval one
eddy-turnover time `tau(k) = (k^3 E(k))^(-1/2)` at the smallest scales the coarse grid resolves,
`k` in 20 to 32. It is reported, never imposed.

There are two ways to split the data, and `--split` chooses between them. Under the default
`time` split a single long run is cut along time into train, validation and test at
70 / 15 / 15, with `--gap` snapshots discarded between the segments and the spin-up dropped by
`--t-start` or `--skip`; further realisations, if the run has any, become a `test_extra` split
in the same file, which `NSLoader2D(state="test_extra")` reads like any other.

Under the `realisation` split each realisation belongs to one split whole, so no state of a
validation or test trajectory appears anywhere in training:

    --split realisation --train-realisations 1234,1235,1236 \
        --val-realisations 1237 --test-realisations 1238

A realisation is identified by the seed it drove itself with, which it records in its own
`run_config.jsonl`; a run holding several `realisation_NNNN` directories under one shared
config is identified by the number in the directory name instead. The E3 production run puts
one seed in each run directory, `seed<N>/Nx1024_Ny1024_nu9e-06/realisation_0000`, so `--run-dir`
is repeated once per realisation. Every split takes all of the post-spin-up snapshots of its
realisations, trimmed to the window they all share so that each contributes the same number of
pairs; pairs are formed inside a realisation and the realisations are then concatenated in the
order given, so no pair straddles two of them and a rollout stays inside one as long as its
length divides that count. That count is the split's own `split_bounds`, `stop - start - 1`,
which is the `T` to hand `NSLoader2D.transform_rollout`; the loader reshapes blindly, so a `T`
that does not divide it would run a trajectory across a boundary. A fourth split, `test_time`, is the last `--time-test-fraction` of
each *training* realisation, held out behind the same `--gap`: the same trajectories the model
trained on, later in time, which separates the cost of predicting a later time from the cost of
predicting another realisation of the same flow. `--decorrelation-frames`, where the run's
decorrelation time is known, is recorded in the metadata; it is what `--stride` and `--gap` are
chosen from.

Two datasets are written per run, matching the layout of `prepare_pino_kf.py`: `X_<state>`,
`y_<state>` of shape `(N, C, nx, ny)` float32 with `times_<state>` of shape `(N, 2)`, the
velocity file adding `psi_X_<state>` and `psi_y_<state>`. The remaining keys are metadata:
`field`, `channels`, `nu`, `alpha`, `dt`, `dt_max_deviation`, `t_start`, `t_end`,
`domain_length`, `cutoff`, `cutoff_rule`, `cutoff_kmax`, `n_fine`, `handler`, `has_pressure`,
`gap`, `skip`, `stride`, `split_mode`, `decorrelation_frames`, `time_test_fraction`, `forcing`,
`forcing_kmin`, `forcing_kmax`, `forcing_retained_fraction`,
`forcing_rms_relative_to_velocity`, `axis_order`, `sign_convention`,
`source`, `source_realisations`, `source_git_commit`, `source_files`, `split_bounds`,
`split_states`, `x_coords`, `y_coords`, `<state>_realisations` and `realisation_ids_<state>`.
The normaliser is computed from the training inputs only, and `has_pressure` is false when the
run wrote no pressure task, which leaves that channel zero.

Dedalus writes at the first timestep past each target time, so the snapshot interval carries a
jitter of one timestep; `dt` is the median interval and `dt_max_deviation` the largest relative
departure from it, recorded rather than hidden and refused above `--dt-tolerance`.

A forcing band that lies entirely above the coarse cut-off must truncate to nothing: that is
what leaves the coarse-grained residual unclosed, and it is asserted rather than assumed. A
resolvable band is recorded and left alone. The diagnostics written beside the datasets hold the
energy and enstrophy history with the split boundaries, the mean energy spectrum before and
after truncation, and both checks.
"""

import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np
import torch

from experiments.data_utils.prepare_pino_kf import (
    CHANNELS,
    DOMAIN_LENGTH,
    VELOCITY_CHANNELS,
    loader_dir,
    normaliser_stats,
    wavenumbers,
)
from utils.compute_physical_statistics import compute_spectra


def cutoff_kmax(n_out, cutoff="sharp"):
    """Largest wavenumber a cut-off rule keeps in either direction on an `n_out` grid."""
    if cutoff == "sharp":
        return n_out // 2 - 1
    if cutoff == "two-thirds":
        return n_out // 3
    raise ValueError(f"unknown cutoff rule {cutoff!r}; expected 'sharp' or 'two-thirds'")


def spectral_truncate(field, n_out, cutoff="sharp"):
    """Truncate a periodic field to an `n_out x n_out` grid.

    Under the `sharp` rule the retained band is `|kx|, |ky| < n_out / 2`, which is exactly what
    Dedalus's `add_task(..., scales=...)` writes in situ. The coarse grid's Nyquist row and
    column are zeroed either way, because `d/dx` and `d/dy` annihilate the Nyquist mode of the
    direction they act on and no real grid velocity field can carry one. Under the
    `two-thirds` rule the band stops at NS2D's rectangular dealiasing filter,
    `|kx|, |ky| <= n_out // 3`, so that nonlinear products formed on the coarse grid do not
    alias.

    Args:
        field: Array of shape `(..., n, n)` with axis -2 the `x` axis and axis -1 the `y` axis.
        n_out: Side of the coarse grid; must not exceed `n` and must be even.
        cutoff: `sharp` or `two-thirds`.

    Returns:
        The truncated field, shape `(..., n_out, n_out)`.
    """
    n = field.shape[-1]
    if n_out > n or n_out % 2:
        raise ValueError(f"n_out must be even and at most n; got n_out={n_out}, n={n}")
    half = n_out // 2
    fine = np.fft.rfft2(field, axes=(-2, -1))
    coarse = np.zeros((*field.shape[:-2], n_out, half + 1), dtype=complex)
    coarse[..., :half, :half] = fine[..., :half, :half]
    coarse[..., half + 1 :, :half] = fine[..., n - half + 1 :, :half]
    kmax = cutoff_kmax(n_out, cutoff)
    if kmax < half - 1:
        kx = np.fft.fftfreq(n_out, d=1.0 / n_out)[:, None]
        ky = np.fft.rfftfreq(n_out, d=1.0 / n_out)[None, :]
        coarse[..., (np.abs(kx) > kmax) | (ky > kmax)] = 0.0
    scale = (n_out / n) ** 2
    return np.fft.irfft2(coarse, s=(n_out, n_out), axes=(-2, -1)) * scale


def derive_vorticity_and_streamfunction(u, v, domain_length=DOMAIN_LENGTH):
    """Derive `omega` and `psi` spectrally from a velocity field.

    The conventions are those of `utils/criterion.py` and of the NS2D solver:
    `omega = dv/dx - du/dy`, `Delta psi = -omega` (so `psi_hat = omega_hat / |k|^2`),
    `u = dpsi/dy` and `v = -dpsi/dx`. `psi` carries zero spatial mean, and the mean velocity
    of the field, which no streamfunction can represent, is not recoverable from it.

    Args:
        u: x-velocity, shape `(..., n, n)`, axis -2 is `x` and axis -1 is `y`.
        v: y-velocity, same shape as `u`.
        domain_length: Side of the periodic box, which NS2D takes as `--Lx` and need not be
            `2 pi`; the wavenumbers scale with it.

    Returns:
        Tuple `(omega, psi)` of arrays with the same shape as `u`.
    """
    n = u.shape[-1]
    kx, ky, inv_k2 = wavenumbers(n, domain_length)
    axes = (-2, -1)
    omega_h = 1j * kx * np.fft.rfft2(v, axes=axes) - 1j * ky * np.fft.rfft2(u, axes=axes)
    omega = np.fft.irfft2(omega_h, s=(n, n), axes=axes)
    psi = np.fft.irfft2(omega_h * inv_k2, s=(n, n), axes=axes)
    return omega, psi


FRACTIONS = (0.70, 0.15, 0.15)


def split_along_time(n_snapshots, gap):
    """Cut a single run into contiguous train/val/test segments separated by `gap` snapshots.

    The gaps are discarded, so no training snapshot is within `gap` steps of a validation or
    test snapshot and the segments decorrelate.

    The validation and test segments are floored at two snapshots, which is one pair, so a
    short run still exercises every split.

    Args:
        n_snapshots: Number of snapshots available after the spin-up has been dropped.
        gap: Snapshots discarded between consecutive segments.

    Returns:
        Dict mapping `train`, `val`, `test` to a half-open `(start, stop)` index range.
    """
    usable = n_snapshots - 2 * gap
    if usable < 6:
        raise ValueError(
            f"run too short: {n_snapshots} snapshots leave {usable} after two gaps of {gap}"
        )
    # Two snapshots is one pair, so a segment shorter than that would carry no sample at all.
    n_val = max(2, round(FRACTIONS[1] * usable))
    n_test = max(2, round(FRACTIONS[2] * usable))
    sizes = (usable - n_val - n_test, n_val, n_test)
    split = {}
    start = 0
    for state, size in zip(("train", "val", "test"), sizes, strict=True):
        split[state] = (start, start + size)
        start += size + gap
    return split


TASK_ALIASES = {
    "velocity": ("velocity", "vel", "velocity_vector", "u"),
    "velocity_x": ("velocity_x", "ux", "u_x", "u"),
    "velocity_y": ("velocity_y", "uy", "u_y", "v"),
    "pressure": ("pressure", "p"),
    "vorticity": ("vorticity", "vort", "omega", "w"),
    "streamfunction": ("streamfunction", "stream", "psi"),
    "forcing": ("forcing", "force", "f"),
}


def _strip_handler_decoration(name):
    """Strip the decoration a handler may add to a task name (`coarse_u`, `u_coarse`)."""
    return name.lower().removeprefix("coarse_").removesuffix("_coarse")


def find_task(keys, field):
    """Return the task name in `keys` holding `field`, or `None` if it is absent.

    The h5 task names the solver writes are not contractual, so each field is matched against
    a list of aliases, most specific first, on the name with any `coarse_`/`_coarse`
    decoration removed.

    Args:
        keys: Task names present in the file.
        field: One of the keys of `TASK_ALIASES`.

    Returns:
        The matching key from `keys`, or `None`.
    """
    normalised = {_strip_handler_decoration(k): k for k in keys}
    for alias in TASK_ALIASES[field]:
        if alias in normalised:
            return normalised[alias]
    return None


def _set_number(path):
    """Sort key for a Dedalus output set: the integer in the `_s{N}.h5` suffix."""
    match = re.search(r"_s(\d+)\.h5$", path.name)
    return int(match.group(1)) if match else 0


def handler_files(run_dir, handler=None):
    """Locate the top-level h5 sets of one output handler, in write order.

    Only the virtual `*_s{N}.h5` files are returned; the per-rank `_p{N}.h5` shards live one
    directory down and reading them directly would duplicate and mis-order every sample.

    Args:
        run_dir: A realisation directory holding the handler subdirectories.
        handler: Handler name, or `None` to prefer the in-situ coarse output over the
            full-resolution snapshots.

    Returns:
        Tuple `(handler, files)` with `files` sorted by set number.
    """
    run_dir = Path(run_dir)
    candidates = [handler] if handler else ["coarse", "snapshots"]
    for name in candidates:
        files = sorted((run_dir / name).glob(f"{name}_s*.h5"), key=_set_number)
        files = [f for f in files if f.is_file()]
        if files:
            return name, files
    raise FileNotFoundError(f"no {'/'.join(str(c) for c in candidates)} h5 sets under {run_dir}")


def _grid_coords(f, axis):
    """Read a grid axis from `scales/{axis}_hash_*`, whose suffix is a Dedalus content hash."""
    scales = f["scales"]
    for key in scales:
        if key.startswith(f"{axis}_hash_"):
            return np.asarray(scales[key])
    raise KeyError(f"no {axis} grid in scales/: {sorted(scales)}")


CONVENTION_ATTRS = {
    "vorticity_convention": (
        ("dx(v) - dy(u)", "dv/dx - du/dy"),
        ("dy(u) - dx(v)", "du/dy - dv/dx"),
    ),
    "streamfunction_convention": (
        ("lap(psi) = -omega", "delta psi = -omega"),
        ("lap(psi) = omega", "delta psi = omega"),
    ),
}


def check_convention_attrs(attrs):
    """Read the sign convention the solver recorded in the file, and refuse the opposite one.

    NS2D stamps `vorticity_convention` and `streamfunction_convention` on each output file.
    Wording that this function does not recognise is recorded and left to the numerical check;
    wording that states the opposite convention is refused outright.

    Args:
        attrs: The h5 root attributes.

    Returns:
        Dict of the convention attributes the file carries, verbatim.

    Raises:
        ValueError: If an attribute states the opposite sign convention.
    """
    recorded = {}
    for key, (agrees, disagrees) in CONVENTION_ATTRS.items():
        text = str(attrs.get(key, "")).strip()
        if not text:
            continue
        recorded[key] = text
        lowered = " ".join(text.lower().split())
        if any(form in lowered for form in agrees):
            continue
        if any(form in lowered for form in disagrees):
            raise ValueError(
                f"{key} records the opposite convention to the one this builder assumes "
                f"(omega = dv/dx - du/dy, Delta psi = -omega): {text!r}"
            )
    return recorded


def check_conventions(u, v, omega_stored, psi_stored, domain_length=DOMAIN_LENGTH, tol=1e-6):
    """Verify the sign convention of the data against the one this builder assumes.

    Args:
        u: x-velocity, shape `(n_time, n, n)`.
        v: y-velocity, same shape.
        omega_stored: Vorticity as written by the solver, or `None` if it was not written.
        psi_stored: Streamfunction as written by the solver, or `None`.
        domain_length: Side of the periodic box.
        tol: Largest relative L2 difference accepted between derived and stored fields.

    Returns:
        Dict of relative L2 differences, one per field that was available to compare.

    Raises:
        ValueError: If a stored field disagrees with the assumed convention.
    """
    omega, psi = derive_vorticity_and_streamfunction(u, v, domain_length)
    errors = {}
    for name, derived, stored in (("omega", omega, omega_stored), ("psi", psi, psi_stored)):
        if stored is None:
            continue
        stored = stored - stored.mean(axis=(-2, -1), keepdims=True) if name == "psi" else stored
        scale = np.sqrt(np.mean(stored**2))
        errors[name] = float(np.sqrt(np.mean((derived - stored) ** 2)) / max(scale, 1e-300))
        flipped = float(np.sqrt(np.mean((derived + stored) ** 2)) / max(scale, 1e-300))
        if errors[name] > tol:
            raise ValueError(
                f"stored {name} disagrees with the assumed convention "
                f"(omega = dv/dx - du/dy, Delta psi = -omega): relative L2 {errors[name]:.3e}, "
                f"{flipped:.3e} with the sign flipped"
            )
    return errors


def read_run(run_dir, n_coarse, handler=None, cutoff="sharp", chunk=32, spectra_samples=64):
    """Read one realisation, truncate every field to `n_coarse` and derive `omega` and `psi`.

    Snapshot sets are read in write order and frames repeated by a restart are dropped. The
    in-situ coarse handler and the full-resolution snapshots go down the same path: the
    truncation is a no-op on data already written at `n_coarse`, up to the Nyquist row and
    column, which are zeroed either way.

    Args:
        run_dir: Realisation directory holding the handler subdirectories.
        n_coarse: Side of the coarse grid.
        handler: Handler name, or `None` to prefer `coarse` over `snapshots`.
        cutoff: `sharp` (`|kx|, |ky| < n_coarse / 2`) or `two-thirds` (the dealiasing band).
        chunk: Snapshots read and truncated at a time.
        spectra_samples: Snapshots used for the mean before/after energy spectra.

    Returns:
        Dict holding `u`, `v`, `p`, `omega`, `psi` of shape `(n_time, n_coarse, n_coarse)`
        float32, the `times`, the coarse grid `x` and `y`, the source `handler`, `n_fine`,
        `domain_length`, the file `attrs`, the `conventions` check, the `forcing` diagnostic
        and the mean `spectra` before and after truncation.
    """
    handler, files = handler_files(run_dir, handler)
    with h5py.File(files[0], "r") as f:
        keys = list(f["tasks"])
        x_fine = _grid_coords(f, "x")
        attrs = {k: _plain(v) for k, v in f.attrs.items()}
    n_fine = len(x_fine)
    domain_length = float(n_fine * (x_fine[1] - x_fine[0]))
    tasks = {field: find_task(keys, field) for field in TASK_ALIASES}
    if tasks["velocity"] is None and (tasks["velocity_x"] is None or tasks["velocity_y"] is None):
        raise KeyError(f"no velocity task in {files[0]}: {sorted(keys)}")

    selected, times = _dedupe_times(files)
    n_time = len(times)
    fields = {
        name: np.empty((n_time, n_coarse, n_coarse), dtype=np.float32)
        for name in ("u", "v", "p", "omega", "psi")
    }
    stride = max(1, n_time // max(spectra_samples, 1))
    spectra = {"fine": None, "coarse": None, "count": 0, "k": None}
    forcing = (
        {"square_fine": 0.0, "square_coarse": 0.0, "count_fine": 0, "count_coarse": 0}
        if tasks["forcing"]
        else None
    )
    conventions = check_convention_attrs(attrs)
    numerical = None

    for file_index, path in enumerate(files):
        local = [(li, pos) for fi, li, pos in selected if fi == file_index]
        if not local:
            continue
        with h5py.File(path, "r") as f:
            for start in range(0, len(local), chunk):
                block = local[start : start + chunk]
                index = [li for li, _ in block]
                out = [pos for _, pos in block]
                u_fine, v_fine = _read_velocity(f, tasks, index)
                u = spectral_truncate(u_fine, n_coarse, cutoff)
                v = spectral_truncate(v_fine, n_coarse, cutoff)
                omega, psi = derive_vorticity_and_streamfunction(u, v, domain_length)
                fields["u"][out], fields["v"][out] = u, v
                fields["omega"][out], fields["psi"][out] = omega, psi
                if tasks["pressure"]:
                    fields["p"][out] = spectral_truncate(
                        np.asarray(f["tasks"][tasks["pressure"]][index], dtype=float),
                        n_coarse,
                        cutoff,
                    )
                if numerical is None:
                    numerical = check_conventions(
                        u,
                        v,
                        *_stored_omega_psi(f, tasks, index, n_coarse, cutoff),
                        domain_length=domain_length,
                    )
                if forcing is not None:
                    _accumulate_forcing(f, tasks["forcing"], index, n_coarse, forcing, cutoff)
                _accumulate_spectra(spectra, u_fine, v_fine, u, v, out, stride, domain_length)

    conventions.update(
        numerical
        if numerical
        else {
            "numerical_check": "not possible: the run wrote neither vorticity nor "
            "streamfunction, so the assumed convention stands unverified"
        }
    )
    if tasks["pressure"] is None:
        fields["p"][:] = 0.0
    return {
        **fields,
        "times": times,
        "x": np.linspace(0.0, domain_length, n_coarse, endpoint=False),
        "y": np.linspace(0.0, domain_length, n_coarse, endpoint=False),
        "handler": handler,
        "n_fine": n_fine,
        "cutoff": cutoff,
        "domain_length": domain_length,
        "attrs": attrs,
        "conventions": conventions,
        "forcing": _finish_forcing(forcing, fields["u"], fields["v"], n_fine > n_coarse),
        "spectra": _finish_spectra(spectra),
        "has_pressure": tasks["pressure"] is not None,
        "files": [str(p) for p in files],
    }


def _plain(value):
    """Convert an h5 attribute to something `json` and `numpy.savez` can both hold."""
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _dedupe_times(files):
    """Index the snapshots across sets in time order, dropping frames a restart repeated.

    Args:
        files: Handler sets in write order.

    Returns:
        Tuple `(selected, times)` where `selected` lists `(file_index, local_index, position)`
        in file order and `times` is the sorted array of unique simulation times.
    """
    seen = {}
    for file_index, path in enumerate(files):
        with h5py.File(path, "r") as f:
            for local_index, t in enumerate(np.asarray(f["scales"]["sim_time"])):
                seen.setdefault(round(float(t), 9), (file_index, local_index))
    times = np.array(sorted(seen))
    order = {key: position for position, key in enumerate(times)}
    selected = sorted(
        (file_index, local_index, order[key]) for key, (file_index, local_index) in seen.items()
    )
    return selected, times


def _read_velocity(f, tasks, index):
    """Read `u` and `v`, whether they are one vector task or two scalar tasks."""
    name = tasks["velocity"]
    if name is not None and f["tasks"][name].ndim == 4:
        vector = np.asarray(f["tasks"][name][index], dtype=float)
        return vector[:, 0], vector[:, 1]
    u = np.asarray(f["tasks"][tasks["velocity_x"]][index], dtype=float)
    v = np.asarray(f["tasks"][tasks["velocity_y"]][index], dtype=float)
    return u, v


def _stored_omega_psi(f, tasks, index, n_coarse, cutoff="sharp"):
    """Truncate the solver's own `omega` and `psi`, where it wrote them, for the sign check."""
    out = []
    for field in ("vorticity", "streamfunction"):
        name = tasks[field]
        if name is None:
            out.append(None)
        else:
            out.append(
                spectral_truncate(
                    np.asarray(f["tasks"][name][index], dtype=float), n_coarse, cutoff
                )
            )
    return out


def _accumulate_forcing(f, name, index, n_coarse, forcing, cutoff="sharp"):
    """Accumulate the mean square of the forcing before and after truncation.

    Each grid carries its own point count: by Parseval the mean square of the truncated field
    is the mean square of the modes the cut-off keeps, so the ratio of the two means is the
    fraction of the forcing that survives. Comparing sums instead would report the ratio of
    the grid sizes.
    """
    fine = np.asarray(f["tasks"][name][index], dtype=float)
    coarse = spectral_truncate(fine, n_coarse, cutoff)
    forcing["square_fine"] += float((fine**2).sum())
    forcing["square_coarse"] += float((coarse**2).sum())
    forcing["count_fine"] += fine.size
    forcing["count_coarse"] += coarse.size


def _finish_forcing(forcing, u, v, truncated):
    """Turn the accumulated sums into the RMS of the forcing before and after truncation.

    `retained_fraction` says what the truncation removed, which only a source finer than the
    coarse grid can answer: the in-situ coarse handler is already truncated, so it truncates
    to itself and retains all of its own RMS whatever that RMS is. `truncated` says which of
    the two the source was. `rms_relative_to_velocity` is the same statement in a form both
    sources can make, the size of the coarse forcing beside the coarse velocity.
    """
    if forcing is None:
        return None
    fine = np.sqrt(forcing["square_fine"] / forcing["count_fine"])
    coarse = np.sqrt(forcing["square_coarse"] / forcing["count_coarse"])
    velocity = np.sqrt(_mean_square(u).mean() + _mean_square(v).mean())
    return {
        "rms_fine": float(fine),
        "rms_coarse": float(coarse),
        "rms_velocity": float(velocity),
        "truncated": bool(truncated),
        "retained_fraction": float(coarse / fine) if fine > 0 else 0.0,
        "rms_relative_to_velocity": float(coarse / velocity) if velocity > 0 else 0.0,
    }


def _accumulate_spectra(spectra, u_fine, v_fine, u, v, positions, stride, domain_length):
    """Accumulate the mean energy spectrum before and after truncation, on a subsample."""
    for local, position in enumerate(positions):
        if position % stride:
            continue
        k, e_fine, _ = compute_spectra(u_fine[local], v_fine[local], domain_length, domain_length)
        _, e_coarse, _ = compute_spectra(u[local], v[local], domain_length, domain_length)
        if spectra["fine"] is None:
            spectra["k"] = k
            spectra["fine"] = np.zeros_like(e_fine)
            spectra["coarse"] = np.zeros_like(e_fine)
        spectra["fine"] += e_fine
        spectra["coarse"][: len(e_coarse)] += e_coarse
        spectra["count"] += 1


def _finish_spectra(spectra):
    """Turn the accumulated spectra into means, or return `None` if nothing was sampled."""
    if not spectra["count"]:
        return None
    return {
        "k": spectra["k"],
        "E_fine": spectra["fine"] / spectra["count"],
        "E_coarse": spectra["coarse"] / spectra["count"],
        "samples": spectra["count"],
    }


def mean_energy_spectrum(spectra_path):
    """Average the shell-averaged energy spectra a run wrote across its output times.

    `spectra.h5` holds one `(nbins, 3)` dataset per output time, columns `[k, E(k), Z(k)]`,
    with the time encoded in the dataset name rather than in a scale, so the keys are
    enumerated rather than assumed.

    Args:
        spectra_path: Path to the run's `spectra.h5`.

    Returns:
        Tuple `(k, E, n_times)`, or `None` if the file holds no energy spectra.
    """
    with h5py.File(spectra_path, "r") as f:
        names = [name for name in f if name.startswith("k_E_Z_t")]
        if not names:
            return None
        first = np.asarray(f[names[0]])
        total = np.zeros(len(first))
        for name in names:
            total += np.asarray(f[name])[:, 1]
    return first[:, 0], total / len(names), len(names)


def recommend_stride(run_dir, dt, k_range=(20, 32), target=1.0):
    """Recommend a time stride from the eddy-turnover time of the smallest resolved scales.

    The turnover time at wavenumber `k` is `tau(k) = (k^3 E(k))^(-1/2)`. Over the band the
    64^2 grid barely resolves, the shortest of those is the fastest motion the dataset has to
    carry, so a pair separated by about that time is neither a near-duplicate nor a jump the
    one-step map cannot follow. The number is reported, not imposed: pass the `--stride` you
    want.

    Args:
        run_dir: Realisation directory holding `spectra.h5`.
        dt: Snapshot interval of the handler being read.
        k_range: Wavenumber band the recommendation is taken over.
        target: Multiple of the shortest turnover time to aim a pair interval at.

    Returns:
        Dict with `tau_min`, the `k` it occurs at, the `k_range`, the number of `snapshots`
        averaged and the recommended `stride`, or `None` if the run wrote no spectra.
    """
    spectra_path = Path(run_dir) / "spectra.h5"
    if not spectra_path.exists():
        return None
    spectrum = mean_energy_spectrum(spectra_path)
    if spectrum is None:
        return None
    k, energy, n_times = spectrum
    band = (k >= k_range[0]) & (k <= k_range[1]) & (energy > 0.0)
    if not band.any():
        return None
    tau = 1.0 / np.sqrt(k[band] ** 3 * energy[band])
    fastest = int(np.argmin(tau))
    return {
        "tau_min": float(tau[fastest]),
        "k_at_tau_min": float(k[band][fastest]),
        "k_range": [float(k_range[0]), float(k_range[1])],
        "snapshots": n_times,
        "stride": max(1, round(target * float(tau[fastest]) / dt)),
    }


def realisation_dirs(run_dir):
    """List the realisation directories of a run, or the run directory itself if it is one.

    The E3 production layout runs one realisation per seed and puts each in its own
    directory, `seed<N>/Nx1024_Ny1024_nu9e-06/realisation_0000`, so the five realisations of
    one dataset arrive as five run directories rather than as five subdirectories of one.

    Args:
        run_dir: A run directory holding `realisation_*` subdirectories, one realisation
            directory, or a sequence of either.

    Returns:
        The realisation directories, those of each `run_dir` in the order it was given.
    """
    if not isinstance(run_dir, (str, Path)):
        return [d for one in run_dir for d in realisation_dirs(one)]
    run_dir = Path(run_dir)
    subdirectories = sorted(p for p in run_dir.glob("realisation_*") if p.is_dir())
    return subdirectories or [run_dir]


def read_run_config(run_dir):
    """Read the last record of the solver's `run_config.jsonl`, if it wrote one.

    Args:
        run_dir: A realisation directory, or the run directory above it.

    Returns:
        The parsed record, or `{}` when no provenance file is present.
    """
    run_dir = Path(run_dir)
    for candidate in (run_dir / "run_config.jsonl", run_dir.parent / "run_config.jsonl"):
        if candidate.exists():
            lines = [line for line in candidate.read_text().splitlines() if line.strip()]
            if lines:
                return json.loads(lines[-1])
    return {}


def realisation_id(directory):
    """The integer identifying one realisation: the solver's seed, else the directory number.

    The E3 production run drives each realisation with its own seed and writes it to that
    realisation's own `run_config.jsonl`, so the seed is the identifier the split is stated
    in. A run that holds several `realisation_NNNN` directories under one shared config is
    identified by the number in the directory name instead, which is what the `test_extra`
    split of the time split already numbers them by.

    Args:
        directory: A realisation directory.

    Returns:
        The realisation id.

    Raises:
        ValueError: If the directory carries neither a seed nor a number in its name.
    """
    directory = Path(directory)
    # Only the realisation's own config identifies it: a config one level up is shared by
    # every realisation of that run and would give them all the same id.
    config = read_run_config(directory) if (directory / "run_config.jsonl").exists() else {}
    seed = config.get("args", config).get("seed")
    if seed is not None:
        return int(seed)
    match = re.search(r"(\d+)$", directory.name)
    if match is None:
        raise ValueError(
            f"cannot identify the realisation {directory}: it records no seed in a "
            f"run_config.jsonl of its own and its name carries no number"
        )
    return int(match.group(1))


def select_realisations(runs, realisations):
    """Resolve the realisation ids of each split to the directories holding them.

    Args:
        runs: Realisation directories, as `realisation_dirs` returns them.
        realisations: Dict mapping `train`, `val` and `test` to lists of realisation ids.

    Returns:
        Dict mapping each split to its `(id, directory)` pairs, in the order the ids are
        given.

    Raises:
        ValueError: If a split is empty, an id names no realisation, two realisations carry
            the same id, or one id appears in two splits.
    """
    available = {}
    for path in runs:
        found = realisation_id(path)
        if found in available:
            raise ValueError(f"realisation {found} is both {available[found]} and {path}")
        available[found] = path
    chosen, owner = {}, {}
    for state in ("train", "val", "test"):
        ids = [int(i) for i in realisations.get(state) or ()]
        if not ids:
            raise ValueError(f"a realisation split needs at least one {state} realisation")
        for identifier in ids:
            if identifier in owner:
                raise ValueError(
                    f"realisation {identifier} is in both the {owner[identifier]} and the "
                    f"{state} split; a realisation belongs to one split only"
                )
            if identifier not in available:
                raise ValueError(
                    f"no realisation {identifier} among {sorted(available)}; pass one "
                    f"--run-dir per realisation"
                )
            owner[identifier] = state
        chosen[state] = [(identifier, available[identifier]) for identifier in ids]
    return chosen


def forcing_band(config, physics):
    """Return the `(kmin, kmax)` the forcing occupies, from the run config or an override.

    A deterministic Kolmogorov forcing occupies the single wavenumber `k_drive`; a stochastic
    forcing occupies the band `[kmin, kmax]`.

    Args:
        config: The parsed `run_config.jsonl` record, possibly empty.
        physics: CLI overrides; any key present here wins.

    Returns:
        Tuple `(kmin, kmax)`, either entry `None` when it cannot be determined.
    """
    if physics.get("kmin") is not None:
        return float(physics["kmin"]), float(physics.get("kmax") or physics["kmin"])
    if config.get("forcing") == "kolmogorov" and config.get("k_drive") is not None:
        return float(config["k_drive"]), float(config["k_drive"])
    if config.get("kmin") is not None:
        return float(config["kmin"]), float(config.get("kmax", config["kmin"]))
    return None, None


def check_forcing_band(band, n_coarse, forcing, cutoff="sharp", tol=1e-8):
    """Check what the truncation leaves of the forcing, and assert nothing when it must not.

    A band that lies entirely above the largest wavenumber the cut-off rule keeps must
    truncate to nothing; that is what makes the coarse-grained residual unclosed and is the
    premise of the E3 experiment, so it is asserted rather than assumed. A resolvable band is
    recorded and left alone.

    Two measures say the forcing vanished, and each is asserted where it means something.
    The coarse forcing RMS as a fraction of the coarse velocity RMS is one every source can
    state. The fraction of its own RMS the truncation retained is the sharper of the two,
    but only a source finer than the coarse grid can state it: the in-situ coarse handler is
    already the truncated field, so it truncates to itself and retains all of its own RMS
    whatever that RMS is.

    Args:
        band: `(kmin, kmax)` of the forcing, either entry possibly `None`.
        n_coarse: Side of the coarse grid.
        forcing: The `forcing` diagnostic from `read_run`, or `None` if it was not written.
        cutoff: The cut-off rule the fields were truncated with.
        tol: Largest fraction either measure may reach.

    Returns:
        Dict recording the band, the cut-off, whether the band is resolvable and, when the
        forcing field was written, the fraction of its RMS the truncation retains and its
        size relative to the coarse velocity.

    Raises:
        ValueError: If a band above the cut-off survives truncation.
    """
    kmin, kmax = band
    cutoff = cutoff_kmax(n_coarse, cutoff)
    resolved = None if kmin is None else bool(kmin <= cutoff)
    retained = None if forcing is None else forcing["retained_fraction"]
    relative = None if forcing is None else forcing["rms_relative_to_velocity"]
    if resolved is False and relative is not None and relative > tol:
        raise ValueError(
            f"forcing band [{kmin}, {kmax}] lies above the cut-off {cutoff} but the coarse "
            f"forcing is {relative:.3e} of the coarse velocity RMS"
        )
    # Only a source finer than the coarse grid can say what the truncation removed; an
    # already-coarse one truncates to itself and retains all of its own RMS regardless.
    measurable = retained if forcing is not None and forcing["truncated"] else None
    if resolved is False and measurable is not None and measurable > tol:
        raise ValueError(
            f"forcing band [{kmin}, {kmax}] lies above the cut-off {cutoff} but truncation "
            f"retains {measurable:.3e} of its RMS"
        )
    return {
        "kmin": kmin,
        "kmax": kmax,
        "cutoff": cutoff,
        "resolved_at_n_coarse": resolved,
        "retained_fraction": retained,
        "rms_relative_to_velocity": relative,
        "asserted_empty": bool(resolved is False and relative is not None),
        "assertion": _forcing_assertion(resolved, relative, cutoff),
    }


def _forcing_assertion(resolved, relative, cutoff):
    """Say in words what the forcing check was able to establish."""
    if resolved is None:
        return "band unknown: no run_config.jsonl and no --kmin"
    if resolved:
        return f"nothing to assert: the band is resolvable at the cut-off {cutoff}"
    if relative is None:
        return "not measurable: the run did not write the forcing field"
    return f"asserted empty: the coarse forcing is {relative:.3e} of the coarse velocity RMS"


AXIS_ORDER = b"(N, C, x, y); axis -2 is x, axis -1 is y"
SIGN_CONVENTION = b"omega = dv/dx - du/dy; Delta psi = -omega; u = dpsi/dy; v = -dpsi/dx"


def _subsample(run, stride):
    """Keep every `stride`-th snapshot, so one pair spans `stride` snapshot intervals."""
    kept = ("u", "v", "p", "omega", "psi", "times")
    subsampled = {name: run[name][::stride] for name in kept} if stride > 1 else {}
    return {**run, **subsampled, "stride": stride}


def _segment(run, channels, start, stop):
    """Stack the channels of one contiguous time segment into `(n, C, nx, ny)` float32."""
    return np.stack([run[c][start:stop] for c in channels], axis=1)


def _pair_times(times, start, stop):
    """One-step pair times `[t, t + dt]` for the snapshots of one segment."""
    return np.stack([times[start : stop - 1], times[start + 1 : stop]], axis=1)


def _psi_pair(run, field, state, start, stop):
    """The streamfunction arrays a velocity dataset carries for one segment, or nothing."""
    if field != "velocity":
        return {}
    psi = _segment(run, ("psi",), start, stop)
    return {f"psi_X_{state}": psi[:-1], f"psi_y_{state}": psi[1:]}


def build_dataset(
    run_dir,
    out_dir,
    prefix="ns2d_e3",
    n_coarse=64,
    fields=("velocity", "vorticity"),
    gap=20,
    t_start=None,
    skip=0,
    stride=1,
    handler=None,
    cutoff="sharp",
    physics=None,
    chunk=32,
    dt_tolerance=0.1,
    split="time",
    realisations=None,
    time_test_fraction=0.15,
    decorrelation_frames=None,
):
    """Convert a Dedalus run into the one-step-pair datasets `NSLoader2D` reads.

    Under the default `time` split the first realisation is split along time into train,
    validation and test segments, separated by `gap` discarded snapshots; any further
    realisation becomes a `test_extra` split in the same file, which
    `NSLoader2D(state="test_extra")` reads like any other.

    Under the `realisation` split each realisation belongs to one split whole: train,
    validation and test draw on the realisations `realisations` names, so no state of a
    validation or test trajectory appears anywhere in training. The E3 production run puts
    one seed in each run directory, so those realisations arrive as several `run_dir` entries
    rather than as subdirectories of one.

    Args:
        run_dir: Run directory holding `realisation_*` subdirectories, one realisation, or a
            sequence of either.
        out_dir: Directory to write the datasets, normalisers and diagnostics into.
        prefix: Stem of every file written.
        n_coarse: Side of the coarse grid the fields are truncated to.
        fields: Which channel sets to write; `velocity` is `(u, v, p)` plus `psi`, `vorticity`
            is `(omega)`.
        gap: Snapshots discarded between the splits.
        t_start: Simulation time the dataset starts at, skipping the spin-up.
        skip: Snapshots skipped instead, when `t_start` is not given, counted in strided
            snapshots.
        stride: Keep every `stride`-th snapshot, so one pair spans `stride` snapshot
            intervals; `recommend_stride` suggests one from the run's own spectra.
        handler: Output handler to read, or `None` to prefer the in-situ coarse output.
        cutoff: `sharp` (`|kx|, |ky| < n_coarse / 2`, what the in-situ output already is) or
            `two-thirds` (NS2D's rectangular dealiasing band, `|kx|, |ky| <= n_coarse // 3`).
        physics: Overrides for `nu`, `alpha`, `kmin` and `kmax` when the run wrote no
            `run_config.jsonl`.
        chunk: Snapshots read and truncated at a time.
        dt_tolerance: Largest relative jitter accepted in the snapshot cadence.
        split: `time` to split one realisation along time, or `realisation` to give each
            split its own realisations.
        realisations: Under the `realisation` split, the realisation ids of the `train`,
            `val` and `test` splits; `realisation_id` reads an id from the seed the
            realisation recorded.
        time_test_fraction: Fraction of each training realisation held out, behind the `gap`,
            as the in-distribution `test_time` split.
        decorrelation_frames: The decorrelation time of the run, in snapshots, which the
            `stride` and the `gap` are chosen from. Recorded in the metadata only.

    Returns:
        Dict with the dataset paths per field, the loader directories, the normaliser paths,
        the per-split shapes and the diagnostics.
    """
    physics = physics or {}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = realisation_dirs(run_dir)
    chosen = select_realisations(runs, realisations or {}) if split == "realisation" else None
    reference = chosen["train"][0][1] if chosen else runs[0]
    raw = read_run(reference, n_coarse, handler=handler, cutoff=cutoff, chunk=chunk)
    advice = recommend_stride(reference, float(np.median(np.diff(raw["times"]))))
    run = _subsample(raw, stride)
    dt, dt_deviation = _snapshot_interval(run["times"], tol=dt_tolerance)

    def read(path):
        """Read one further realisation, on the cadence the reference realisation set."""
        other = _subsample(
            read_run(path, n_coarse, handler=handler, cutoff=cutoff, chunk=chunk), stride
        )
        # A realisation on another cadence would put a different dt inside the same pairs.
        other_dt, _ = _snapshot_interval(other["times"], tol=dt_tolerance)
        if abs(other_dt - dt) > dt_tolerance * dt:
            raise ValueError(
                f"{path} has a snapshot cadence of {other_dt:.6g} against {dt:.6g} for "
                f"{reference}; its pairs would not span the same time step"
            )
        return other

    resolved = _resolve_physics(read_run_config(reference), physics, run)
    forcing_check = check_forcing_band(resolved["band"], n_coarse, run["forcing"], cutoff)
    if chosen:
        reads = {
            identifier: run if path == reference else read(path)
            for members in chosen.values()
            for identifier, path in members
        }
        # One index window for every realisation, so a split means the same stretch of the
        # flow in each of them and each contributes the same number of pairs.
        start = max(_spin_up_start(r["times"], t_start, skip) for r in reads.values())
        length = min(len(r["times"]) for r in reads.values())
        train_bounds, tail_bounds = time_test_tail(start, length, gap, time_test_fraction)
        bounds = {
            "train": train_bounds,
            "val": (start, length),
            "test": (start, length),
            "test_time": tail_bounds,
        }
        members = {
            state: [(identifier, reads[identifier], bounds[state]) for identifier, _ in group]
            for state, group in chosen.items()
        }
        # The in-distribution test set is the tail of the training realisations themselves.
        members["test_time"] = [
            (identifier, reads[identifier], tail_bounds) for identifier, _ in chosen["train"]
        ]
        sources = [
            (path, reads[identifier]) for group in chosen.values() for identifier, path in group
        ]
    else:
        start = _spin_up_start(run["times"], t_start, skip)
        bounds = {
            state: (start + a, start + b)
            for state, (a, b) in split_along_time(len(run["times"]) - start, gap).items()
        }
        # The time split draws every one of its splits from the one realisation, numbered 0.
        members = {state: [(0, run, bound)] for state, bound in bounds.items()}
        extras = [read(path) for path in runs[1:]]
        if extras:
            members["test_extra"] = [
                (index, extra, (start, len(extra["times"])))
                for index, extra in enumerate(extras, start=1)
            ]
        sources = [(runs[0], run), *zip(runs[1:], extras, strict=True)]

    summary = {"datasets": {}, "loader_dirs": {}, "normalisers": {}, "shapes": {}, "rms": {}}
    for field in fields:
        channels = CHANNELS[field]
        arrays = {}
        for state, group in members.items():
            arrays.update(_split_arrays(group, channels, field, state))
        arrays.update(
            _metadata(
                run,
                sources,
                resolved,
                field,
                channels,
                (dt, dt_deviation),
                bounds,
                forcing_check,
                {
                    "gap": gap,
                    "split_mode": split,
                    "decorrelation_frames": decorrelation_frames,
                    "time_test_fraction": time_test_fraction if chosen else None,
                },
            )
        )
        summary["shapes"][field] = {
            state.removeprefix("X_"): arrays[state].shape
            for state in arrays
            if state.startswith("X_")
        }
        path = out_dir / f"{prefix}_{field}.npz"
        np.savez(path, **arrays)
        summary["datasets"][field] = path
        summary["loader_dirs"][field] = loader_dir(path)
        stats, rms = _train_normaliser(arrays["X_train"], VELOCITY_CHANNELS[field])
        summary["normalisers"][field] = out_dir / f"{prefix}_{field}_normstat.pt"
        torch.save(stats, summary["normalisers"][field])
        summary["rms"][field] = dict(zip(channels, rms, strict=True))

    summary["diagnostics"] = write_diagnostics(
        run,
        bounds,
        (dt, dt_deviation),
        forcing_check,
        out_dir,
        prefix,
        n_coarse,
        advice,
        split_mode=split,
        realisations={s: [i for i, _, _ in group] for s, group in members.items()}
        if chosen
        else None,
    )
    summary["stride_recommendation"] = advice
    return summary


def _snapshot_interval(times, tol=0.1):
    """Return the snapshot interval and its jitter, refusing a grossly uneven cadence.

    Dedalus writes at the first timestep past each target time, so a snapshot interval carries
    a jitter of one timestep and the pairs of a run do not share an identical `dt`. The jitter
    is measured and recorded rather than hidden; a run whose cadence wanders further than
    `tol` is refused, because one-step pairs would then mean different things.

    Args:
        times: Simulation times of the snapshots, in order.
        tol: Largest fraction of `dt` an interval may deviate by.

    Returns:
        Tuple `(dt, deviation)` of the median interval and the largest relative deviation.

    Raises:
        ValueError: If the cadence wanders further than `tol`.
    """
    steps = np.diff(times)
    dt = float(np.median(steps))
    deviation = float(np.abs(steps - dt).max() / dt)
    if deviation > tol:
        raise ValueError(
            f"snapshot cadence is not uniform: dt = {dt:.6g} with a maximum deviation of "
            f"{deviation:.3e}, over the tolerance {tol:.3e}; one-step pairs would not share "
            f"a time step"
        )
    return dt, deviation


def _split_arrays(members, channels, field, state):
    """Stack the pairs of every realisation of one split, realisation-major.

    Pairs are formed inside a realisation and the realisations are then concatenated, so no
    pair straddles two of them and a rollout of `T` steps stays inside one realisation as
    long as `T` divides the pairs each contributes.

    Args:
        members: The `(id, run, (start, stop))` of every realisation in the split, in the
            order they are to be stored.
        channels: Channel names to stack.
        field: `velocity` or `vorticity`; the velocity datasets carry `psi` as well.
        state: Name of the split, which every key written is suffixed with.

    Returns:
        Dict of the `X`, `y`, `times`, `realisation_ids` and `psi` arrays of the split, plus
        the ids of the realisations it draws on.
    """
    parts = []
    for identifier, run, (start, stop) in members:
        block = _segment(run, channels, start, stop)
        part = {
            f"X_{state}": block[:-1],
            f"y_{state}": block[1:],
            f"times_{state}": _pair_times(run["times"], start, stop),
            f"realisation_ids_{state}": np.full(stop - start - 1, identifier, dtype=np.int32),
        }
        part.update(_psi_pair(run, field, state, start, stop))
        parts.append(part)
    arrays = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    arrays[f"{state}_realisations"] = np.asarray([i for i, _, _ in members], dtype=np.int32)
    return arrays


def time_test_tail(start, n_snapshots, gap, fraction):
    """Cut the last `fraction` of a training realisation off as an in-distribution test set.

    The realisation split holds out whole realisations, so its test set differs from training
    in initial condition as well as in time. `test_time` is the complement of that: the tail
    of a training realisation, the same trajectory the model was trained on, `gap` snapshots
    later. Comparing the two separates the error of predicting a later time from the error of
    predicting another realisation of the same flow.

    Args:
        start: First snapshot index after the spin-up.
        n_snapshots: Snapshots the realisation holds.
        gap: Snapshots discarded between the training frames and the tail.
        fraction: Fraction of the post-spin-up snapshots the tail takes.

    Returns:
        Tuple of the `(start, stop)` index range of the training frames and of the tail.

    Raises:
        ValueError: If the tail and the gap leave the training frames no pair.
    """
    usable = n_snapshots - start
    # Two snapshots is one pair, so a tail shorter than that would carry no sample at all.
    tail_start = n_snapshots - max(2, round(fraction * usable))
    train_stop = tail_start - gap
    if train_stop - start < 2:
        raise ValueError(
            f"realisation too short: {usable} snapshots after the spin-up leave "
            f"{train_stop - start} for training once the gap of {gap} and the last "
            f"{fraction:.0%} are held out"
        )
    return (start, train_stop), (tail_start, n_snapshots)


def _spin_up_start(times, t_start, skip):
    """First snapshot index the dataset uses, from a simulation time or a snapshot count."""
    return int(np.searchsorted(times, t_start)) if t_start is not None else int(skip)


def _train_normaliser(x_train, velocity_channels):
    """Build the `NSLoader2D` normaliser from the training inputs only."""
    train = np.asarray(x_train, dtype=np.float64)
    mean = train.mean(axis=(0, 2, 3))
    mean_square = (train**2).mean(axis=(0, 2, 3))
    return normaliser_stats(mean, mean_square - mean**2, velocity_channels), np.sqrt(mean_square)


def _resolve_physics(config, physics, run):
    """Merge the run's own provenance with the command-line overrides into one record.

    Args:
        config: The parsed `run_config.jsonl` record, possibly empty.
        physics: Overrides for `nu`, `alpha`, `kmin` and `kmax`; any key present here wins.
        run: The dict `read_run` returned, whose h5 attributes may carry the git commit.

    Returns:
        Dict with `nu`, `alpha`, the forcing `band` and `forcing_type`, and `git_commit`.
    """
    args = config.get("args", config)
    return {
        "nu": _number(physics.get("nu"), args.get("nu")),
        "alpha": _number(physics.get("alpha"), args.get("alpha")),
        "band": forcing_band(args, physics),
        "forcing_type": args.get("forcing", "unknown"),
        "git_commit": str(run["attrs"].get("git_commit", config.get("git", {}).get("commit", ""))),
    }


def _metadata(run, sources, physics, field, channels, cadence, bounds, forcing_check, build):
    """Assemble the metadata arrays stored alongside the pairs.

    Args:
        run: The realisation the grid, the cut-off and the cadence are taken from; under the
            `realisation` split that is the first training realisation.
        sources: The `(directory, run)` of every realisation the datasets draw on.
        physics: The record `_resolve_physics` returned.
        field: `velocity` or `vorticity`.
        channels: The channels of that field, in order.
        cadence: Tuple `(dt, deviation)` of the snapshot interval and its largest jitter.
        bounds: Snapshot index ranges per split, within one realisation.
        forcing_check: The dict `check_forcing_band` returned.
        build: The `gap`, the `split_mode`, the `decorrelation_frames` and the
            `time_test_fraction` of the build.

    Returns:
        Dict of the metadata arrays.
    """
    dt, dt_deviation = cadence
    band = physics["band"]
    times = run["times"]
    starts = np.array([a for a, _ in bounds.values()])
    stops = np.array([b for _, b in bounds.values()])
    metadata = {
        "field": np.asarray(field.encode()),
        "channels": np.asarray([c.encode() for c in channels]),
        "nu": np.asarray(physics["nu"]),
        "alpha": np.asarray(physics["alpha"]),
        "dt": np.asarray(dt),
        "dt_max_deviation": np.asarray(dt_deviation),
        "t_start": np.asarray(times[starts.min()]),
        "t_end": np.asarray(times[stops.max() - 1]),
        "domain_length": np.asarray(run["domain_length"]),
        "cutoff": np.asarray(len(run["x"])),
        "cutoff_rule": np.asarray(run["cutoff"].encode()),
        "cutoff_kmax": np.asarray(cutoff_kmax(len(run["x"]), run["cutoff"])),
        "n_fine": np.asarray(run["n_fine"]),
        "handler": np.asarray(run["handler"].encode()),
        "has_pressure": np.asarray(run["has_pressure"]),
        "gap": np.asarray(build["gap"]),
        "skip": np.asarray(starts.min()),
        "stride": np.asarray(run["stride"]),
        "split_mode": np.asarray(build["split_mode"].encode()),
        "decorrelation_frames": np.asarray(_number(build["decorrelation_frames"])),
        "time_test_fraction": np.asarray(_number(build["time_test_fraction"])),
        "forcing": np.asarray(f"{physics['forcing_type']} in [{band[0]}, {band[1]}]".encode()),
        "forcing_kmin": np.asarray(_number(band[0])),
        "forcing_kmax": np.asarray(_number(band[1])),
        "forcing_retained_fraction": np.asarray(_number(forcing_check["retained_fraction"])),
        "forcing_rms_relative_to_velocity": np.asarray(
            _number(forcing_check["rms_relative_to_velocity"])
        ),
        "axis_order": np.asarray(AXIS_ORDER),
        "sign_convention": np.asarray(SIGN_CONVENTION),
        "source": np.asarray(str(sources[0][0]).encode()),
        "source_realisations": np.asarray([str(path).encode() for path, _ in sources]),
        "source_git_commit": np.asarray(physics["git_commit"].encode()),
        "source_files": np.asarray([f.encode() for _, other in sources for f in other["files"]]),
        "split_bounds": np.stack([starts, stops], axis=1),
        "split_states": np.asarray([state.encode() for state in bounds]),
        "x_coords": run["x"],
        "y_coords": run["y"],
    }
    return metadata


def _number(*candidates):
    """First candidate that is not `None`, as a float, or `nan` if there is none."""
    for candidate in candidates:
        if candidate is not None:
            return float(candidate)
    return float("nan")


def _mean_square(field, block=1024):
    """Snapshot-by-snapshot spatial mean square, in blocks so a long run stays in memory."""
    out = np.empty(len(field))
    for start in range(0, len(field), block):
        chunk = np.asarray(field[start : start + block], dtype=np.float64)
        out[start : start + block] = np.mean(chunk**2, axis=(-2, -1))
    return out


def write_diagnostics(
    run,
    bounds,
    cadence,
    forcing_check,
    out_dir,
    prefix,
    n_coarse,
    advice=None,
    split_mode="time",
    realisations=None,
    points=2000,
):
    """Write the energy and enstrophy history, the split boundaries and the spectra.

    Args:
        run: The dict `read_run` returned.
        bounds: Snapshot index ranges per split.
        cadence: Tuple `(dt, deviation)` of the snapshot interval and its largest jitter.
        forcing_check: The dict `check_forcing_band` returned.
        advice: The dict `recommend_stride` returned, or `None`.
        out_dir: Directory to write into.
        prefix: Stem of the files written.
        n_coarse: Side of the coarse grid.
        split_mode: `time` or `realisation`.
        realisations: The realisation ids of each split, under the `realisation` split. An
            index range then names the same window in each realisation of that split, and the
            times reported for it are those of the realisation `run` holds.
        points: Largest number of samples of the time series kept in the JSON.

    Returns:
        Dict with the `json` and `png` paths.
    """
    dt, dt_deviation = cadence
    times = run["times"]
    energy = 0.5 * (_mean_square(run["u"]) + _mean_square(run["v"]))
    enstrophy = 0.5 * _mean_square(run["omega"])
    stride = max(1, len(times) // points)
    spectra = run["spectra"]
    counts = {state: len(realisations[state]) for state in bounds} if realisations else {}
    record = {
        "handler": run["handler"],
        "n_fine": run["n_fine"],
        "n_coarse": n_coarse,
        "cutoff_rule": run["cutoff"],
        "cutoff_kmax": cutoff_kmax(n_coarse, run["cutoff"]),
        "dt": dt,
        "dt_max_deviation": dt_deviation,
        "stride": run["stride"],
        "stride_recommendation": advice,
        "n_snapshots": len(times),
        "domain_length": run["domain_length"],
        "conventions": run["conventions"],
        "has_pressure": run["has_pressure"],
        "forcing": forcing_check,
        "split_mode": split_mode,
        "splits": {
            state: {
                "index": [int(a), int(b)],
                "time": [float(times[a]), float(times[b - 1])],
                "pairs": int(b - a - 1) * counts.get(state, 1),
                **({"realisations": realisations[state]} if realisations else {}),
            }
            for state, (a, b) in bounds.items()
        },
        "series": {
            "time": times[::stride].tolist(),
            "energy": energy[::stride].tolist(),
            "enstrophy": enstrophy[::stride].tolist(),
        },
        "spectra": None
        if spectra is None
        else {
            "k": spectra["k"].tolist(),
            "E_fine": spectra["E_fine"].tolist(),
            "E_coarse": spectra["E_coarse"].tolist(),
            "samples": spectra["samples"],
        },
    }
    json_path = out_dir / f"{prefix}_diagnostics.json"
    json_path.write_text(json.dumps(record, indent=2))
    k_cut = cutoff_kmax(n_coarse, run["cutoff"]) * 2 * np.pi / run["domain_length"]
    png_path = _plot_diagnostics(times, energy, enstrophy, bounds, spectra, k_cut, out_dir, prefix)
    return {"json": json_path, "png": png_path}


def _plot_diagnostics(times, energy, enstrophy, split, spectra, k_cut, out_dir, prefix):
    """Plot the energy and enstrophy history with the split boundaries, and the spectra."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(times, energy, label="energy", color="tab:blue")
    twin = axes[0].twinx()
    twin.plot(times, enstrophy, label="enstrophy", color="tab:red")
    twin.set_ylabel("enstrophy")
    for state, (a, b) in split.items():
        axes[0].axvspan(times[a], times[b - 1], alpha=0.12, color="tab:green")
        axes[0].text(times[a], axes[0].get_ylim()[1], state, fontsize=8, va="top")
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("energy")
    axes[0].set_title("history and splits (shaded)")
    if spectra is not None:
        # A truncated mode returns from the transform pair as round-off dust rather than an
        # exact zero, so the floor keeps it off a logarithmic axis.
        floor = spectra["E_fine"].max() * 1e-16
        k = spectra["k"]
        for name, style, label in (
            ("E_fine", "-", "before truncation"),
            ("E_coarse", "--", "after"),
        ):
            keep = (k > 0) & (spectra[name] > floor)
            axes[1].loglog(k[keep], spectra[name][keep], style, label=label)
        axes[1].axvline(k_cut, ls=":", color="grey", label="cut-off")
        axes[1].set_xlabel("k")
        axes[1].set_ylabel("E(k)")
        axes[1].set_title(f"mean energy spectrum ({spectra['samples']} snapshots)")
        axes[1].legend()
    figure.tight_layout()
    png_path = out_dir / f"{prefix}_diagnostics.png"
    figure.savefig(png_path, dpi=120)
    plt.close(figure)
    return png_path


def _id_list(text):
    """Parse a comma-separated list of realisation ids."""
    return [int(part) for part in text.split(",") if part.strip()]


OUT_DIR = "DATA_ROOT/ns2d"


def main(argv=None):
    """Run the conversion from the command line and print a summary of what was written."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--run-dir",
        required=True,
        action="append",
        help="run or realisation directory; repeat it once per realisation, which is how the "
        "E3 production run lays its seeds out",
    )
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--prefix", default="ns2d_e3")
    parser.add_argument("--n-coarse", type=int, default=64)
    parser.add_argument("--field", choices=sorted(CHANNELS), action="append", default=None)
    parser.add_argument(
        "--handler", default=None, help="coarse (in-situ) or snapshots (offline truncation)"
    )
    parser.add_argument(
        "--cutoff",
        choices=("sharp", "two-thirds"),
        default="sharp",
        help="retain |kx|, |ky| < n_coarse / 2, or NS2D's rectangular dealiasing band",
    )
    parser.add_argument(
        "--gap", type=int, default=20, help="snapshots discarded between the splits"
    )
    parser.add_argument("--t-start", type=float, default=None, help="skip the spin-up by time")
    parser.add_argument("--skip", type=int, default=0, help="skip the spin-up by snapshot count")
    parser.add_argument(
        "--stride", type=int, default=1, help="keep every stride-th snapshot in time"
    )
    parser.add_argument(
        "--split",
        choices=("time", "realisation"),
        default="time",
        help="split one realisation along time, or give each split its own realisations",
    )
    for state in ("train", "val", "test"):
        parser.add_argument(
            f"--{state}-realisations",
            type=_id_list,
            default=None,
            help=f"comma-separated realisation ids of the {state} split, under --split realisation",
        )
    parser.add_argument(
        "--time-test-fraction",
        type=float,
        default=0.15,
        help="fraction of each training realisation held out behind the gap as test_time",
    )
    parser.add_argument(
        "--decorrelation-frames",
        type=int,
        default=None,
        help="the decorrelation time in snapshots, recorded in the metadata only",
    )
    parser.add_argument("--chunk", type=int, default=32)
    parser.add_argument(
        "--dt-tolerance",
        type=float,
        default=0.1,
        help="largest relative jitter accepted in the snapshot cadence",
    )
    for name in ("nu", "alpha", "kmin", "kmax"):
        parser.add_argument(f"--{name}", type=float, default=None, help="overrides run_config")
    args = parser.parse_args(argv)

    summary = build_dataset(
        args.run_dir,
        args.out_dir,
        prefix=args.prefix,
        n_coarse=args.n_coarse,
        fields=tuple(args.field or sorted(CHANNELS)),
        gap=args.gap,
        t_start=args.t_start,
        skip=args.skip,
        stride=args.stride,
        handler=args.handler,
        cutoff=args.cutoff,
        physics={n: getattr(args, n) for n in ("nu", "alpha", "kmin", "kmax")},
        chunk=args.chunk,
        dt_tolerance=args.dt_tolerance,
        split=args.split,
        realisations={s: getattr(args, f"{s}_realisations") for s in ("train", "val", "test")},
        time_test_fraction=args.time_test_fraction,
        decorrelation_frames=args.decorrelation_frames,
    )
    for field, path in summary["datasets"].items():
        print(f"[{field}] wrote {path}")
        print(f"[{field}] loader dir {summary['loader_dirs'][field]}")
        print(f"[{field}] normaliser {summary['normalisers'][field]}")
        for state, shape in summary["shapes"][field].items():
            print(f"[{field}]   {state}: X/y {shape}")
        for name, value in summary["rms"][field].items():
            print(f"[{field}]   {name}: train RMS {value:.6e}")
    advice = summary["stride_recommendation"]
    if advice is not None:
        print(
            f"stride: {advice['stride']} would give a pair interval of one turnover time "
            f"({advice['tau_min']:.4g} at k = {advice['k_at_tau_min']:.0f}, from "
            f"{advice['snapshots']} spectra over k in {advice['k_range']})"
        )
    print(f"diagnostics {summary['diagnostics']['json']}")
    print(f"diagnostics {summary['diagnostics']['png']}")
    return summary


if __name__ == "__main__":
    main()
