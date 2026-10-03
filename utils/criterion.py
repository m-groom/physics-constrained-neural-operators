import math
import math as mt
import os
from dataclasses import dataclass
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch_harmonics as th
from einops import rearrange
from scipy import stats
from scipy.optimize import fsolve, minimize_scalar
from torch import nn
from torch.nn.modules.loss import _WeightedLoss

from utils.compute_diagnostics import streamfunction_to_velocity
from utils.compute_physical_statistics import compute_spectra

FORCING_TYPES = ("kolmogorov_diag", "pino_cos4y", "none")
# Parameters each forcing type needs; like the rest of the physics they must be given.
REQUIRED_FORCING_KEYS = {
    "kolmogorov_diag": ("amplitude", "wavenumber", "phase"),
    "pino_cos4y": ("amplitude", "wavenumber"),
    "none": (),
}
FORMULATIONS = ("velocity", "vorticity")
_REQUIRED_PHYSICS_KEYS = (
    "nu",
    "alpha",
    "dt",
    "domain_length",
    "nx",
    "ny",
    "formulation",
    "forcing",
)


@dataclass(frozen=True)
class Physics:
    """The physical setup of one experiment, read from the YAML ``data:`` block.

    Attributes:
        nu: Kinematic viscosity.
        alpha: Linear (Rayleigh) friction coefficient.
        dt: Time step between consecutive snapshots of the dataset.
        domain_length: Side length of the square periodic domain.
        nx: Grid points in x.
        ny: Grid points in y.
        formulation: ``"velocity"`` (channels ux, uy, p) or ``"vorticity"`` (channel w).
        forcing: Forcing block; ``type`` is one of :data:`FORCING_TYPES`, plus the
            parameters :data:`REQUIRED_FORCING_KEYS` lists for that type.
        residual_upsample: Refinement factor for the vorticity residual (PINO's recipe).
    """

    nu: float
    alpha: float
    dt: float
    domain_length: float
    nx: int
    ny: int
    formulation: str
    forcing: dict
    residual_upsample: int


def physics_from_config(data_config):
    """Build a :class:`Physics` from the ``data:`` block of a run configuration.

    Every physical parameter of a run lives in one place, so the residual, the
    forcing and the projectors cannot drift apart.  Missing or invalid entries
    raise instead of falling back to a default.

    Args:
        data_config: The ``data`` mapping of the loaded YAML configuration.

    Returns:
        Physics: The validated physical setup.

    Raises:
        KeyError: A required key is missing from ``config['data']``.
        ValueError: A key is present but holds an unsupported value.
    """
    for key in _REQUIRED_PHYSICS_KEYS:
        if key not in data_config:
            raise KeyError(
                f"config['data'] is missing the required physics key '{key}'; "
                f"expected all of {_REQUIRED_PHYSICS_KEYS}"
            )

    formulation = data_config["formulation"]
    if formulation not in FORMULATIONS:
        raise ValueError(
            f"config['data']['formulation'] must be one of {FORMULATIONS}, got {formulation!r}"
        )

    forcing = data_config["forcing"]
    if not isinstance(forcing, dict) or "type" not in forcing:
        raise ValueError("config['data']['forcing'] must be a mapping with a 'type' entry")
    if forcing["type"] not in FORCING_TYPES:
        raise ValueError(
            f"config['data']['forcing']['type'] must be one of {FORCING_TYPES}, "
            f"got {forcing['type']!r}"
        )
    for key in REQUIRED_FORCING_KEYS[forcing["type"]]:
        if key not in forcing:
            raise KeyError(
                f"config['data']['forcing'] of type '{forcing['type']}' is missing '{key}'; "
                f"expected all of {REQUIRED_FORCING_KEYS[forcing['type']]}"
            )

    residual_upsample = int(data_config.get("residual_upsample", 4))
    if residual_upsample < 1:
        raise ValueError("config['data']['residual_upsample'] must be a positive integer")

    return Physics(
        nu=float(data_config["nu"]),
        alpha=float(data_config["alpha"]),
        dt=float(data_config["dt"]),
        domain_length=float(data_config["domain_length"]),
        nx=int(data_config["nx"]),
        ny=int(data_config["ny"]),
        formulation=formulation,
        forcing=dict(forcing),
        residual_upsample=residual_upsample,
    )


def state_channels(physics):
    """Number of state channels the formulation carries: (ux, uy, p), or (w) alone."""
    return 1 if physics.formulation == "vorticity" else 3


def carries_velocity(physics):
    """Whether the state carries a velocity vector, so velocity diagnostics apply.

    The divergence, the domain-mean velocity and the two momentum residuals are
    defined on (ux, uy); the vorticity formulation carries one channel and has
    none of them.
    """
    return physics.formulation == "velocity"


def residual_has_scale(physics, forcing):
    """Whether the configured residual has a denominator to be reported against.

    The vorticity residual follows PINO in scoring the residual relative to the
    forcing, so an unforced vorticity run has nothing to divide by: ``PINO_loss3d``
    refuses it rather than returning a meaningless number. The velocity residuals are
    normalised by truth statistics instead (#16), so they survive an unforced run --
    which is E3's own case, where the coarse field carries no forcing at all.

    The test is made on the forcing field rather than on ``physics.forcing["type"]``,
    so that it is the same question ``PINO_loss3d`` asks and cannot drift from it: a
    named forcing of zero amplitude is unforced too.

    Args:
        physics: The run's :class:`Physics`.
        forcing: The forcing field the residual would be scored against.

    Returns:
        bool: False only for a vorticity run whose forcing is identically zero.

    Callers that compute the residual as a *diagnostic* consult this first and report
    the metric as missing when it is false. Callers that compute it as a *loss* do not:
    a configuration that weights a residual it cannot define should still fail loudly.
    """
    return physics.formulation == "velocity" or bool(torch.any(forcing))


def check_model_channels(physics, out_dim):
    """Raise unless a model's output width matches the state the formulation carries.

    Args:
        physics: The run's :class:`Physics`.
        out_dim: Number of output channels the model is configured with.
    """
    if int(out_dim) != state_channels(physics):
        raise ValueError(
            f"the {physics.formulation} formulation needs model.out_dim = "
            f"{state_channels(physics)}, got {out_dim}"
        )


def _residual_wavenumbers(nx, ny, domain_length, device, dtype):
    """Wavenumber grids for the residual layout ``(batch, nx, ny, nt)``.

    Args:
        nx: Grid points along x (axis 1).
        ny: Grid points along y (axis 2).
        domain_length: Side length of the square periodic domain.
        device: Torch device of the field.
        dtype: Real dtype of the field.

    Returns:
        tuple: ``(k_x, k_y)``, each of shape ``(1, nx, ny, 1)`` in radians per
        unit length.
    """
    scale = 2.0 * math.pi / domain_length
    k_x = scale * torch.fft.fftfreq(nx, d=1.0 / nx, device=device, dtype=dtype)
    k_y = scale * torch.fft.fftfreq(ny, d=1.0 / ny, device=device, dtype=dtype)
    return k_x.reshape(1, nx, 1, 1), k_y.reshape(1, 1, ny, 1)


def fourier_upsample_2d(w, factor):
    """Refine a periodic field by zero-padding its spectrum (PINO's residual recipe).

    The Nyquist row and column are dropped: on the coarse grid they carry a
    cosine whose refinement is ambiguous, and keeping them would inject a
    spurious complex mode.

    Args:
        w: Real field of shape ``(batch, nx, ny, nt)``.
        factor: Integer refinement factor; ``1`` returns ``w`` unchanged.

    Returns:
        torch.Tensor: Field of shape ``(batch, factor*nx, factor*ny, nt)``.
    """
    if factor == 1:
        return w
    if factor < 1:
        raise ValueError(f"factor must be a positive integer, got {factor}")

    batchsize, nx, ny, nt = w.shape
    if nx < 4 or ny < 4:
        raise ValueError(f"fourier_upsample_2d needs at least 4 points per side, got {nx}x{ny}")
    if nx % 2 or ny % 2:
        raise ValueError(f"fourier_upsample_2d needs an even number of points, got {nx}x{ny}")
    kx, ky = nx // 2, ny // 2
    w_h = torch.fft.fft2(w, dim=[1, 2])
    out_h = torch.zeros(batchsize, factor * nx, factor * ny, nt, dtype=w_h.dtype, device=w_h.device)
    # Four corner blocks: (positive, negative) x (positive, negative) wavenumbers.
    out_h[:, :kx, :ky] = w_h[:, :kx, :ky]
    out_h[:, :kx, -(ky - 1) :] = w_h[:, :kx, -(ky - 1) :]
    out_h[:, -(kx - 1) :, :ky] = w_h[:, -(kx - 1) :, :ky]
    out_h[:, -(kx - 1) :, -(ky - 1) :] = w_h[:, -(kx - 1) :, -(ky - 1) :]
    return torch.fft.ifft2(out_h * factor**2, dim=[1, 2]).real


def velocity_from_vorticity(w, domain_length):
    """Velocity of the periodic flow whose vorticity is ``w``.

    The stream function is recovered spectrally from ``-lap(psi) = w`` and the
    velocity from ``u = grad_perp(psi)``, so ``curl(u) = w`` by construction.

    Args:
        w: Vorticity of shape ``(batch, nx, ny, nt)``.
        domain_length: Side length of the square periodic domain.

    Returns:
        tuple: ``(ux, uy)``, each of the same shape as ``w``.
    """
    _, nx, ny, _ = w.shape
    k_x, k_y = _residual_wavenumbers(nx, ny, domain_length, w.device, w.dtype)
    lap = k_x**2 + k_y**2
    # Invert -lap on the non-constant modes; the k=0 mode of psi is arbitrary.
    inv_lap = torch.zeros_like(lap)
    inv_lap[lap > 0] = 1.0 / lap[lap > 0]
    psi_h = torch.fft.fft2(w, dim=[1, 2]) * inv_lap
    ux = torch.fft.ifft2(1j * k_y * psi_h, dim=[1, 2]).real
    uy = torch.fft.ifft2(-1j * k_x * psi_h, dim=[1, 2]).real
    return ux, uy


def _NS_vorticity_terms(w, nu, alpha, domain_length):
    """Spatial part of the vorticity residual, ``u.grad(w) - nu*lap(w) + alpha*w``.

    The stream function is recovered spectrally from ``-lap(psi) = w`` and the
    velocity from ``u = grad_perp(psi)``, so ``curl(u) = w`` by construction.

    Args:
        w: Vorticity of shape ``(batch, nx, ny, nt)``.
        nu: Kinematic viscosity.
        alpha: Linear friction coefficient.
        domain_length: Side length of the square periodic domain.

    Returns:
        torch.Tensor: Same shape as ``w``.
    """
    batchsize, nx, ny, nt = w.shape
    k_x, k_y = _residual_wavenumbers(nx, ny, domain_length, w.device, w.dtype)

    w_h = torch.fft.fft2(w, dim=[1, 2])
    lap = k_x**2 + k_y**2

    ux, uy = velocity_from_vorticity(w, domain_length)
    wx = torch.fft.ifft2(1j * k_x * w_h, dim=[1, 2]).real
    wy = torch.fft.ifft2(1j * k_y * w_h, dim=[1, 2]).real
    wlap = torch.fft.ifft2(-lap * w_h, dim=[1, 2]).real

    return ux * wx + uy * wy - nu * wlap + alpha * w


def FDM_NS_vorticity(w, nu, alpha, t_interval, domain_length):
    """Vorticity residual with central differences in time (PINO's recipe).

    The equation is ``dw/dt + u.grad(w) - nu*lap(w) + alpha*w = f``, so the
    return value is to be compared against the forcing.

    Args:
        w: Vorticity of shape ``(batch, nx, ny, nt)``, ``nt >= 3``.
        nu: Kinematic viscosity.
        alpha: Linear friction coefficient.
        t_interval: Total time spanned by the ``nt`` snapshots.
        domain_length: Side length of the square periodic domain.

    Returns:
        torch.Tensor: Residual of shape ``(batch, nx, ny, nt-2)``.
    """
    nt = w.size(3)
    if nt < 3:
        raise ValueError(f"FDM_NS_vorticity needs at least 3 time levels, got {nt}")
    dt = t_interval / (nt - 1)
    wt = (w[:, :, :, 2:] - w[:, :, :, :-2]) / (2 * dt)
    return wt + _NS_vorticity_terms(w, nu, alpha, domain_length)[..., 1:-1]


def INT_NS_vorticity(w, nu, alpha, t_interval, domain_length):
    """Vorticity residual in integral (trapezoidal) form in time.

    Same equation as :func:`FDM_NS_vorticity`, integrated between consecutive
    snapshots, so it is defined for a single pair of time levels.  The return
    value is to be compared against ``dt * f``.

    Args:
        w: Vorticity of shape ``(batch, nx, ny, nt)``, ``nt >= 2``.
        nu: Kinematic viscosity.
        alpha: Linear friction coefficient.
        t_interval: Total time spanned by the ``nt`` snapshots.
        domain_length: Side length of the square periodic domain.

    Returns:
        torch.Tensor: Residual of shape ``(batch, nx, ny, nt-1)``.
    """
    nt = w.size(3)
    if nt < 2:
        raise ValueError(f"INT_NS_vorticity needs at least 2 time levels, got {nt}")
    dt = t_interval / (nt - 1)
    terms = _NS_vorticity_terms(w, nu, alpha, domain_length)
    delta_w = w[..., 1:] - w[..., :-1]
    return delta_w + dt * 0.5 * (terms[..., :-1] + terms[..., 1:])


def guess_dt(w, v, f):
    batchsize = w.size(0)
    nx = w.size(1)
    ny = w.size(2)
    nt = w.size(3)
    device = w.device
    w = w.reshape(batchsize, nx, ny, nt)

    w_h = torch.fft.fft2(w, dim=[1, 2])
    # Wavenumbers in y-direction
    k_max = nx // 2
    N = nx
    k_x = (
        torch.cat(
            (
                torch.arange(start=0, end=k_max, step=1, device=device),
                torch.arange(start=-k_max, end=0, step=1, device=device),
            ),
            0,
        )
        .reshape(N, 1)
        .repeat(1, N)
        .reshape(1, N, N, 1)
    )
    k_y = (
        torch.cat(
            (
                torch.arange(start=0, end=k_max, step=1, device=device),
                torch.arange(start=-k_max, end=0, step=1, device=device),
            ),
            0,
        )
        .reshape(1, N)
        .repeat(N, 1)
        .reshape(1, N, N, 1)
    )
    # Negative Laplacian in Fourier space
    lap = k_x**2 + k_y**2
    lap[0, 0, 0, 0] = 1.0
    f_h = w_h / lap

    ux_h = 1j * k_y * f_h
    uy_h = -1j * k_x * f_h
    wx_h = 1j * k_x * w_h
    wy_h = 1j * k_y * w_h
    wlap_h = -lap * w_h

    ux = torch.fft.irfft2(ux_h[:, :, : k_max + 1], dim=[1, 2])
    uy = torch.fft.irfft2(uy_h[:, :, : k_max + 1], dim=[1, 2])
    wx = torch.fft.irfft2(wx_h[:, :, : k_max + 1], dim=[1, 2])
    wy = torch.fft.irfft2(wy_h[:, :, : k_max + 1], dim=[1, 2])
    wlap = torch.fft.irfft2(wlap_h[:, :, : k_max + 1], dim=[1, 2])

    residual = (
        f.repeat(batchsize, 1, 1, nt - 2) - (ux * wx + uy * wy - v * wlap)[..., 1:-1]
    )  # - forcing
    return residual


def PINO_loss3d(
    u,
    u0,
    forcing,
    nu,
    alpha,
    t_interval,
    domain_length,
    time_method="integral",
    upsample=1,
):
    """Physics-informed loss for the vorticity formulation.

    Following PINO, the residual is evaluated on a grid refined ``upsample``
    times by zero-padding the spectrum of the vorticity; the forcing must
    already be given on that refined grid.

    Args:
        u: Predicted vorticity, shape ``(batch, nx, ny, nt)``.
        u0: Initial condition, shape ``(batch, nx, ny)`` or broadcastable.
        forcing: Vorticity forcing, shape ``(1, upsample*nx, upsample*ny, 1)``.
        nu: Kinematic viscosity.
        alpha: Linear friction coefficient.
        t_interval: Total time spanned by the ``nt`` snapshots.
        domain_length: Side length of the square periodic domain.
        time_method: ``"integral"`` (trapezoidal, ``nt >= 2``) or ``"fdm"``
            (central differences, ``nt >= 3``).
        upsample: Refinement factor for the residual grid.

    Returns:
        tuple: ``(loss_ic, loss_f)`` — relative L2 error on the initial
        condition, and relative L2 of the residual against the forcing.
    """
    batchsize = u.size(0)
    nx = u.size(1)
    ny = u.size(2)
    nt = u.size(3)

    u = u.reshape(batchsize, nx, ny, nt)
    lploss = LpLoss(size_average=True)
    # loss_ic should always be 0 because we give the initial condition as the input
    u_in = u[:, :, :, 0]
    loss_ic = lploss(u_in, u0)

    # PDE loss, on the refined grid
    u_fine = fourier_upsample_2d(u, upsample)
    expected = (1, upsample * nx, upsample * ny, 1)
    if tuple(forcing.shape) != expected:
        raise ValueError(
            f"vorticity forcing must have shape {expected} to match the residual grid, "
            f"got {tuple(forcing.shape)}"
        )
    if not torch.any(forcing):
        # The loss is the residual relative to the forcing, so a zero forcing has no
        # scale to divide by; score an unforced run on the residual itself instead.
        raise ValueError(
            "PINO_loss3d needs a non-zero forcing; use FDM_NS_vorticity or "
            "INT_NS_vorticity directly for an unforced run"
        )

    if time_method == "fdm":
        Du = FDM_NS_vorticity(u_fine, nu, alpha, t_interval, domain_length)
        f = forcing.repeat(batchsize, 1, 1, nt - 2)
    elif time_method == "integral":
        Du = INT_NS_vorticity(u_fine, nu, alpha, t_interval, domain_length)
        f = (t_interval / (nt - 1)) * forcing.repeat(batchsize, 1, 1, nt - 1)
    else:
        raise ValueError('time_method must be "integral" or "fdm"')
    loss_f = lploss(Du, f)

    return loss_ic, loss_f


def _forcing_grid(S, domain_length, dtype):
    """Coordinate grids ``(x, y)``, each ``(S, S)``, on a square periodic domain."""
    axis = torch.tensor(np.linspace(0, domain_length, S, endpoint=False), dtype=dtype)
    return axis.reshape(S, 1).repeat(1, S), axis.reshape(1, S).repeat(S, 1)


def get_forcing(S, amplitude, wavenumber, domain_length, dtype=torch.float32):
    """Return the PINO-paper Kolmogorov forcing in vorticity form.

    The curl of the velocity forcing ``f = (amplitude*sin(k*y), 0)`` is
    ``-amplitude*k*cos(k*y)``, which is the paper's ``-4*cos(4*x2)`` for
    ``amplitude = 1`` and ``k = 4``.

    Args:
        S: Spatial grid resolution (points per side).
        amplitude: Amplitude of the equivalent velocity forcing.
        wavenumber: Wavenumber of the forcing.
        domain_length: Side length of the square periodic domain.
        dtype: Real dtype of the returned tensor.

    Returns:
        torch.Tensor: Shape ``(1, S, S, 1)``.
    """
    _, x2 = _forcing_grid(S, domain_length, dtype)
    k = 2.0 * math.pi * wavenumber / domain_length
    return (-amplitude * k * torch.cos(k * x2)).reshape(1, S, S, 1)


def _NS_velocity_derivatives(w, domain_length):
    """Spectral spatial derivatives of a velocity-pressure state.

    Args:
        w: State of shape ``(batch, 3, nx, ny, nt)`` — channels ux, uy, p.
        domain_length: Side length of the square periodic domain.

    Returns:
        tuple: ``(dux_dx, dux_dy, duy_dx, duy_dy, dp_dx, dp_dy, lap_ux, lap_uy)``,
        each of shape ``(batch, nx, ny, nt)``.
    """
    nx = w.size(2)
    ny = w.size(3)
    k_x, k_y = _residual_wavenumbers(nx, ny, domain_length, w.device, w.dtype)
    lap = k_x**2 + k_y**2

    ux_h = torch.fft.fft2(w[:, 0], dim=[1, 2])
    uy_h = torch.fft.fft2(w[:, 1], dim=[1, 2])
    p_h = torch.fft.fft2(w[:, 2], dim=[1, 2])

    return (
        torch.fft.ifft2(1j * k_x * ux_h, dim=[1, 2]).real,
        torch.fft.ifft2(1j * k_y * ux_h, dim=[1, 2]).real,
        torch.fft.ifft2(1j * k_x * uy_h, dim=[1, 2]).real,
        torch.fft.ifft2(1j * k_y * uy_h, dim=[1, 2]).real,
        torch.fft.ifft2(1j * k_x * p_h, dim=[1, 2]).real,
        torch.fft.ifft2(1j * k_y * p_h, dim=[1, 2]).real,
        torch.fft.ifft2(-lap * ux_h, dim=[1, 2]).real,
        torch.fft.ifft2(-lap * uy_h, dim=[1, 2]).real,
    )


def FDM_NS_velocity(w, nu, alpha, t_interval, domain_length):
    """Compute PDE residuals for the 2D incompressible Navier-Stokes equations
    in velocity-pressure (primitive variable) form using spectral derivatives.

    Equations (matching ns2d/solver.py):
        Continuity:   div(u) = dux/dx + duy/dy = 0
        x-momentum:   dux/dt + ux*dux/dx + uy*dux/dy + dp/dx - nu*lap(ux) + alpha*ux = fx
        y-momentum:   duy/dt + ux*duy/dx + uy*duy/dy + dp/dy - nu*lap(uy) + alpha*uy = fy

    Spatial derivatives are computed spectrally (FFT) on a periodic square
    domain. Time derivatives use second-order central differences.

    Args:
        w: Input tensor of shape (batchsize, 3, nx, ny, nt).
            Channel 0: ux (x-velocity), channel 1: uy (y-velocity),
            channel 2: p (pressure).
        nu: Kinematic viscosity.
        alpha: Linear friction coefficient.
        t_interval: Total time spanned by the nt snapshots.
        domain_length: Side length of the square periodic domain.

    Returns:
        tuple: (R_cont, R_momx, R_momy)
            R_cont: continuity residual, shape (batchsize, nx, ny, nt)
            R_momx: x-momentum LHS residual (without forcing), (batchsize, nx, ny, nt-2)
            R_momy: y-momentum LHS residual (without forcing), (batchsize, nx, ny, nt-2)
    """
    nt = w.size(4)
    ux = w[:, 0]  # (B, nx, ny, nt)
    uy = w[:, 1]

    dux_dx, dux_dy, duy_dx, duy_dy, dp_dx, dp_dy, lap_ux, lap_uy = _NS_velocity_derivatives(
        w, domain_length
    )

    # Continuity residual (all time steps)
    R_cont = dux_dx + duy_dy  # (B, nx, ny, nt)

    # Time derivatives via central differences (interior time steps only)
    dt = t_interval / (nt - 1)
    dux_dt = (ux[:, :, :, 2:] - ux[:, :, :, :-2]) / (2 * dt)
    duy_dt = (uy[:, :, :, 2:] - uy[:, :, :, :-2]) / (2 * dt)

    # Momentum residuals (LHS; compare against forcing to close the equations)
    R_momx = dux_dt + (ux * dux_dx + uy * dux_dy + dp_dx - nu * lap_ux + alpha * ux)[..., 1:-1]
    R_momy = duy_dt + (ux * duy_dx + uy * duy_dy + dp_dy - nu * lap_uy + alpha * uy)[..., 1:-1]

    return R_cont, R_momx, R_momy


def INT_NS_velocity(w, nu, alpha, t_interval, domain_length):
    """Compute PDE residuals using the integral (trapezoidal) form in time.

    Same equations as FDM_NS_velocity, but instead of approximating du/dt
    with central differences, enforces the integrated form between consecutive
    time steps: u(t_{n+1}) - u(t_n) = dt/2 * (RHS_n + RHS_{n+1}).

    This avoids O(dt^2) truncation error from time derivatives and is better
    suited to coarse output time spacing.

    Args:
        w: Input tensor of shape (batchsize, 3, nx, ny, nt).
        nu, alpha, t_interval, domain_length: Same as FDM_NS_velocity.

    Returns:
        tuple: (R_cont, R_momx, R_momy)
            R_cont: continuity residual, shape (batchsize, nx, ny, nt)
            R_momx: integral residual (delta_ux - dt*avg_RHS_x), (batchsize, nx, ny, nt-1)
            R_momy: integral residual (delta_uy - dt*avg_RHS_y), (batchsize, nx, ny, nt-1)
    """
    nt = w.size(4)
    ux = w[:, 0]
    uy = w[:, 1]

    dux_dx, dux_dy, duy_dx, duy_dy, dp_dx, dp_dy, lap_ux, lap_uy = _NS_velocity_derivatives(
        w, domain_length
    )

    R_cont = dux_dx + duy_dy

    # RHS without forcing (forcing added in loss)
    RHS_x = -ux * dux_dx - uy * dux_dy - dp_dx + nu * lap_ux - alpha * ux
    RHS_y = -ux * duy_dx - uy * duy_dy - dp_dy + nu * lap_uy - alpha * uy

    dt = t_interval / (nt - 1)
    delta_ux = ux[..., 1:] - ux[..., :-1]
    delta_uy = uy[..., 1:] - uy[..., :-1]
    avg_RHS_x = 0.5 * (RHS_x[..., :-1] + RHS_x[..., 1:])
    avg_RHS_y = 0.5 * (RHS_y[..., :-1] + RHS_y[..., 1:])
    R_momx = delta_ux - dt * avg_RHS_x
    R_momy = delta_uy - dt * avg_RHS_y

    return R_cont, R_momx, R_momy


def max_abs_divergence_2d_velocity(velocity, domain_lengths):
    """Return per-sample max |div u| for a 2D velocity field in physical space.

    Args:
        velocity: Tensor of shape (batch, nx, ny, 2).
        domain_lengths: Physical domain lengths (Lx, Ly).

    Returns:
        Tensor of shape (batch,) with the maximum absolute divergence value for
        each sample.
    """
    if velocity.ndim != 4 or velocity.shape[-1] != 2:
        raise ValueError("velocity must have shape (batch, nx, ny, 2)")

    batchsize, nx, ny, _ = velocity.shape
    device = velocity.device
    dtype = velocity.dtype

    dx = domain_lengths[0] / nx
    dy = domain_lengths[1] / ny
    kx = 2 * math.pi * torch.fft.fftfreq(nx, d=dx).to(device=device, dtype=dtype)
    ky = 2 * math.pi * torch.fft.fftfreq(ny, d=dy).to(device=device, dtype=dtype)
    kx = kx.reshape(1, nx, 1)
    ky = ky.reshape(1, 1, ny)

    ux = velocity[..., 0]
    uy = velocity[..., 1]
    ux_h = torch.fft.fft2(ux, dim=(1, 2))
    uy_h = torch.fft.fft2(uy, dim=(1, 2))

    dux_dx = torch.fft.ifft2(1j * kx * ux_h, dim=(1, 2)).real
    duy_dy = torch.fft.ifft2(1j * ky * uy_h, dim=(1, 2)).real
    div = dux_dx + duy_dy
    return div.abs().amax(dim=(1, 2))


@lru_cache(maxsize=8)
def _spectral_curl_2d(domain_lengths):
    """The 2D spectral curl operator for one domain, built once and reused.

    ``SpectralCurl`` caches its wavenumbers by grid shape, device and dtype internally,
    so a single instance per domain serves every batch of a run.

    The import is deferred rather than hoisted because this is the only thing in
    ``utils`` that reaches into ``models``, and keeping it here means importing a loss
    function does not drag the model package in behind it. There is no cycle to avoid:
    ``models/layers.py`` imports nothing from ``utils``.
    """
    from models.layers import SpectralCurl

    return SpectralCurl(domain_lengths=domain_lengths)


def curl_rel_l2_2d(pred_velocity, truth_velocity, domain_lengths):
    """Relative L2 between the spectral vorticities of two 2D velocity fields.

    The velocity relative L2 weighs every wavenumber the same, so an error in the band
    above the model's spectral cutoff costs it almost nothing -- which is the mechanism
    #42 measured behind the E3 rollout blow-up. Taking the curl first weights wavenumber
    ``k`` by ``k``, because ``omega_hat = i (kx v_hat - ky u_hat)``, so the same term a
    vorticity-formulation model gets for free is available to a velocity one.

    Args:
        pred_velocity: Predicted velocity, shape ``(batch, nx, ny, 2)``, physical units.
        truth_velocity: Ground-truth velocity of the same shape and units.
        domain_lengths: Physical domain lengths ``(Lx, Ly)``.

    Returns:
        torch.Tensor: Scalar, the batch-mean relative L2 of the vorticity. Zero when the
        prediction is the data.

    Raises:
        ValueError: The state is not a two-component velocity on a 2D grid.
    """
    for name, field in (("pred_velocity", pred_velocity), ("truth_velocity", truth_velocity)):
        if field.ndim != 4 or field.shape[-1] != 2:
            raise ValueError(f"{name} must have shape (batch, nx, ny, 2), got {tuple(field.shape)}")

    curl = _spectral_curl_2d(tuple(float(length) for length in domain_lengths))
    # SpectralCurl reads the channel axis immediately before the spatial axes, and its
    # axis 0 is x -- the same convention as max_abs_divergence_2d_velocity above.
    pred_vorticity = curl(pred_velocity.permute(0, 3, 1, 2))
    truth_vorticity = curl(truth_velocity.permute(0, 3, 1, 2))
    # LpLoss is the data term's own norm, so the two are directly comparable and the
    # weight that balances them at initialisation is a plain ratio.
    return LpLoss(size_average=True).rel(pred_vorticity, truth_vorticity)


@dataclass(frozen=True)
class ResidualScales:
    """Reference scales of the relative velocity residuals, measured on truth.

    Attributes:
        cont: RMS of the truth's divergence components, sqrt(mean(dux_dx^2 + duy_dy^2)).
        momx: RMS of the truth's x-velocity increment over one snapshot interval.
        momy: RMS of the truth's y-velocity increment over one snapshot interval.
    """

    cont: float
    momx: float
    momy: float


def residual_scales_from_truth(u_truth, physics, t_interval, time_method="integral"):
    """Measure the reference scales of the relative residuals on ground-truth states.

    The relative residuals divide by these, so they are comparable across models:
    a prediction cannot lower its own score by inflating its own statistics.

    The scales are a property of the sample they are measured on, so the sample is
    recorded alongside the numbers: training measures them once on the first
    validation batch, evaluation on the test trajectories it scores.

    Args:
        u_truth: Ground-truth states, shape ``(batch, 3, nx, ny, nt)``, ``nt >= 2``
            (``nt >= 3`` for ``time_method="fdm"``).
        physics: The run's :class:`Physics`; supplies the domain length.
        t_interval: Total time spanned by the ``nt`` snapshots.
        time_method: ``"integral"`` (one-step increment) or ``"fdm"`` (time
            derivative), matching the residual the scales are used with.

    Returns:
        ResidualScales: the scales, as plain floats.
    """
    nt = u_truth.size(4)
    if nt < 2:
        raise ValueError("reference scales need at least two ground-truth time levels")

    ux = u_truth[:, 0]
    uy = u_truth[:, 1]
    dux_dx, _, _, duy_dy, *_ = _NS_velocity_derivatives(u_truth, physics.domain_length)
    cont = torch.sqrt(torch.mean(dux_dx**2 + duy_dy**2))

    if time_method == "fdm":
        if nt < 3:
            raise ValueError('time_method "fdm" needs at least three time levels')
        dt = t_interval / (nt - 1)
        d_ux = (ux[..., 2:] - ux[..., :-2]) / (2 * dt)
        d_uy = (uy[..., 2:] - uy[..., :-2]) / (2 * dt)
    elif time_method == "integral":
        d_ux = ux[..., 1:] - ux[..., :-1]
        d_uy = uy[..., 1:] - uy[..., :-1]
    else:
        raise ValueError('time_method must be "integral" or "fdm"')

    return ResidualScales(
        cont=float(cont),
        momx=float(torch.sqrt(torch.mean(d_ux**2))),
        momy=float(torch.sqrt(torch.mean(d_uy**2))),
    )


def PINO_loss3d_vel(
    u,
    u0,
    forcing,
    nu,
    alpha,
    t_interval,
    domain_length,
    scales,
    time_method="integral",
    return_relative=True,
    eps=1e-8,
):
    """Physics-informed loss for the velocity-pressure NS formulation.

    Computes initial-condition loss plus PDE residual losses for the
    continuity and momentum equations.

    Args:
        u: Predicted fields, shape (batchsize, 3, nx, ny, nt).
            Channels: ux, uy, p.
        u0: Initial condition, shape (batchsize, 3, nx, ny) or broadcastable.
        forcing: Velocity forcing, shape (1, 2, nx, ny, 1).
            Channel 0: fx, channel 1: fy.
        nu: Kinematic viscosity.
        alpha: Linear friction coefficient.
        t_interval: Total time spanned by the nt snapshots.
        domain_length: Side length of the square periodic domain.
        scales: ResidualScales measured on the ground truth, as returned by
            residual_scales_from_truth; the denominators of the relative losses.
        time_method: "integral" (default) or "fdm". Integral uses trapezoidal
            form in time (no du/dt); fdm uses central-difference time derivatives.

    Returns:
        If return_relative is False:
            tuple: (loss_ic, loss_cont, loss_momx, loss_momy)
            loss_ic:   relative L2 error on initial condition
            loss_cont: MSE of the continuity residual over time levels 1..nt-1
            loss_momx: MSE of x-momentum residual vs forcing
            loss_momy: MSE of y-momentum residual vs forcing
        If return_relative is True:
            tuple: (loss_ic, loss_cont, loss_momx, loss_momy, loss_cont_rel, loss_momx_rel, loss_momy_rel)
            where each *_rel is the RMS residual divided by the matching
            ground-truth scale in ``scales``.

    Time level 0 is the input state, which is ground truth when the model is
    scored one step ahead; the continuity residual therefore covers levels
    1..nt-1 only, so that a known-zero level cannot halve it.
    """
    batchsize = u.size(0)
    nx = u.size(2)
    ny = u.size(3)
    nt = u.size(4)

    u = u.reshape(batchsize, 3, nx, ny, nt)
    lploss = LpLoss(size_average=True)

    u_in = u[:, :, :, :, 0]
    loss_ic = lploss(u_in, u0)

    if time_method == "fdm":
        R_cont, R_momx, R_momy = FDM_NS_velocity(u, nu, alpha, t_interval, domain_length)
        n_mom = nt - 2
        fx = forcing[:, 0].repeat(batchsize, 1, 1, n_mom)
        fy = forcing[:, 1].repeat(batchsize, 1, 1, n_mom)
        momx_def = R_momx - fx
        momy_def = R_momy - fy
        loss_momx = torch.mean(momx_def**2)
        loss_momy = torch.mean(momy_def**2)
    else:
        if time_method != "integral":
            raise ValueError('time_method must be "integral" or "fdm"')
        R_cont, R_momx, R_momy = INT_NS_velocity(u, nu, alpha, t_interval, domain_length)
        dt = t_interval / (nt - 1)
        n_mom = nt - 1
        fx = forcing[:, 0].repeat(batchsize, 1, 1, n_mom)
        fy = forcing[:, 1].repeat(batchsize, 1, 1, n_mom)
        momx_def = R_momx - dt * fx
        momy_def = R_momy - dt * fy
        loss_momx = torch.mean(momx_def**2)
        loss_momy = torch.mean(momy_def**2)

    # Score continuity on the predicted levels only; level 0 is the input state.
    R_cont = R_cont[..., 1:]
    loss_cont = torch.mean(R_cont**2)
    if not return_relative:
        return loss_ic, loss_cont, loss_momx, loss_momy

    loss_cont_rel = torch.sqrt(torch.mean(R_cont**2)) / (scales.cont + eps)
    loss_momx_rel = torch.sqrt(torch.mean(momx_def**2)) / (scales.momx + eps)
    loss_momy_rel = torch.sqrt(torch.mean(momy_def**2)) / (scales.momy + eps)
    return loss_ic, loss_cont, loss_momx, loss_momy, loss_cont_rel, loss_momx_rel, loss_momy_rel


def PINO_loss3d_physics(u, forcing, physics, t_interval, scales=None):
    """PDE losses for a trajectory, in the formulation named by the configuration.

    Args:
        u: State of shape ``(batch, channels, nx, ny, nt)``; three channels
            (ux, uy, p) for the velocity formulation, one (w) for vorticity.
        forcing: Forcing on the grid the residual is evaluated on, as returned
            by :func:`build_forcing`.
        physics: The run's :class:`Physics`.
        t_interval: Total time spanned by the ``nt`` snapshots.
        scales: :class:`ResidualScales` measured on the ground truth. Required
            by the velocity formulation; the vorticity residual is relative to
            the forcing and ignores it.

    Returns:
        tuple: ``(loss_ic, loss_cont, loss_momx, loss_momy, loss_cont_rel,
        loss_momx_rel, loss_momy_rel)``. The vorticity formulation has a single
        residual: it is returned in both continuity slots (the ones weighted by
        ``train.f_loss``, as in PINO) and the momentum slots are zero.
    """
    if u.size(1) != state_channels(physics):
        raise ValueError(
            f"the {physics.formulation} formulation carries {state_channels(physics)} state "
            f"channel(s), got {u.size(1)}"
        )

    if physics.formulation == "vorticity":
        w = u[:, 0]
        loss_ic, loss_f = PINO_loss3d(
            w,
            w[..., 0],
            forcing,
            nu=physics.nu,
            alpha=physics.alpha,
            t_interval=t_interval,
            domain_length=physics.domain_length,
            upsample=physics.residual_upsample,
        )
        zero = torch.zeros((), device=w.device, dtype=w.dtype)
        return loss_ic, loss_f, zero, zero, loss_f, zero, zero

    if scales is None:
        raise ValueError(
            "the velocity residual needs reference scales measured on the ground truth; "
            "build them with residual_scales_from_truth"
        )
    return PINO_loss3d_vel(
        u,
        u[..., 0],
        forcing,
        nu=physics.nu,
        alpha=physics.alpha,
        t_interval=t_interval,
        domain_length=physics.domain_length,
        scales=scales,
        return_relative=True,
    )


def get_forcing_vel(S, amplitude, wavenumber, phase, domain_length, dtype=torch.float32):
    """Return the diagonal Kolmogorov velocity forcing on an S x S grid.

    The vector forcing (fx, fy) is constructed via the streamfunction
    approach used in ns2d/forcing.py::distributed_kolmogorov_forcing,
    guaranteeing divergence-free forcing.  The dataset of
    examples/run_kolmogorov.sh uses amplitude 2*sqrt(2), wavenumber 4 and
    phase 5*pi/4 on a 2*pi domain.

    Args:
        S: Spatial grid resolution (number of points per side).
        amplitude: Forcing amplitude (f0).
        wavenumber: Mode number of the driving mode along each axis (k_drive).
        phase: Phase of the driving mode.
        domain_length: Side length of the square periodic domain.
        dtype: Real dtype of the returned tensor.

    Returns:
        torch.Tensor: Shape (1, 2, S, S, 1) with fx and fy components.
    """
    x, y = _forcing_grid(S, domain_length, dtype)

    # Physical wavenumbers of the diagonal forcing mode
    kx = 2.0 * np.pi * wavenumber / domain_length  # = wavenumber when L = 2*pi
    ky = 2.0 * np.pi * wavenumber / domain_length  # = wavenumber when L = 2*pi
    k2 = kx**2 + ky**2

    # theta directly in physical coordinates
    theta = kx * x + ky * y + phase

    # F = grad_perp(psi) where -lap(psi) = amplitude*(sin(theta)+cos(theta))
    dpsi_dtheta = (amplitude / k2) * (torch.cos(theta) - torch.sin(theta))
    fx = dpsi_dtheta * ky  # d(psi)/dy
    fy = -dpsi_dtheta * kx  # -d(psi)/dx

    return torch.stack([fx, fy], dim=0).reshape(1, 2, S, S, 1)


def build_forcing(physics, S, device=None, dtype=torch.float32):
    """Build the forcing field named by ``physics.forcing`` on an S x S grid.

    Args:
        physics: The run's :class:`Physics`.
        S: Spatial grid resolution (points per side); for the vorticity
            residual this is the refined resolution.
        device: Torch device for the returned tensor.
        dtype: Real dtype of the returned tensor.

    Returns:
        torch.Tensor: Shape ``(1, 2, S, S, 1)`` for the velocity formulation
        (channels fx, fy), or ``(1, S, S, 1)`` for the vorticity formulation.
    """
    forcing_type = physics.forcing["type"]
    velocity_form = physics.formulation == "velocity"
    shape = (1, 2, S, S, 1) if velocity_form else (1, S, S, 1)

    if forcing_type == "none":
        forcing = torch.zeros(shape, dtype=dtype)
    elif forcing_type == "kolmogorov_diag":
        amplitude = float(physics.forcing["amplitude"])
        wavenumber = float(physics.forcing["wavenumber"])
        phase = float(physics.forcing["phase"])
        if velocity_form:
            forcing = get_forcing_vel(
                S,
                amplitude=amplitude,
                wavenumber=wavenumber,
                phase=phase,
                domain_length=physics.domain_length,
                dtype=dtype,
            )
        else:
            # curl of that forcing: it is grad_perp(psi) with -lap(psi) the field below.
            x, y = _forcing_grid(S, physics.domain_length, dtype)
            k = 2.0 * math.pi * wavenumber / physics.domain_length
            theta = k * x + k * y + phase
            forcing = (amplitude * (torch.sin(theta) + torch.cos(theta))).reshape(1, S, S, 1)
    else:  # pino_cos4y
        amplitude = float(physics.forcing["amplitude"])
        wavenumber = float(physics.forcing["wavenumber"])
        if velocity_form:
            # f = (amplitude*sin(k*y), 0); its curl is the vorticity forcing below.
            _, y = _forcing_grid(S, physics.domain_length, dtype)
            k = 2.0 * math.pi * wavenumber / physics.domain_length
            fx = amplitude * torch.sin(k * y)
            forcing = torch.stack([fx, torch.zeros_like(fx)], dim=0).reshape(1, 2, S, S, 1)
        else:
            forcing = get_forcing(
                S,
                amplitude=amplitude,
                wavenumber=wavenumber,
                domain_length=physics.domain_length,
                dtype=dtype,
            )

    return forcing if device is None else forcing.to(device)


def build_forcing_for_data(physics, grid_shape, device=None):
    """Build the forcing on the grid the residual is evaluated on.

    Args:
        physics: The run's :class:`Physics`.
        grid_shape: ``(nx, ny)`` of the loaded dataset, checked against the
            resolution the configuration declares.
        device: Torch device for the returned tensor.

    Returns:
        torch.Tensor: the forcing at the data resolution for the velocity
        formulation, and at ``residual_upsample`` times it for the vorticity
        formulation, where PINO evaluates the residual.
    """
    if tuple(grid_shape) != (physics.nx, physics.ny):
        raise ValueError(
            f"data resolution {tuple(grid_shape)} does not match config['data'] "
            f"nx/ny ({physics.nx}, {physics.ny})"
        )
    S = grid_shape[0]  # need to change this if rectangular grid
    if physics.formulation == "vorticity":
        S *= physics.residual_upsample
    return build_forcing(physics, S, device=device)


def get_loss_func(name, component, normalizer):
    if name == "rel2":
        return RelLpLoss(p=2, component=component, normalizer=normalizer)
    elif name == "rel1":
        return RelLpLoss(p=1, component=component, normalizer=normalizer)
    elif name == "l2":
        return LpLoss(p=2, component=component, normalizer=normalizer)
    elif name == "l1":
        return LpLoss(p=1, component=component, normalizer=normalizer)
    else:
        raise NotImplementedError


class RelL2Norm(_WeightedLoss):
    def __init__(self, d=2, p=2, size_average=True, reduction=True, return_comps=False):
        super().__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average
        self.return_comps = return_comps

    def forward(self, pred, y):
        # x: shape (B, H, W, T, C), y: shape (B, H, W, T, C)
        B, C = y.shape[0], y.shape[-1]
        # reshape the x and y to (B*T, HW, C)
        # pred = rearrange(pred, 'b h w t c -> (b t) (h w) c')
        # y = rearrange(y, 'b h w t c -> (b t) (h w) c')
        pred = pred.reshape(B, -1, C)
        y = y.reshape(B, -1, C)

        diff_norms = torch.sqrt(torch.sum((pred - y) ** 2, dim=1))  # (B*T, C)
        y_norms = torch.sqrt(torch.sum(y**2, dim=1))  # (B*T, C)
        loss = torch.mean(
            diff_norms / (y_norms + 1e-8)
        )  # average over the batch size, time steps, and channels
        return loss


class RMSE(_WeightedLoss):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, y):
        # x: shape (B, H, W, T, C), y: shape (B, H, W, T, C)
        B, H, W, T, C = y.shape
        # reshape the x and y to (B*T, HW, C)
        rmse = torch.sqrt(self.mse(pred, y))
        return rmse


class BoundaryRMSE(_WeightedLoss):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def extract_boundary(self, x):
        # x (N, H, W, C),
        # return (N, 2*H+2*W, C)
        x_bound = torch.cat(
            [x[..., 0, :, :], x[..., -1, :, :], x[..., :, 0, :], x[..., :, -1, :]], dim=-2
        )
        return x_bound

    def forward(self, pred, y):
        B, H, W, T, C = y.shape
        # reshape the x and y to (B*T, HW, C)
        pred = rearrange(pred, "b h w t c -> (b t) h w c")
        y = rearrange(y, "b h w t c -> (b t) h w c")

        # extrat
        pred_bound = self.extract_boundary(pred)
        y_bound = self.extract_boundary(y)
        rmse = torch.sqrt(self.mse(pred_bound, y_bound))
        return rmse


class MaxAbsError(_WeightedLoss):
    """Compute the max absolute error sample average"""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, y):
        B, H, W, T, C = y.shape
        # reshape the x and y to (B*T, HW, C)
        pred = rearrange(pred, "b h w t c -> (b t) (h w) c")
        y = rearrange(y, "b h w t c -> (b t) (h w) c")

        # compute the max error across the (H, W) then compute the sample average
        max_error = torch.max(torch.abs(pred - y), dim=1).values  # (B, C)
        max_error = torch.mean(max_error)  # average over the batch size and channels
        return max_error


class GlobalMaxAbsError(_WeightedLoss):
    """Compute the max absolute error global average"""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, y):
        B, H, W, T, C = y.shape
        # reshape the x and y to (B*T, HW, C)
        pred = rearrange(pred, "b h w t c -> (b t) (h w) c")
        y = rearrange(y, "b h w t c -> (b t) (h w) c")

        # compute the global max error
        max_error = torch.max(torch.abs(pred - y))  # just global max error
        return max_error


class SpectralError(_WeightedLoss):
    def __init__(
        self, model_name, save_path, low_percentile=0.80, high_percentile=0.99, method="radial"
    ):
        super().__init__()
        self.mse = nn.MSELoss()
        self.k_low = None
        self.k_high = None
        self.model_name = model_name
        self.save_path = save_path
        self.low_percentile = low_percentile
        self.high_percentile = high_percentile
        self.method = method
        if method == "radial":
            self.method_f = get_frequency_bands_from_cumulative_energy
        elif method == "square approximation":
            self.method_f = spectrum_2d
        elif method == "cfd":
            self.method_f = spectral_cfd
        self.method = method

    def forward(self, pred, y, channel=None, time_step=None, save_plot=False):
        B, H, W, T, C = y.shape
        # find the spectral band edges from the truth using OLD method
        k_low_new, k_high_new, k_freq, E_bins_target = self.method_f(
            y, low_percentile=self.low_percentile, high_percentile=self.high_percentile
        )
        _, _, _, E_bins_pred = self.method_f(  # new method
            pred, low_percentile=self.low_percentile, high_percentile=self.high_percentile
        )
        if self.k_low is None:
            self.k_low = k_low_new
        if self.k_high is None:
            self.k_high = k_high_new
        # Aggregate spectral errors by frequency bands
        low_err, mid_err, high_err = aggregate_spectral_energy_by_bands(
            self.k_low, self.k_high, np.abs(np.log(E_bins_pred) - np.log(E_bins_target))
        )

        # print(f"\nSpectral error by frequency bands:")
        # print(f"Low frequency error (0 to {self.k_low}): {low_err:.6f}")
        # print(f"Mid frequency error ({self.k_low} to {self.k_high}): {mid_err:.6f}")
        # print(f"High frequency error ({self.k_high}+): {high_err:.6f}")

        if save_plot:
            # increase the font size
            plt.loglog(k_freq, E_bins_target, "X-", markersize=2, label="target", linewidth=2)
            plt.loglog(
                k_freq,
                E_bins_pred,
                "o-",
                markersize=2,
                label=f"{self.model_name} pred",
                linewidth=2,
            )
            font_size = 16
            plt.legend(fontsize=font_size)
            # draw the dotted line at two points line 1: (self.k_low,0) to (self.k_low, k_freq[self.k_low]), line 2: (self.k_high,0) to (self.k_high, k_freq[self.k_high])
            plt.axvline(
                x=self.k_low,
                color="black",
                linestyle="--",
                linewidth=1,
                ymin=0,
                ymax=k_freq[self.k_low],
            )
            plt.axvline(
                x=self.k_high,
                color="black",
                linestyle="--",
                linewidth=1,
                ymin=0,
                ymax=k_freq[self.k_high],
            )
            # set the font size for x tick and y tick labels
            plt.xticks(fontsize=font_size)
            plt.yticks(fontsize=font_size)
            if not os.path.exists(f"{self.save_path}/spectral_error"):
                os.makedirs(f"{self.save_path}/spectral_error")
            plt.savefig(
                f"{self.save_path}/spectral_error/spectral_error_{self.model_name}_{self.method}_c{channel}_t{time_step}.png"
            )
            plt.clf()
        # plt.show()

        return {
            "spec_low": low_err,
            "spec_mid": mid_err,
            "spec_high": high_err,
            "k_low": self.k_low,
            "k_high": self.k_high,
        }


class Energy_Enstropy_SpectrumError(_WeightedLoss):
    def __init__(self, model_name, save_path):
        super().__init__()
        self.model_name = model_name
        self.save_path = save_path

    def forward(
        self, pred, y, Lx=2 * np.pi, Ly=2 * np.pi, channel=None, time_step=None, save_plot=False
    ):
        B, H, W, C = y.shape
        y_dict = {"pred": pred, "y": y}
        # Get fields at this time
        for y_key in ["pred", "y"]:  # pred and y
            psi_grid = y_dict[y_key][0, ..., 1].detach().cpu().numpy()  # streamfunction
            omega_grid = y_dict[y_key][0, ..., 0].detach().cpu().numpy()  # vorticity

            # Compute velocity from streamfunction
            ux_grid, uy_grid = streamfunction_to_velocity(psi_grid, Lx, Ly)

            # Spectra (every time step)
            k_bins, Ek, Zk = compute_spectra(ux_grid, uy_grid, Lx, Ly)

            y_dict[y_key + "_Ek"] = Ek
            y_dict[y_key + "_Zk"] = Zk
            y_dict["k_bins"] = k_bins
        if save_plot:
            # plot the energy and enstropy spectra
            font_size = 16
            fig, axs = plt.subplots(2, 1, figsize=(10, 10))

            k_nyquist = int((np.pi * H) // Lx)
            # print('k_nyquist', k_nyquist)
            # y_temp = y_dict['y_Ek'][:k_nyquist]
            # print('y temp min', np.min(y_temp), 'y temp max', np.max(y_temp))
            # # sort y_temp and return the indices
            # y_temp_sorted_indices = np.argsort(y_temp)
            # y_temp_sorted = y_temp[y_temp_sorted_indices]
            # print('index', y_temp_sorted_indices, "top 5", y_temp_sorted[:5], "bottom 5", y_temp_sorted[-5:])

            start_truth = 1
            # increase the font size
            axs[0].loglog(
                y_dict["k_bins"][start_truth:k_nyquist],
                y_dict["y_Ek"][start_truth:k_nyquist],
                "X-",
                markersize=2,
                label="target",
                linewidth=2,
            )
            axs[0].loglog(
                y_dict["k_bins"][start_truth:k_nyquist],
                y_dict["pred_Ek"][start_truth:k_nyquist],
                "o-",
                markersize=2,
                label=f"{self.model_name} pred",
                linewidth=2,
            )
            # axs[0].set_xlabel('Wavenumber', fontsize=font_size)
            # axs[0].set_ylim(1e-10, 1e-2) # TODO: remove this later
            axs[0].set_ylabel("Energy", fontsize=font_size)
            axs[0].set_title("Energy Spectrum", fontsize=font_size)
            axs[0].legend(fontsize=font_size)
            axs[0].grid(True)

            axs[1].loglog(
                y_dict["k_bins"][start_truth:k_nyquist],
                y_dict["y_Zk"][start_truth:k_nyquist],
                "X-",
                markersize=2,
                label="target",
                linewidth=2,
            )
            axs[1].loglog(
                y_dict["k_bins"][start_truth:k_nyquist],
                y_dict["pred_Zk"][start_truth:k_nyquist],
                "o-",
                markersize=2,
                label=f"{self.model_name} pred",
                linewidth=2,
            )
            axs[1].set_xlabel("Wavenumber", fontsize=font_size)
            axs[1].set_ylabel("Enstropy", fontsize=font_size)
            axs[1].set_title("Enstropy Spectrum", fontsize=font_size)
            axs[1].legend(fontsize=font_size)
            axs[1].grid(True)
            # set the font size for x tick and y tick labels
            plt.xticks(fontsize=font_size)
            plt.yticks(fontsize=font_size)
            if not os.path.exists(f"{self.save_path}/spectral_error"):
                os.makedirs(f"{self.save_path}/spectral_error")
            plt.savefig(
                f"{self.save_path}/spectral_error/energy_enstropy_spectra_{self.model_name}_t{time_step}.png"
            )
            plt.clf()


def compute_frequency_spectrum(y_pred, y):
    # y_pred: (B, H, W, T, C), y: (B, H, W, T, C)
    B, H, W, T, C = y.shape

    # Absolute error averaged over batch, time, channels -> (H, W)
    abs_error = torch.abs(y - y_pred)  # (B, H, W, T, C)
    abs_error = rearrange(abs_error, "b h w t c -> b t c h w")

    # Use full 2D FFT so the spectrum shape matches (B T C H, W)
    abs_error_fft = torch.abs(torch.fft.fft2(abs_error))

    # average over batch, time, channels
    abs_error_fft = abs_error_fft.mean(dim=(0, 1, 2))  # (H, W)
    # Take magnitude and move to numpy for binning
    fourier_amplitudes = abs_error_fft.detach().cpu().numpy()

    # Create the k-frequency grid for rectangular image
    kfreq_x = np.fft.fftfreq(W) * W
    kfreq_y = np.fft.fftfreq(H) * H
    kfreq2D = np.meshgrid(kfreq_x, kfreq_y)
    knrm = np.sqrt(kfreq2D[0] ** 2 + kfreq2D[1] ** 2)

    # Flatten the arrays to use in binning (1D arrays of equal length)
    knrm = knrm.ravel()
    fourier_amplitudes = fourier_amplitudes.ravel()

    # Define the bins for the wavenumber - use the minimum dimension for binning
    min_dim = min(H, W)
    kbins = np.arange(0.5, min_dim // 2 + 1, 1.0)

    # Bin the data (radial mean)
    Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes, statistic="mean", bins=kbins)

    return Abins


def compute_error_fft(model, test_loader, num_bins, device, args):
    # compute the frequency spectrum of the error on the validation set
    with torch.no_grad():
        model.eval()
        error_fft_epoch = torch.zeros(num_bins)
        for xx, yy in test_loader:
            xx = xx.to(device)  ## B, n, n, T_in, C
            yy = yy.to(device)  ## B, n, n, T_ar, C
            xx = test_loader.dataset.normalize_x(xx)
            yy_norm = test_loader.dataset.normalize_x(yy)
            for t in range(0, yy_norm.shape[-2], args.T_bundle):
                # print("t", t)
                y = yy_norm[..., t : t + args.T_bundle, :]
                pred_step = model(xx)
                break
            err_spec = compute_frequency_spectrum(pred_step, y)
            y_spec = compute_frequency_spectrum(torch.zeros_like(y), y)
            rel_error_spec = np.log(np.abs(err_spec / (y_spec + 1e-8)))
            error_fft_epoch += rel_error_spec * xx.shape[0]
        error_fft_epoch /= len(test_loader)
        return error_fft_epoch


def compute_spectra_torch(ux_grid, uy_grid, Lx, Ly):
    """PyTorch version of compute_spectra that supports gradient computation and batch operations.

    Compute isotropic 1D energy spectra from 2D velocity field.

    Uses shell-averaging in Fourier space to compute spectra as a function
    of wavenumber magnitude |k|.

    Args:
        ux_grid (torch.Tensor): x-velocity in physical space
            - Single sample: (Nx, Ny)
            - Batch: (B, Nx, Ny)
        uy_grid (torch.Tensor): y-velocity in physical space, same shape as ux_grid
        Lx (float): Domain length in x
        Ly (float): Domain length in y

    Returns:
        tuple: (k_bins, E_k)
            - k_bins: Physical wavenumber bins (rad/length) as torch.Tensor, shape (mmax+1,)
            - E_k: Energy spectrum E(k) = 0.5 <|û|²>_shell as torch.Tensor
                - Single sample: (mmax+1,)
                - Batch: (B, mmax+1)

    Notes:
        - Assumes Lx ≈ Ly for isotropic shell averaging
        - Accounts for rfft symmetry factors
        - Shell index n corresponds to physical wavenumber n*k0 where k0=2π/L
        - All operations are differentiable
        - FFT operations are batched, binning uses a loop over batches
    """
    if torch is None:
        raise ImportError("PyTorch is required for compute_spectra_torch")

    device = ux_grid.device
    dtype = ux_grid.dtype

    # Handle both single sample and batch inputs
    if ux_grid.dim() == 2:
        # Single sample: (Nx, Ny) -> add batch dimension
        ux_grid = ux_grid.unsqueeze(0)
        uy_grid = uy_grid.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    B, Nx, Ny = ux_grid.shape
    N = Nx * Ny
    assert abs(Lx - Ly) < 1e-2, (
        f"Isotropic shell binning requires Lx ≈ Ly, but Lx - Ly = {Lx - Ly}, Lx = {Lx}, Ly = {Ly}"
    )
    k0 = 2 * torch.tensor(np.pi, device=device, dtype=dtype) / Lx

    # Transform to spectral space (batched FFT)
    uxh = torch.fft.rfft2(ux_grid)  # (B, Nx, Ny//2+1)
    uyh = torch.fft.rfft2(uy_grid)  # (B, Nx, Ny//2+1)

    # Energy per mode (normalised)
    E_mode = 0.5 * (torch.abs(uxh) ** 2 + torch.abs(uyh) ** 2) / (N * N)  # (B, Nx, Ny//2+1)

    # rfft symmetry weight: double ky>0 interior modes
    # Create weight for single sample, will broadcast to batch
    weight = 2.0 * torch.ones(Nx, Ny // 2 + 1, device=device, dtype=dtype)
    weight[:, 0] = 1.0  # ky=0 is not doubled
    if Ny % 2 == 0:
        weight[:, -1] = 1.0  # Nyquist is real-valued

    E_mode = E_mode * weight.unsqueeze(0)  # Broadcast: (B, Nx, Ny//2+1)

    # Shell indices (integer radius in index space)
    # Create index arrays (same for all batches)
    ix = torch.fft.fftfreq(Nx, d=1.0 / Nx, device=device)
    iy = torch.arange(0, Ny // 2 + 1, device=device, dtype=dtype)
    IX, IY = torch.meshgrid(ix, iy, indexing="ij")
    shell_idx = torch.floor(torch.sqrt(IX**2 + IY**2)).long()  # (Nx, Ny//2+1)

    mmax = shell_idx.max().item()
    shell_idx_flat = shell_idx.ravel()  # (Nx * (Ny//2+1),)

    # Bin into shells using scatter_add (loop over batches)
    Ek_list = []
    for b in range(B):
        E_mode_b = E_mode[b]  # (Nx, Ny//2+1)
        E_mode_flat = E_mode_b.ravel()  # (Nx * (Ny//2+1),)

        # Use scatter_add to sum values in each shell (differentiable)
        Ek_b = torch.zeros(mmax + 1, device=device, dtype=dtype)
        Ek_b.scatter_add_(0, shell_idx_flat, E_mode_flat)
        Ek_list.append(Ek_b)

    Ek = torch.stack(Ek_list, dim=0)  # (B, mmax+1)

    k_bins = torch.arange(mmax + 1, device=device, dtype=dtype) * k0  # (mmax+1,)

    # Remove batch dimension if input was single sample
    if squeeze_output:
        Ek = Ek.squeeze(0)  # (mmax+1,)

    return k_bins, Ek


class LogEnstropyEnergyLoss(_WeightedLoss):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # pred: (B, H, W)
        # target: (B, H, W)

        Nx, Ny = pred.shape[1], pred.shape[2]
        N = Nx * Ny

        # Transform to spectral space
        e_pred = torch.fft.rfft2(pred)
        e_target = torch.fft.rfft2(target)

        # Energy per mode (normalised)
        e_pred = 0.5 * (torch.abs(e_pred) ** 2) / (N * N)
        e_target = 0.5 * (torch.abs(e_target) ** 2) / (N * N)

        # Log energy per mode (normalised)
        log_e_pred = torch.log(e_pred)
        log_e_target = torch.log(e_target)

        err = torch.abs(log_e_pred - log_e_target)
        return torch.mean(err)  # average over frequency bins and over the samples


def compute_2d_spectral_energy(ux_grid, uy_grid):
    """ux_grid and uy_grid need to be in shape (B, *, H, W)"""
    Nx, Ny = ux_grid.shape[-2], ux_grid.shape[-1]
    N = Nx * Ny

    # Transform to spectral space
    uxh = torch.fft.rfft2(ux_grid)
    uyh = torch.fft.rfft2(uy_grid)

    # Energy per mode (normalised)
    E_mode = 0.5 * (torch.abs(uxh) ** 2 + torch.abs(uyh) ** 2) / (N * N)
    return E_mode


def compute_2d_enstropy_spectrum(
    ux_grid=None, uy_grid=None, w_grid=None, Lx=2 * np.pi, Ly=2 * np.pi
):
    """Compute 2D enstropy spectrum.

    Can be called in two ways:
    1. From velocity: compute_2d_enstropy_spectrum(ux_grid, uy_grid, ...)
       ux_grid and uy_grid need to be in shape (B, *, H, W)
    2. From vorticity: compute_2d_enstropy_spectrum(w_grid=vorticity)
       w_grid should be in shape (B, *, H, W) - vorticity in physical space
       This is simpler and more efficient if vorticity is already available.

    If w_grid is provided, computation is simplified (direct FFT of vorticity).
    Otherwise, vorticity is computed from velocity components in spectral space.
    """
    if w_grid is not None:
        # Simplified path: vorticity provided directly in physical space
        Nx, Ny = w_grid.shape[-2], w_grid.shape[-1]
        N = Nx * Ny

        # Transform vorticity directly to spectral space
        omegah = torch.fft.rfft2(w_grid)
        Z_mode = (torch.abs(omegah) ** 2) / (N * N)

        # rfft symmetry weight: double ky>0 interior modes
        weight = 2.0 * torch.ones_like(Z_mode)
        weight[..., 0] = 1.0  # ky=0 is not doubled
        if Ny % 2 == 0:
            weight[..., -1] = 1.0  # Nyquist is real-valued

        Z_mode = Z_mode * weight
        return Z_mode

    # Original path: compute vorticity from velocity components
    if ux_grid is None or uy_grid is None:
        raise ValueError("Either (ux_grid, uy_grid) or w_grid must be provided")

    Nx, Ny = ux_grid.shape[-2], ux_grid.shape[-1]
    N = Nx * Ny

    # Transform to spectral space
    uxh = torch.fft.rfft2(ux_grid)
    uyh = torch.fft.rfft2(uy_grid)

    # Vorticity in spectral space: omega = curl(u) = d(uy)/dx - d(ux)/dy
    kx = 2 * math.pi * torch.fft.fftfreq(Nx, d=Lx / Nx).to(ux_grid.device)
    ky = 2 * math.pi * torch.fft.rfftfreq(Ny, d=Ly / Ny).to(ux_grid.device)
    KX, KY = torch.meshgrid(kx, ky, indexing="ij")
    omegah = 1j * (KX * uyh - KY * uxh)
    Z_mode = (torch.abs(omegah) ** 2) / (N * N)

    # rfft symmetry weight: double ky>0 interior modes
    weight = 2.0 * torch.ones_like(Z_mode)
    weight[..., 0] = 1.0  # ky=0 is not doubled
    if Ny % 2 == 0:
        weight[..., -1] = 1.0  # Nyquist is real-valued

    Z_mode = Z_mode * weight
    return Z_mode


class MeanEnergyAbsolutePercentageError(_WeightedLoss):
    def __init__(self):
        super().__init__()

    def forward(self, Ek_pred, Ek_target):
        # pred: (B, H, W)
        # target: (B, H, W)
        err = torch.abs((Ek_pred - Ek_target) / Ek_target)
        return torch.mean(err)  # percentage error


class MeanEnergyLogRatioError(_WeightedLoss):
    def __init__(self):
        super().__init__()

    def forward(self, Ek_pred, Ek_target):
        # pred: (B, H, W)
        # target: (B, H, W)
        # print("ratio max", torch.max(Ek_pred / Ek_target+1e-10), "ratio min", torch.min(Ek_pred / Ek_target+1e-10))
        err = torch.abs(torch.log(Ek_pred / (Ek_target + 1e-10)))
        # print("err max", torch.max(err), "err min", torch.min(err))
        return torch.mean(err)  # average over frequency bins and over the samples


class EnergySpectrumBias1D(_WeightedLoss):
    def __init__(self, log_scale=False):
        super().__init__()
        self.log_scale = log_scale

    def forward(self, pred, target, Lx=2 * np.pi, Ly=2 * np.pi, ux_dim=1, uy_dim=2):
        """pred: (B, H, W, T, C)
        target: (B, H, W, T, C)
        ux_dim: the dimension of the x-velocity
        uy_dim: the dimension of the y-velocity
        """
        k_bins, Ek_pred = compute_spectra_torch(
            pred[..., 0, ux_dim], pred[..., 0, uy_dim], Lx, Ly
        )  # Ek shape (B,K_max+1)
        k_bins, Ek_target = compute_spectra_torch(
            target[..., 0, ux_dim], target[..., 0, uy_dim], Lx, Ly
        )
        # get the nyquist frequency
        nyquist_freq = int((np.pi * pred.shape[1]) / Lx)
        # print("nyquist_freq", nyquist_freq)
        start_freq = 1
        # get the energy spectrum of the error
        if self.log_scale:
            Ek_error = torch.abs(
                torch.log(Ek_pred[:, start_freq:nyquist_freq])
                - torch.log(Ek_target[:, start_freq:nyquist_freq])
            )
        else:
            Ek_error = torch.abs(
                Ek_pred[:, start_freq:nyquist_freq] - Ek_target[:, start_freq:nyquist_freq]
            )

        # print the shape of Ek_error (B, K_range)
        # print("Ek_error shape", Ek_error.shape, ""Ek_pred.shape)
        # get the bias of the energy spectrum in log scale
        loss = Ek_error.mean()  # average over frequency bins and over the samples
        return loss


class Rel_Spectral_bias(_WeightedLoss):
    def __init__(
        self,
        target_data=None,
        Lx=2 * np.pi,
        Ly=2 * np.pi,
        ux_dim=1,
        uy_dim=2,
        low_percentile=0.7,
        high_percentile=0.99,
        dataset_form="velocity",
        convert_streamfunction=True,
        method=None,
    ):
        """target_data: Optional target data (T, H, W, C) to compute k_low and k_high from.
                     If provided, averages over time steps and computes frequency bands.
        Lx, Ly: Domain sizes (default 2*pi)
        ux_dim, uy_dim: Channel dimensions for velocity components
        low_percentile, high_percentile: Percentiles for frequency band computation
        dataset_form: 'velocity' or 'vorticity' to determine data format
        convert_streamfunction: If True and dataset_form='vorticity', convert streamfunction to velocity
        method: Deprecated. Method is now passed to forward() instead. Kept for backward compatibility.
        """
        super().__init__()
        self.Lx = Lx
        self.Ly = Ly
        self.ux_dim = ux_dim
        self.uy_dim = uy_dim
        self.low_percentile = low_percentile
        self.high_percentile = high_percentile
        self.dataset_form = dataset_form
        self.convert_streamfunction = convert_streamfunction

        # Handle deprecated method parameter (backward compatibility)
        if method is not None:
            import warnings

            warnings.warn(
                "The 'method' parameter in __init__ is deprecated. Pass method to forward() instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Initialize k_low and k_high (will be computed if target_data provided)
        self.k_low = None
        self.k_high = None

        # Compute frequency bands from target data if provided
        if target_data is not None:
            self.set_frequency_bands(
                target_data,
                dataset_form=dataset_form,
                convert_streamfunction=convert_streamfunction,
            )

    def set_frequency_bands(
        self,
        target_data,
        Lx=None,
        Ly=None,
        ux_dim=None,
        uy_dim=None,
        dataset_form="velocity",
        convert_streamfunction=True,
    ):
        """Compute k_low and k_high from target data by computing energy spectrum for each time step
        and then averaging the spectra (not averaging the fields first).

        Args:
            target_data: Target data tensor with shape (T, H, W, C)
            Lx, Ly: Domain sizes (uses class defaults if None)
            ux_dim, uy_dim: Velocity channel dimensions (uses class defaults if None).
                           For vorticity form, these refer to streamfunction channel if convert_streamfunction=True.
            dataset_form: 'velocity' or 'vorticity' to determine data format
            convert_streamfunction: If True and dataset_form='vorticity', convert streamfunction to velocity
        """
        Lx = Lx if Lx is not None else self.Lx
        Ly = Ly if Ly is not None else self.Ly
        ux_dim = ux_dim if ux_dim is not None else self.ux_dim
        uy_dim = uy_dim if uy_dim is not None else self.uy_dim

        # Ensure target_data is torch tensor
        if not isinstance(target_data, torch.Tensor):
            target_data = torch.tensor(target_data, dtype=torch.float32)

        # target_data shape: (T, H, W, C)
        T = target_data.shape[0]
        H, W = target_data.shape[1], target_data.shape[2]

        # Import streamfunction conversion if needed
        if dataset_form == "vorticity" and convert_streamfunction:
            from utils.compute_diagnostics import streamfunction_to_velocity

        # Compute energy spectrum for each time step, then average
        Ek_list = []
        k_bins = None

        for t_idx in range(T):
            target_t = target_data[t_idx]  # (H, W, C)

            # Extract velocity components based on dataset form
            if dataset_form == "vorticity" and convert_streamfunction:
                # For vorticity form, extract streamfunction and convert to velocity
                if target_t.shape[-1] > max(ux_dim, uy_dim):
                    # Assume streamfunction is at the channel index specified by ux_dim
                    # (typically channel 1 for [vorticity, streamfunction])
                    psi_t = target_t[..., ux_dim]  # (H, W) - streamfunction
                    # Convert streamfunction to velocity using numpy version (for initialization)
                    ux_t_np, uy_t_np = streamfunction_to_velocity(
                        psi_t.detach().cpu().numpy(), Lx, Ly
                    )
                    ux_t = torch.from_numpy(ux_t_np).to(target_t.device).to(target_t.dtype)
                    uy_t = torch.from_numpy(uy_t_np).to(target_t.device).to(target_t.dtype)
                else:
                    raise ValueError(
                        f"Vorticity form requires at least {max(ux_dim, uy_dim) + 1} channels, but got {target_t.shape[-1]}"
                    )
            else:
                # For velocity form or when not converting, use channels directly
                ux_t = target_t[..., ux_dim]  # (H, W)
                uy_t = target_t[..., uy_dim]  # (H, W)

            # Compute spectral energy for this time step (returns (mmax+1,) for single sample)
            k_bins_t, Ek_t = compute_spectra_torch(ux_t, uy_t, Lx, Ly)

            # Store k_bins from first time step (should be the same for all)
            if k_bins is None:
                k_bins = k_bins_t

            # Store energy spectrum for this time step
            Ek_list.append(Ek_t)

        # Average energy spectra across all time steps
        # Stack all spectra: each Ek_t has shape (mmax+1,), stack to get (T, mmax+1)
        Ek_stack = torch.stack(Ek_list, dim=0)  # (T, mmax+1)
        Ek_target_avg = Ek_stack.mean(dim=0)  # (mmax+1,) - average over time

        # Get nyquist frequency
        nyquist_freq = int((np.pi * H) / Lx)
        start_freq = 1

        # Extract frequency range
        E_freq = Ek_target_avg[start_freq:nyquist_freq].detach().cpu().numpy()  # (K_range,)

        # Compute cumulative sum
        E_freq_cumsum = np.cumsum(E_freq)
        if E_freq_cumsum[-1] > 0:
            E_freq_cumsum = E_freq_cumsum / E_freq_cumsum[-1]
        else:
            # Handle edge case where all energy is zero
            E_freq_cumsum = np.ones_like(E_freq_cumsum)

        # Find frequency indices from percentiles
        self.k_low, self.k_high = find_freq_from_percentile(
            E_freq_cumsum, self.low_percentile, self.high_percentile
        )

        # Store k_bins for reference (first few for debugging)
        self.k_bins_sample = (
            k_bins.detach().cpu().numpy() if isinstance(k_bins, torch.Tensor) else k_bins
        )

    def forward(
        self,
        pred,
        target,
        method="avg",
        Lx=None,
        Ly=None,
        ux_dim=None,
        uy_dim=None,
        low_percentile=None,
        high_percentile=None,
    ):
        """pred: (B, H, W, T, C)
        target: (B, H, W, T, C)
        method: 'avg' for average over frequency bins and over the samples
                'high' for high frequency bins
                'mid' for mid frequency bins
                'low' for low frequency bins
        Lx, Ly: Domain sizes (uses class defaults if None)
        ux_dim, uy_dim: Velocity channel dimensions (uses class defaults if None)
        low_percentile, high_percentile: Percentiles (uses class defaults if None,
                                         only used if k_low/k_high not precomputed)
        """
        # Use class defaults if not provided
        Lx = Lx if Lx is not None else self.Lx
        Ly = Ly if Ly is not None else self.Ly
        ux_dim = ux_dim if ux_dim is not None else self.ux_dim
        uy_dim = uy_dim if uy_dim is not None else self.uy_dim

        k_bins, Ek_pred = compute_spectra_torch(
            pred[..., 0, ux_dim], pred[..., 0, uy_dim], Lx, Ly
        )  # Ek shape (B,K_max+1)
        k_bins, Ek_target = compute_spectra_torch(
            target[..., 0, ux_dim], target[..., 0, uy_dim], Lx, Ly
        )
        # get the nyquist frequency
        nyquist_freq = int((np.pi * pred.shape[1]) / Lx)
        start_freq = 1
        # compute the relative spectral bias, shape (B, K_range)
        rel_spectral_bias = torch.abs(
            Ek_pred[:, start_freq:nyquist_freq] - Ek_target[:, start_freq:nyquist_freq]
        ) / (torch.abs(Ek_target[:, start_freq:nyquist_freq]))
        # print("rel_spectral_bias shape", rel_spectral_bias.shape)
        # get the energy spectrum of the error
        if method == "avg":
            loss = rel_spectral_bias.mean()
        else:
            # Use precomputed k_low and k_high if available, otherwise compute from current batch
            if self.k_low is not None and self.k_high is not None:
                k_low = self.k_low
                k_high = self.k_high
            else:
                # Fallback to per-batch computation (backward compatibility)
                low_percentile = (
                    low_percentile if low_percentile is not None else self.low_percentile
                )
                high_percentile = (
                    high_percentile if high_percentile is not None else self.high_percentile
                )
                E_freq = Ek_target[:, start_freq:nyquist_freq].mean(dim=0)  # shape (K_range,)
                E_freq_cumsum = np.cumsum(E_freq.detach().cpu().numpy())
                if E_freq_cumsum[-1] > 0:
                    E_freq_cumsum = E_freq_cumsum / E_freq_cumsum[-1]
                else:
                    E_freq_cumsum = np.ones_like(E_freq_cumsum)
                k_low, k_high = find_freq_from_percentile(
                    E_freq_cumsum, low_percentile, high_percentile
                )
                k_low += start_freq
                k_high += start_freq

            if method == "high":
                loss = rel_spectral_bias[:, :k_high].mean()
            elif method == "mid":
                loss = rel_spectral_bias[:, k_low:k_high].mean()
            elif method == "low":
                loss = rel_spectral_bias[:, :k_low].mean()
        return loss


class FourierLoss1D(_WeightedLoss):
    """1D fourier loss
    aggregate the spectral into radial bins and then compute the loss
    Args:
        beta: the weight of the fourier loss
        log_scale: whether to use log scale for the fourier loss
    """

    def __init__(self, d=2, p=2, beta=0.05, log_scale=False):
        super().__init__()
        self.beta = beta
        self.rel_l2_loss = RelL2Norm()
        self.spectral_loss = EnergySpectrumBias1D(log_scale=log_scale)

    def forward(self, pred, target, ux_dim=1, uy_dim=2):
        fft_loss = self.spectral_loss(pred, target, ux_dim=ux_dim, uy_dim=uy_dim)
        pred_loss = self.rel_l2_loss(pred, target)
        loss = pred_loss + self.beta * fft_loss
        return loss, pred_loss, fft_loss


class FourierLoss2D(_WeightedLoss):
    """2D fourier loss
    compute the loss in the frequency domain
    Args:
        beta: the weight of the fourier loss
        log_scale: whether to use log scale for the fourier loss
    """

    def __init__(self, beta=0.05, log_scale=False):
        super().__init__()
        self.beta = beta
        self.rel_l2_loss = RelL2Norm()
        self.log_scale = log_scale

    def spectral_loss2d(self, pred, target):
        pred_fft = torch.fft.rfft2(pred)
        target_fft = torch.fft.rfft2(target)
        fft_loss = torch.abs(pred_fft - target_fft)
        if self.log_scale:
            fft_loss = torch.log(fft_loss)
        # print("fft_loss shape", fft_loss.shape)
        # print("fft_loss min", fft_loss.min(), "fft_loss max", fft_loss.max())
        fft_loss = torch.sum(fft_loss, dim=(1, 2, 3))  # sum over the height and width, time steps
        fft_loss = torch.mean(fft_loss)  # average over channels and samples
        return fft_loss

    def forward(self, pred, target):
        fft_loss = self.spectral_loss2d(pred, target)
        pred_loss = self.rel_l2_loss(pred, target)
        loss = pred_loss + self.beta * fft_loss
        return loss, pred_loss, fft_loss


class NLLLoss(_WeightedLoss):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # predict: (B, H, W, T, 2*C) with mean and std
        # target: (B, H, W, T, C)
        pred = pred.view(pred.shape[0], -1, target.shape[-1] * 2)
        target = target.view(target.shape[0], -1, target.shape[-1])
        pred_mean, pred_std = pred[:, :, : pred.shape[-1] // 2], pred[:, :, pred.shape[-1] // 2 :]

        # check if pred_std is non-negative
        if pred_std.min() < 0:
            print("pred_std is negative", pred_std.min())
            pred_std = torch.abs(pred_std)
        # pred_std = torch.exp(pred_pre_std)
        # pred_std = pred_std.clamp(1e-6, 10)
        # print("pred_mean shape", pred_mean.shape, pred_mean.min(), pred_mean.max())
        # print("pred_std shape", pred_std.shape, pred_std.min(), pred_std.max())
        # create a gaussian distribution with pred_mean and pred_std
        dist = torch.distributions.Normal(pred_mean, pred_std)
        # compute the nll loss
        nll_loss = -dist.log_prob(target)
        # print("nll_loss shape", nll_loss.shape)
        nll_loss = nll_loss.sum(dim=1) / target.shape[1]  # (B, C)
        # average over the batch size, height, width, and time
        return nll_loss.mean()


class LpLoss:
    """loss function with rel/abs Lp loss"""

    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super().__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * torch.norm(
            x.view(num_examples, -1) - y.view(num_examples, -1), self.p, 1
        )

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(
            x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1
        )
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


def legendre_gauss_weights(n, a=-1.0, b=1.0):
    r"""Helper routine which returns the Legendre-Gauss nodes and weights
    on the interval [a, b]
    """
    xlg, wlg = np.polynomial.legendre.leggauss(n)
    xlg = (b - a) * 0.5 * xlg + (b + a) * 0.5
    wlg = wlg * (b - a) * 0.5

    return xlg, wlg


class SphericalLpLoss(_WeightedLoss):
    def __init__(
        self,
        size_average=True,
        reduction=True,
        nlon=96,
        nlat=48,
        radius=1,
        quad_weights=None,
        layout="lat-lon",
        channel_last=False,
    ):
        super().__init__()
        self.size_average = size_average
        self.reduction = reduction
        self.nlon = nlon
        self.nlat = nlat
        self.radius = radius
        self.layout = layout  # 'lat-lon': spatial (nlat, nlon); 'lon-lat': (nlon, nlat)
        self.channel_last = channel_last  # True if shape is (B, H, W, C); else (B, C, H, W)
        _, quad_weights = legendre_gauss_weights(nlat, -1, 1)
        w = torch.as_tensor(quad_weights)
        if layout == "lat-lon":
            self.quad_weights = w.reshape(-1, 1)  # (nlat, 1) broadcasts over (..., nlat, nlon)
        else:
            self.quad_weights = w.reshape(1, -1)  # (1, nlat) broadcasts over (..., nlon, nlat)

    def forward(self, pred, target):
        ugrid = (pred - target) ** 2
        dlon = 2 * torch.pi / self.nlon
        radius = self.radius
        w = self.quad_weights.to(pred.device)
        if self.channel_last:
            # Spatial dims are (-3, -2); channel is (-1). Broadcast w to (1, nlat, 1, 1) or (1, 1, nlat, 1)
            if self.layout == "lat-lon":
                w = w.reshape(1, -1, 1, 1)  # (1, nlat, 1, 1) for (B, nlat, nlon, C)
            else:
                w = w.reshape(1, 1, -1, 1)  # (1, 1, nlat, 1) for (B, nlon, nlat, C)
            out = torch.sum(ugrid * w * dlon * radius**2, dim=(-3, -2))
        else:
            out = torch.sum(ugrid * w * dlon * radius**2, dim=(-2, -1))
        loss = torch.sqrt(out).mean()
        return loss


class RelLpLoss(_WeightedLoss):
    def __init__(self, d=2, p=2, component=0, regularizer=False, normalizer=None):
        super().__init__()

        self.d = d
        self.p = p
        self.component = component if component in ["all", "all-reduce"] else int(component)
        self.regularizer = regularizer
        self.normalizer = normalizer

    ### all reduce is used in temporal cases, use only one metric for all components
    def _lp_losses(self, pred, target):
        if (self.component == "all") or (self.component == "all-reduce"):
            err_pool = (
                (pred - target).view(pred.shape[0], -1, pred.shape[-1]).abs() ** self.p
            ).sum(dim=1, keepdim=False)
            target_pool = (target.view(target.shape[0], -1, target.shape[-1]).abs() ** self.p).sum(
                dim=1, keepdim=False
            )
            losses = (err_pool / target_pool) ** (1 / self.p)
            if self.component == "all":
                # metrics = losses.mean(dim=0).clone().detach().cpu().numpy()
                metrics = losses.mean(dim=0).unsqueeze(0).clone().cpu().detach().numpy()  # 1, n

            else:
                # metrics = losses.mean().clone().detach().cpu().numpy()
                metrics = losses.mean().unsqueeze(0).clone().cpu().detach().numpy()  # 1, 1

        else:
            assert self.component <= target.shape[1]
            err_pool = (
                (pred - target[..., self.component]).view(pred.shape[0], -1, pred.shape[-1]).abs()
                ** self.p
            ).sum(dim=1, keepdim=False)
            target_pool = (
                target.view(target.shape[0], -1, target.shape[-1])[..., self.component].abs()
                ** self.p
            ).sum(dim=1, keepdim=False)
            losses = (err_pool / target_pool) ** (1 / self.p)
            # metrics = losses.mean().clone().detach().cpu().numpy()
            metrics = losses.mean().unsqueeze(0).clone().cpu().detach().numpy()

        loss = losses.mean()

        return loss, metrics

    ### pred, target: B, N1, N2..., Nm, C-> B, C
    def forward(self, pred, target):
        loss, metrics = self._lp_losses(pred, target)

        ### only for computing metrics
        if self.normalizer is not None:
            ori_pred, ori_target = (
                self.normalizer.transform(pred, component=self.component, inverse=True),
                self.normalizer.transform(target, inverse=True),
            )
            _, metrics = self._lp_losses(ori_pred, ori_target)

        if self.regularizer:
            raise NotImplementedError
        else:
            reg = torch.zeros_like(loss)

        return loss, reg, metrics


class RFNELoss(_WeightedLoss):
    """RFNE(y, y_hat) = Frobenius_norm(y-y_hat) / Frobenius_norm(y)
    y: target, (batch, nx^i..., timesteps, nc)
    y_hat: prediction, (batch, nx^i..., timesteps, nc)
    """

    def forward(self, pred, target):
        dims = target.size()
        error_norm = torch.norm(pred - target, dim=dims[1:-2])
        target_norm = torch.norm(target, dim=dims[1:-2])
        return torch.mean(error_norm / target_norm)


class Evaluator(_WeightedLoss):
    def __init__(
        self, temporal=False, griddata=False, component=0, normalizer=None, ilow=4, ihigh=12
    ):
        super().__init__()

        self.component = component if component in ["all", "all-reduce"] else int(component)
        self.normalizer = normalizer
        self.temporal = temporal
        self.griddata = griddata
        self.ilow = ilow
        self.ihigh = ihigh

    ### pred, target: B, N1, N2..., Nm, C-> B, C
    def forward(self, pred, target):

        with torch.no_grad():
            ### only for computing metrics
            if self.normalizer is not None:
                pred, target = (
                    self.normalizer.transform(pred, component=self.component, inverse=True),
                    self.normalizer.transform(target, inverse=True),
                )

            if self.component not in ["all", "all-reduce"]:
                target = target[..., self.component]
                pred, target = pred.unsqueeze(-1), target.unsqueeze(-1)

            metrics = {}
            ## 1, C
            _pred, _target = (
                pred.view(pred.shape[0], -1, pred.shape[-1]),
                target.view(target.shape[0], -1, target.shape[-1]),
            )
            nmae = (
                (_pred - _target).abs().sum(dim=1, keepdim=False)
                / (_target.abs().sum(dim=1, keepdim=False))
            ).mean(dim=0, keepdim=True)
            nmse = torch.sqrt(
                ((_pred - _target) ** 2).sum(dim=1, keepdim=False)
                / ((_target) ** 2).sum(dim=1, keepdim=False)
            ).mean(dim=0, keepdim=True)
            nmxe = (
                torch.amax((_pred - _target).abs(), dim=1, keepdim=False)
                / torch.amax(_target.abs(), dim=1, keepdim=False)
            ).mean(dim=0, keepdim=True)

            metrics.update({"nmae": nmae, "nmse": nmse, "nmxe": nmxe})
            if self.temporal:
                _pred, _target = (
                    pred.view(pred.shape[0], -1, pred.shape[-2], pred.shape[-1]),
                    target.view(target.shape[0], -1, target.shape[-2], target.shape[-1]),
                )

                nmae_t = (
                    (_pred - _target).abs().sum(dim=1, keepdim=False)
                    / (_target.abs().sum(dim=1, keepdim=False))
                ).mean(dim=0, keepdim=True)
                nmse_t = torch.sqrt(
                    ((_pred - _target) ** 2).sum(dim=1, keepdim=False)
                    / (_target**2).sum(dim=1, keepdim=False)
                ).mean(dim=0, keepdim=True)
                nmxe_t = (
                    torch.amax((_pred - _target).abs(), dim=1, keepdim=False)
                    / torch.amax(_target.abs(), dim=1, keepdim=False)
                ).mean(dim=0, keepdim=True)

                metrics.update({"nmae_t": nmae_t, "nmse_t": nmse_t, "nmxe_t": nmxe_t})
            if self.griddata:
                bdmse, fmse_low, fmse_mid, fmse_high = compute_fourier_error(
                    pred, target, self.ilow, self.ihigh
                )
                metrics.update(
                    {
                        "bdmse": bdmse,
                        "fmse_low": fmse_low,
                        "fmse_mid": fmse_mid,
                        "fmse_high": fmse_high,
                    }
                )

            metrics = {key: value.cpu().numpy() for key, value in metrics.items()}
        return metrics


# adapted from Galerkin Transformer
def central_diff(x: torch.Tensor):
    # assuming PBC
    # x: (batch, seq_len, n), h is the step size, assuming n = h*w
    B, HW, T, C = x.shape
    res = int(mt.sqrt(HW))
    H = W = res
    x = rearrange(x, "b (h w) t c -> b t c h w", h=res, w=res)
    # print("x shape", x.shape)
    h = 1.0 / res
    x = x.view(B * T, C, H, W)  # reshape to [16, 3, 128, 128]
    x = F.pad(
        x, (1, 1, 1, 1), mode="circular"
    )  #  [b t*c h+2 w+2] pad height and width by 1,  pad for the last two dimensions
    x = x.view(B, T, C, H + 2, W + 2)  # reshape back to [16, 1, 3, 130, 130]

    grad_x = (x[..., 1:-1, 2:] - x[..., 1:-1, :-2]) / (2 * h)  # f(x+h) - f(x-h) / 2h
    grad_y = (x[..., 2:, 1:-1] - x[..., :-2, 1:-1]) / (2 * h)  # f(x+h) - f(x-h) / 2h

    grad_x = rearrange(grad_x, "b t c h w -> b t h w c")
    grad_y = rearrange(grad_y, "b t c h w -> b t h w c")
    return grad_x, grad_y


def compute_fourier_error(pred, target, iLow=4, iHigh=12, if_mean=False):
    # (batch, nx^i..., timesteps, nc)
    idxs = target.size()
    if len(idxs) == 4:
        pred = pred.permute(0, 3, 1, 2)
        target = target.permute(0, 3, 1, 2)
    if len(idxs) == 5:
        pred = pred.permute(0, 4, 1, 2, 3)
        target = target.permute(0, 4, 1, 2, 3)
    elif len(idxs) == 6:
        pred = pred.permute(0, 5, 1, 2, 3, 4)
        target = target.permute(0, 5, 1, 2, 3, 4)
    idxs = target.size()
    nb, nc, nt = idxs[0], idxs[1], idxs[-1]

    # RMSE
    err_mean = torch.sqrt(
        torch.mean((pred.view([nb, nc, -1, nt]) - target.view([nb, nc, -1, nt])) ** 2, dim=2)
    )
    err_RMSE = torch.mean(err_mean, axis=0)
    # print("err RMSE", err_RMSE.squeeze().detach().cpu().numpy())
    nrm = torch.sqrt(torch.mean(target.view([nb, nc, -1, nt]) ** 2, dim=2))
    err_nRMSE = torch.mean(err_mean / nrm, dim=0)
    # print("err nRMSE", err_nRMSE.squeeze().detach().cpu().numpy())

    err_CSV = torch.sqrt(
        torch.mean(
            (
                torch.sum(pred.view([nb, nc, -1, nt]), dim=2)
                - torch.sum(target.view([nb, nc, -1, nt]), dim=2)
            )
            ** 2,
            dim=0,
        )
    )
    if len(idxs) == 4:
        nx = idxs[2]
        err_CSV /= nx
    elif len(idxs) == 5:
        nx, ny = idxs[2:4]
        err_CSV /= nx * ny
    elif len(idxs) == 6:
        nx, ny, nz = idxs[2:5]
        err_CSV /= nx * ny * nz
    # worst case in all the data
    err_Max = torch.max(
        torch.max(torch.abs(pred.view([nb, nc, -1, nt]) - target.view([nb, nc, -1, nt])), dim=2)[0],
        dim=0,
    )[0]
    # print("error max", err_Max.squeeze().detach().cpu().numpy())

    if len(idxs) == 4:  # 1D
        err_BD = (pred[:, :, 0, :] - target[:, :, 0, :]) ** 2
        err_BD += (pred[:, :, -1, :] - target[:, :, -1, :]) ** 2
        err_BD = torch.mean(torch.sqrt(err_BD / 2.0), dim=0)
    elif len(idxs) == 5:  # 2D
        nx, ny = idxs[2:4]
        err_BD_x = (pred[:, :, 0, :, :] - target[:, :, 0, :, :]) ** 2
        err_BD_x += (pred[:, :, -1, :, :] - target[:, :, -1, :, :]) ** 2
        err_BD_y = (pred[:, :, :, 0, :] - target[:, :, :, 0, :]) ** 2
        err_BD_y += (pred[:, :, :, -1, :] - target[:, :, :, -1, :]) ** 2
        err_BD = (torch.sum(err_BD_x, dim=-2) + torch.sum(err_BD_y, dim=-2)) / (2 * nx + 2 * ny)
        err_BD = torch.mean(torch.sqrt(err_BD), dim=0)
    elif len(idxs) == 6:  # 3D
        nx, ny, nz = idxs[2:5]
        err_BD_x = (pred[:, :, 0, :, :] - target[:, :, 0, :, :]) ** 2
        err_BD_x += (pred[:, :, -1, :, :] - target[:, :, -1, :, :]) ** 2
        err_BD_y = (pred[:, :, :, 0, :] - target[:, :, :, 0, :]) ** 2
        err_BD_y += (pred[:, :, :, -1, :] - target[:, :, :, -1, :]) ** 2
        err_BD_z = (pred[:, :, :, :, 0] - target[:, :, :, :, 0]) ** 2
        err_BD_z += (pred[:, :, :, :, -1] - target[:, :, :, :, -1]) ** 2
        err_BD = (
            torch.sum(err_BD_x.view([nb, -1, nt]), dim=-2)
            + torch.sum(err_BD_y.view([nb, -1, nt]), dim=-2)
            + torch.sum(err_BD_z.view([nb, -1, nt]), dim=-2)
        )
        err_BD = err_BD / (2 * nx * ny + 2 * ny * nz + 2 * nz * nx)
        err_BD = torch.mean(torch.sqrt(err_BD), dim=0)

    if len(idxs) == 4:  # 1D
        nx = idxs[2]
        pred_F = torch.fft.rfft(pred, dim=2)
        target_F = torch.fft.rfft(target, dim=2)
        _err_F = (
            torch.sqrt(torch.mean(torch.abs(pred_F - target_F) ** 2, axis=0)) / nx
        )  # Lx, Ly, Lz=1
    if len(idxs) == 5:  # 2D
        pred_F = torch.fft.fftn(pred, dim=[2, 3])
        target_F = torch.fft.fftn(target, dim=[2, 3])
        nx, ny = idxs[2:4]
        _err_F = torch.abs(pred_F - target_F) ** 2
        err_F = torch.zeros([nb, nc, min(nx // 2, ny // 2), nt]).to(pred.device)
        for i in range(nx // 2):
            for j in range(ny // 2):
                it = mt.floor(mt.sqrt(i**2 + j**2))
                if it > min(nx // 2, ny // 2) - 1:
                    continue
                err_F[:, :, it] += _err_F[:, :, i, j]
        _err_F = torch.sqrt(torch.mean(err_F, axis=0)) / (nx * ny)
    elif len(idxs) == 6:  # 3D
        pred_F = torch.fft.fftn(pred, dim=[2, 3, 4])
        target_F = torch.fft.fftn(target, dim=[2, 3, 4])
        nx, ny, nz = idxs[2:5]
        _err_F = torch.abs(pred_F - target_F) ** 2
        err_F = torch.zeros([nb, nc, min(nx // 2, ny // 2, nz // 2), nt]).to(pred.device)
        for i in range(nx // 2):
            for j in range(ny // 2):
                for k in range(nz // 2):
                    it = mt.floor(mt.sqrt(i**2 + j**2 + k**2))
                    if it > min(nx // 2, ny // 2, nz // 2) - 1:
                        continue
                    err_F[:, :, it] += _err_F[:, :, i, j, k]
        _err_F = torch.sqrt(torch.mean(err_F, axis=0)) / (nx * ny * nz)

    fmse_low = torch.mean(_err_F[:, :iLow], dim=1).T  # low freq
    fmse_mid = torch.mean(_err_F[:, iLow:iHigh], dim=1).T  # middle freq
    fmse_high = torch.mean(_err_F[:, iHigh:], dim=1).T

    # err_F = torch.zeros([nc, 3, nt]).to(pred.device)
    # err_F[:, 0] += torch.mean(_err_F[:, :iLow], dim=1)  # low freq
    # err_F[:, 1] += torch.mean(_err_F[:, iLow:iHigh], dim=1)  # middle freq
    # err_F[:, 2] += torch.mean(_err_F[:, iHigh:], dim=1)  # high freq

    # if if_mean:
    #     return torch.mean(err_RMSE, dim=[0, -1]), \
    #            torch.mean(err_nRMSE, dim=[0, -1]), \
    #            torch.mean(err_CSV, dim=[0, -1]), \
    #            torch.mean(err_Max, dim=[0, -1]), \
    #            torch.mean(err_BD, dim=[0, -1]), \
    #            torch.mean(err_F, dim=[0, -1])
    # else:
    #     return err_RMSE, err_nRMSE, err_CSV, err_Max, err_BD, err_F
    return err_BD, fmse_low, fmse_mid, fmse_high  ## T, C, ### T, C


def find_freq_from_linear_fit(freq, energy, slope=-5.0 / 3):
    # find the intercept that best fit the data (freq, energy), slope is given
    # f(freq)) = slope * freq + intersept
    # error = (energy - f(freq)) ** 2
    # minimize the error by finding the intercept using convex optimization
    def objective(intercept):
        return np.sum((energy - slope * freq - intercept) ** 2)

    # use scipy.optimize.minimize to find the intercept
    result = minimize_scalar(objective)
    intercept = result.x

    # also need to get the x values of the intercept between two lines:
    # f(freq) = slope * freq + intercept and energy, use energy - f(freq) and find the roots

    # Define function to find roots: energy - (slope * freq + intercept) = 0
    def difference_func(f):
        # Interpolate energy at frequency f
        energy_interp = np.interp(f, freq, energy)
        fitted_value = slope * f + intercept
        return energy_interp - fitted_value

    # Find multiple intersection points by trying different initial guesses
    freq_min, freq_max = np.min(freq), np.max(freq)
    initial_guesses = np.linspace(freq_min, freq_max, 10)

    roots = []
    for guess in initial_guesses:
        try:
            root = fsolve(difference_func, guess)[0]
            # Check if root is valid and within bounds
            if freq_min <= root <= freq_max and abs(difference_func(root)) < 1e-6:
                # Avoid duplicate roots
                if not any(abs(root - existing_root) < 1e-3 for existing_root in roots):
                    roots.append(int(np.floor(np.exp(root))))
        except Exception:
            continue

    roots = sorted(roots)
    print(f"Intersection frequencies: {roots}")

    return roots[0], roots[1], intercept


def find_freq_from_percentile(E_freq_cumsum, low_percentile, high_percentile):
    ## return the first index that is greater than the percentile
    low_res = E_freq_cumsum > low_percentile
    # find the first index where low_res is True
    low_idx = np.argmax(low_res)
    high_res = E_freq_cumsum > high_percentile
    # find the first index where high_res is True
    high_idx = np.argmax(high_res)
    return low_idx, high_idx


# USED FOR testing evaluation
def get_frequency_bands_from_cumulative_energy_old(
    y: torch.Tensor,
    low_percentile: float = 0.67,
    high_percentile: float = 0.99,
    max_freq: int | None = None,
    eps: float = 1e-12,
) -> tuple[int, int, torch.Tensor, torch.Tensor]:
    """Determine frequency band boundaries using cumulative energy distribution.
    Returns discrete frequency bin indices for low/mid/high frequency aggregation.

    Args:
        y: Tensor of shape (N, H, W, C) - input field data.
        low_percentile: Cumulative energy fraction for low/mid boundary (default 0.33).
        high_percentile: Cumulative energy fraction for mid/high boundary (default 0.67).
        max_freq: Maximum frequency to consider (default = min(H, W)//2).
        eps: Small number to avoid divide-by-zero.

    Returns:
        k_low: int, frequency bin where cumulative energy >= low_percentile.
        k_high: int, frequency bin where cumulative energy >= high_percentile.
        freq_bins: tensor of frequency bin centers [0, 1, 2, ..., max_freq].
        cumulative_energy: tensor of cumulative energy fractions.
    """
    assert y.ndim == 5, "y must have shape (N,H,W,T,C)"
    N, H, W, T, C = y.shape
    if max_freq is None:
        max_freq = min(H, W) // 2

    device = y.device
    dtype = y.dtype

    y = rearrange(y, "b h w t c -> b t c h w")

    # Use full 2D FFT so the spectrum shape matches (B T C H, W)
    y_fft = torch.abs(torch.fft.fft2(y))

    # average over batch, time, channels
    y_fft = y_fft.mean(dim=(0, 1, 2))  # (H, W)
    # Take magnitude and move to numpy for binning
    fourier_amplitudes = y_fft.detach().cpu().numpy()

    # Create the k-frequency grid for rectangular image
    kfreq_x = np.fft.fftfreq(W) * W
    kfreq_y = np.fft.fftfreq(H) * H
    kfreq2D = np.meshgrid(kfreq_x, kfreq_y)
    knrm = np.sqrt(kfreq2D[0] ** 2 + kfreq2D[1] ** 2)

    # Flatten the arrays to use in binning (1D arrays of equal length,  (H*W,) )
    knrm = knrm.ravel()  # ALL the frequences in the image
    fourier_amplitudes = fourier_amplitudes.ravel()  # ALL the fourier amplitudes in the image

    # Define the bins for the wavenumber - use the minimum dimension for binning
    min_dim = min(H, W)
    kbins = np.arange(1, min_dim // 2 + 1, 1.0)

    # Bin the data (radial mean), turn the 2D array into 1D array
    E_freq, _, _ = stats.binned_statistic(knrm, fourier_amplitudes, statistic="mean", bins=kbins)

    log_E_freq = np.log(E_freq)
    log_freq = np.log(kbins[: len(log_E_freq)])
    k_freq = kbins[: len(E_freq)]

    # k_low, k_high, intercept = find_freq_from_linear_fit(log_freq, log_E_freq)
    # plot it temporarily
    # plt.loglog(k_freq, E_freq, 'X-',markersize=1, label='data')
    # plt.loglog(k_freq, np.exp(-5.0/3*log_freq + intercept), 'r--', label='linear fit') # linear fit
    #     # Mark intersection points
    # plt.legend()
    # plt.show()
    # k_low = 12
    # k_high = 40

    # compute the cumulative sum of the energy
    E_freq_cumsum = np.cumsum(E_freq)
    E_freq_cumsum = E_freq_cumsum / E_freq_cumsum[-1]
    k_low, k_high = find_freq_from_percentile(E_freq_cumsum, low_percentile, high_percentile)

    return k_low, k_high, k_freq, E_freq


def get_frequency_bands_from_cumulative_energy(
    y: torch.Tensor,
    low_percentile: float = 0.67,
    high_percentile: float = 0.99,
    max_freq: int | None = None,
    eps: float = 1e-12,
) -> tuple[int, int, torch.Tensor, torch.Tensor]:
    """Determine frequency band boundaries using cumulative energy distribution.
    Returns discrete frequency bin indices for low/mid/high frequency aggregation.

    Args:
        y: Tensor of shape (N, H, W, C) - input field data.
        low_percentile: Cumulative energy fraction for low/mid boundary (default 0.33).
        high_percentile: Cumulative energy fraction for mid/high boundary (default 0.67).
        max_freq: Maximum frequency to consider (default = min(H, W)//2).
        eps: Small number to avoid divide-by-zero.

    Returns:
        k_low: int, frequency bin where cumulative energy >= low_percentile.
        k_high: int, frequency bin where cumulative energy >= high_percentile.
        freq_bins: tensor of frequency bin centers [0, 1, 2, ..., max_freq].
        cumulative_energy: tensor of cumulative energy fractions.
    """
    assert y.ndim == 5, "y must have shape (N,H,W,T,C)"
    N, H, W, T, C = y.shape
    if max_freq is None:
        max_freq = min(H, W) // 2

    device = y.device
    dtype = y.dtype

    y = rearrange(y, "b h w t c -> (b t c) h w")

    # Use full 2D FFT so the spectrum shape matches (B T C H, W)
    print("y shape", y.shape)
    y_fft = torch.fft.fft2(y)

    # Take magnitude and move to numpy for binning
    # fourier_amplitudes = (torch.abs(y_fft)**2).detach().cpu().numpy()
    fourier_amplitudes = (torch.abs(y_fft) ** 2).detach().cpu().numpy()
    # Create the k-frequency grid for rectangular image
    kfreq_x = np.fft.fftfreq(W) * W
    kfreq_y = np.fft.fftfreq(H) * H
    kfreq2D = np.meshgrid(kfreq_x, kfreq_y)
    knrm = np.sqrt(kfreq2D[0] ** 2 + kfreq2D[1] ** 2)
    # Flatten the arrays to use in binning (1D arrays of equal length,  (H*W,) )
    knrm = knrm.ravel()  # ALL the frequences in the image

    # Define the bins for the wavenumber - use the minimum dimension for binning
    min_dim = min(H, W)
    kbins = np.arange(1, min_dim // 2 + 1, 1.0)

    amplitudes = []
    for idx in range(fourier_amplitudes.shape[0]):
        fourier_idx = fourier_amplitudes[idx].ravel()
        # Bin the data (radial mean), turn the 2D array into 1D array
        E_freq, _, _ = stats.binned_statistic(knrm, fourier_idx, statistic="mean", bins=kbins)
        amplitudes.append(E_freq)
    amplitudes = np.array(amplitudes)
    E_freq = amplitudes.mean(axis=0)

    log_E_freq = np.log(E_freq)
    log_freq = np.log(kbins[: len(log_E_freq)])
    # k_freq = 0.5 * (kbins[1:] + kbins[:-1])
    k_freq = kbins[: len(E_freq)]

    # k_low, k_high, intercept = find_freq_from_linear_fit(log_freq, log_E_freq)
    # plot it temporarily
    # plt.loglog(k_freq, E_freq, 'X-',markersize=1, label='data')
    # plt.loglog(k_freq, np.exp(-5.0/3*log_freq + intercept), 'r--', label='linear fit') # linear fit
    #     # Mark intersection points
    # plt.legend()
    # plt.show()
    # k_low = 12
    # k_high = 40

    # compute the cumulative sum of the energy
    E_freq_cumsum = np.cumsum(E_freq)
    E_freq_cumsum = E_freq_cumsum / E_freq_cumsum[-1]
    k_low, k_high = find_freq_from_percentile(E_freq_cumsum, low_percentile, high_percentile)

    return k_low, k_high, k_freq, E_freq


def aggregate_spectral_energy_by_bands(
    k_low: int,
    k_high: int,
    E_diff: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # E_diff: (H, W)
    # k_low: int
    # k_high: int

    # Aggregate errors by frequency bands
    low_band_error = E_diff[:k_low].mean()
    mid_band_error = E_diff[k_low:k_high].mean()
    high_band_error = E_diff[k_high:].mean()

    return low_band_error, mid_band_error, high_band_error


def spectrum_2d(
    y: torch.Tensor,
    low_percentile: float = 0.67,
    high_percentile: float = 0.99,
    max_freq: int | None = None,
    eps: float = 1e-12,
    normalize=True,
) -> tuple[int, int, torch.Tensor, torch.Tensor]:
    """This function computes the spectrum of a 2D signal using the Fast Fourier Transform (FFT).

    Paramaters
    ----------
    signal : a tensor of shape (T * n_observations * n_observations)
        A 2D discretized signal represented as a 1D tensor with shape
        (T * n_observations * n_observations), where T is the number of time
        steps and n_observations is the spatial size of the signal.

        T can be any number of channels that we reshape into and
        n_observations * n_observations is the spatial resolution.
    n_observations: an integer
        Number of discretized points. Basically the resolution of the signal.
    normalize: bool
        whether to apply normalization to the output of the 2D FFT.
        If True, normalizes the outputs by ``1/n_observations``
        (actually ``1/sqrt(n_observations * n_observations)``).

    Returns:
    --------
    spectrum: a tensor
        A 1D tensor of shape (s,) representing the computed spectrum.
        The spectrum is computed using a square approximation to radial
        binning, meaning that the wavenumber 'bin' into which a particular
        coefficient is the coefficient's location along the diagonal, indexed
        from the top-left corner of the 2d FFT output.
    """
    T = y.shape[0]
    y = rearrange(y, "b h w t c -> (b t c) h w")
    n_observations = y.shape[-1]
    signal = y.view(T, n_observations, n_observations)

    if normalize:
        signal = torch.fft.fft2(signal, norm="ortho")
    else:
        signal = torch.fft.rfft2(signal, s=(n_observations, n_observations), norm="backward")

    # 2d wavenumbers following PyTorch fft convention
    k_max = n_observations // 2
    wavenumers = torch.cat(
        (
            torch.arange(start=0, end=k_max, step=1),
            torch.arange(start=-k_max, end=0, step=1),
        ),
        0,
    ).repeat(n_observations, 1)
    k_x = wavenumers.transpose(0, 1)
    k_y = wavenumers

    # Sum wavenumbers
    sum_k = torch.abs(k_x) + torch.abs(k_y)

    # Remove symmetric components from wavenumbers
    index = -1.0 * torch.ones((n_observations, n_observations))
    k_max1 = k_max + 1
    index[0:k_max1, 0:k_max1] = sum_k[0:k_max1, 0:k_max1]
    print(index)
    spectrum = torch.zeros((T, n_observations))
    for j in range(1, n_observations + 1):
        ind = torch.where(index == j)
        spectrum[:, j - 1] = (signal[:, ind[0], ind[1]].sum(dim=1)).abs() ** 2

    E_freq = spectrum.mean(dim=0)
    E_freq = E_freq[: n_observations // 2]  # keep only the positive frequencies
    print("E_freq shape", E_freq.shape)

    min_dim = n_observations
    kbins = np.arange(1, min_dim // 2 + 1, 1.0)
    k_freq = kbins[: len(E_freq)]

    E_freq_cumsum = np.cumsum(E_freq)
    E_freq_cumsum = E_freq_cumsum / E_freq_cumsum[-1]
    k_low, k_high = find_freq_from_percentile(E_freq_cumsum, low_percentile, high_percentile)
    return k_low, k_high, k_freq, E_freq


def spectral_cfd(
    y: torch.Tensor,
    low_percentile: float = 0.67,
    high_percentile: float = 0.99,
    max_freq: int | None = None,
    eps: float = 1e-12,
) -> tuple[int, int, torch.Tensor, torch.Tensor]:
    y = rearrange(y, "b h w t c -> (b t c) h w")
    vorticity = y
    if isinstance(vorticity, np.ndarray):
        vorticity = torch.from_numpy(vorticity)
    n = vorticity.shape[-1]
    h = 2 * np.pi / n
    kx = torch.fft.fftfreq(n, d=h)
    ky = torch.fft.fftfreq(n, d=h)
    kx, ky = torch.meshgrid([kx, ky], indexing="ij")
    kmax = n // 2
    kx = kx[..., : kmax + 1]
    ky = ky[..., : kmax + 1]
    k2 = (4 * torch.pi**2) * (kx**2 + ky**2)
    print("k2 shape", k2.shape)
    k2[0, 0] = 1.0

    wh = torch.fft.rfft2(vorticity)

    tke = (torch.abs(wh) ** 2).cpu()
    kmod = torch.sqrt(k2)
    k = torch.arange(1, kmax + 1, dtype=torch.float64)  # Nyquist limit for this grid
    print("k shape", k.shape)
    dk = (torch.max(k) - torch.min(k)) / (2 * n)

    n_samples = tke.shape[0]
    E_freq = []
    for s in range(n_samples):
        Ens = torch.zeros_like(k)
        for i in range(len(k)):
            Ens[i] += (tke[s, (kmod < k[i] + dk) & (kmod >= k[i] - dk)]).sum()
        E_freq.append(Ens)
    E_freq = torch.stack(E_freq)
    E_freq = E_freq.mean(dim=0)

    n_observations = n
    E_freq = Ens
    print("E_freq shape", E_freq.shape)

    min_dim = n_observations
    kbins = np.arange(1, min_dim // 2 + 1, 1.0)
    k_freq = kbins[: len(E_freq)]

    E_freq_cumsum = np.cumsum(E_freq)
    E_freq_cumsum = E_freq_cumsum / E_freq_cumsum[-1]
    k_low, k_high = find_freq_from_percentile(E_freq_cumsum, low_percentile, high_percentile)
    return k_low, k_high, k_freq, E_freq


def _fft_wavenumbers_2d(H: int, W: int, device, dtype, Lx: float, Ly: float):
    """Returns 2D wavenumber grids kx, ky with physical scaling for a periodic domain
    of size Lx x Ly, on an HxW grid.
    """
    # torch.fft.fftfreq returns cycles per unit; multiply by 2*pi for radians per unit
    kx_1d = (2.0 * math.pi) * torch.fft.fftfreq(W, d=Lx / W, device=device)  # shape (W,)
    ky_1d = (2.0 * math.pi) * torch.fft.fftfreq(H, d=Ly / H, device=device)  # shape (H,)

    # meshgrid -> (H, W)
    ky, kx = torch.meshgrid(ky_1d, kx_1d, indexing="ij")
    return kx.to(dtype=dtype), ky.to(dtype=dtype)


def spectral_grad_2d(
    phi: torch.Tensor, Lx: float = 1.0, Ly: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Spectral gradient (periodic) for a scalar field phi on the last two dims (H, W).

    phi: (..., H, W)
    returns: (dphi_dx, dphi_dy) each (..., H, W)
    """
    H, W = phi.shape[-2], phi.shape[-1]
    device, dtype = phi.device, phi.dtype

    kx, ky = _fft_wavenumbers_2d(H, W, device, dtype, Lx, Ly)
    Phi = torch.fft.fft2(phi, dim=(-2, -1))
    dphi_dx = torch.fft.ifft2(1j * kx * Phi, dim=(-2, -1)).real
    dphi_dy = torch.fft.ifft2(1j * ky * Phi, dim=(-2, -1)).real
    return dphi_dx, dphi_dy


def spectral_laplacian_2d(phi: torch.Tensor, Lx: float = 1.0, Ly: float = 1.0) -> torch.Tensor:
    """Spectral Laplacian (periodic) for a scalar field phi on the last two dims (H, W).

    phi: (..., H, W)
    returns: laplacian(phi) with same shape
    """
    H, W = phi.shape[-2], phi.shape[-1]
    device, dtype = phi.device, phi.dtype

    kx, ky = _fft_wavenumbers_2d(H, W, device, dtype, Lx, Ly)
    k2 = kx**2 + ky**2

    Phi = torch.fft.fft2(phi, dim=(-2, -1))
    lap = torch.fft.ifft2(-k2 * Phi, dim=(-2, -1)).real
    return lap


def curl_f_to_g(
    fx: torch.Tensor, fy: torch.Tensor, Lx: float = 1.0, Ly: float = 1.0
) -> torch.Tensor:
    """Compute g = (curl f)_z = d(fy)/dx - d(fx)/dy for 2D forcing f=(fx,fy),
    using spectral derivatives (periodic).
    """
    dfy_dx, _ = spectral_grad_2d(fy, Lx=Lx, Ly=Ly)
    _, dfx_dy = spectral_grad_2d(fx, Lx=Lx, Ly=Ly)
    return dfy_dx - dfx_dy


def time_derivative(w: torch.Tensor, dt: float, scheme: str = "central") -> torch.Tensor:
    """Finite-difference time derivative along dim=1 (time).
    w: (B, T, H, W) or (T, H, W) (we'll treat first dim as batch if 4D)
    Returns dw/dt with same shape (endpoints handled by one-sided differences).
    """
    if w.dim() == 3:
        w_ = w.unsqueeze(0)  # (1,T,H,W)
        squeeze_back = True
    elif w.dim() == 4:
        w_ = w
        squeeze_back = False
    else:
        raise ValueError("w must be (T,H,W) or (B,T,H,W)")

    B, T, H, W = w_.shape
    dw = torch.empty_like(w_)

    if scheme == "central":
        # interior
        dw[:, 1:-1] = (w_[:, 2:] - w_[:, :-2]) / (2.0 * dt)
        # endpoints one-sided
        dw[:, 0] = (w_[:, 1] - w_[:, 0]) / dt
        dw[:, -1] = (w_[:, -1] - w_[:, -2]) / dt
    elif scheme == "forward":
        dw[:, :-1] = (w_[:, 1:] - w_[:, :-1]) / dt
        dw[:, -1] = dw[:, -2]
    else:
        raise ValueError("scheme must be 'central' or 'forward'")

    return dw.squeeze(0) if squeeze_back else dw


@torch.no_grad()
def vorticity_residual_spatial_2d(
    ux: torch.Tensor,
    uy: torch.Tensor,
    w: torch.Tensor,
    nu: float,
    f: torch.Tensor | None = None,
    Lx: float = 1.0,
    Ly: float = 1.0,
    return_terms: bool = False,
):
    """Compute spatial-only 2D vorticity-form NS residual (without time derivative):
        r_spatial = u·∇w - nu Δw - g
    where g = curl(f) = d(fy)/dx - d(fx)/dy

    This is useful for evaluating residual at a single time snapshot.

    Inputs
    ------
    ux : (B, H, W) or (H, W)  x-component of velocity field
    uy : (B, H, W) or (H, W)  y-component of velocity field
    w : (B, H, W) or (H, W)   vorticity field (omega)
    nu: viscosity (kinematic), scalar
    f : optional forcing (2, H, W) or (H, W, 2). If None, g=0.
    Lx,Ly: domain sizes for spectral derivatives (periodic)
    return_terms: if True, also return a dict of terms.

    Returns:
    -------
    r : same shape as w (B, H, W) or (H, W)
    (optionally terms dict)
    """
    # Handle input shapes - all should be (B, H, W) or (H, W)
    squeeze_w = False
    squeeze_ux = False
    squeeze_uy = False

    if w.dim() == 2:  # (H, W)
        w_ = w.unsqueeze(0)  # (1, H, W)
        squeeze_w = True
    elif w.dim() == 3:  # (B, H, W)
        w_ = w
    else:
        raise ValueError("w must be (H,W) or (B,H,W)")

    if ux.dim() == 2:  # (H, W)
        ux_ = ux.unsqueeze(0)
        squeeze_ux = True
    elif ux.dim() == 3:
        ux_ = ux
    else:
        raise ValueError("ux must be (H,W) or (B,H,W)")

    if uy.dim() == 2:
        uy_ = uy.unsqueeze(0)
        squeeze_uy = True
    elif uy.dim() == 3:
        uy_ = uy
    else:
        raise ValueError("uy must be (H,W) or (B,H,W)")

    # Ensure batch dimensions match
    if ux_.shape[0] != w_.shape[0] or uy_.shape[0] != w_.shape[0]:
        raise ValueError(
            f"Batch size mismatch: ux {ux_.shape[0]}, uy {uy_.shape[0]}, w {w_.shape[0]}"
        )
    if ux_.shape[-2:] != w_.shape[-2:] or uy_.shape[-2:] != w_.shape[-2:]:
        raise ValueError(
            f"Spatial size mismatch: ux {ux_.shape[-2:]}, uy {uy_.shape[-2:]}, w {w_.shape[-2:]}"
        )

    B, H, W = w_.shape

    # Compute grad w (spectral)
    dw_dx, dw_dy = spectral_grad_2d(w_, Lx=Lx, Ly=Ly)  # (B, H, W)

    # Advection term: u·∇w = ux * dw_dx + uy * dw_dy
    adv = ux_ * dw_dx + uy_ * dw_dy  # (B, H, W)

    # Laplacian w
    lapw = spectral_laplacian_2d(w_, Lx=Lx, Ly=Ly)  # (B, H, W)

    # g = curl f (if provided)
    if f is None:
        g = torch.zeros_like(w_)
    else:
        # Handle f shape: (2, H, W), (H, W, 2), or (B, H, W, 2)
        if f.dim() == 2 and f.shape[0] == 2:  # (2, H, W)
            fx = f[0].unsqueeze(0).expand(B, -1, -1)  # (B, H, W)
            fy = f[1].unsqueeze(0).expand(B, -1, -1)  # (B, H, W)
        elif f.dim() == 3 and f.shape[-1] == 2:  # (H, W, 2)
            fx = f[..., 0].unsqueeze(0).expand(B, -1, -1)  # (B, H, W)
            fy = f[..., 1].unsqueeze(0).expand(B, -1, -1)  # (B, H, W)
        elif f.dim() == 4 and f.shape[-1] == 2:  # (B, H, W, 2)
            fx = f[..., 0]  # (B, H, W)
            fy = f[..., 1]  # (B, H, W)
        else:
            raise ValueError(f"f must be (2,H,W), (H,W,2), or (B,H,W,2), got shape {f.shape}")

        g = curl_f_to_g(fx, fy, Lx=Lx, Ly=Ly)  # (B, H, W)

    # Spatial residual: u·∇w - nu Δw - g
    r = adv - float(nu) * lapw - g  # (B, H, W)

    # Squeeze batch dimension if input was 2D
    r_out = r.squeeze(0) if squeeze_w else r

    if return_terms:
        terms = {
            "advection": adv.squeeze(0) if squeeze_w else adv,
            "laplacian": lapw.squeeze(0) if squeeze_w else lapw,
            "g": g.squeeze(0) if squeeze_w else g,
        }
        return r_out, terms
    return r_out


@torch.no_grad()
def vorticity_residual_2d(
    u: torch.Tensor,
    w: torch.Tensor,
    nu: float,
    f: torch.Tensor | None = None,
    dt: float | None = None,
    Lx: float = 1.0,
    Ly: float = 1.0,
    return_terms: bool = False,
):
    """Compute 2D incompressible vorticity-form NS residual:
        r = w_t + u·∇w - nu Δw - g
    where g = curl(f) = d(fy)/dx - d(fx)/dy

    Inputs
    ------
    u : (B,T,2,H,W) or (T,2,H,W)  velocity field
    w : (B,T,H,W)   or (T,H,W)    vorticity field (omega)
    nu: viscosity (kinematic), scalar
    f : optional forcing (B,T,2,H,W) or (T,2,H,W) or (2,H,W)/(T,2,H,W). If None, g=0.
    dt: required for time derivative (unless you precompute w_t yourself and modify code)
    Lx,Ly: domain sizes for spectral derivatives (periodic)
    return_terms: if True, also return a dict of terms.

    Returns:
    -------
    r : same shape as w (B,T,H,W) or (T,H,W)
    (optionally terms dict)
    """
    # Normalize shapes to (B,T,*,H,W)
    squeeze_u = False
    squeeze_w = False

    if u.dim() == 4:  # (T,2,H,W)
        u_ = u.unsqueeze(0)
        squeeze_u = True
    elif u.dim() == 5:
        u_ = u
    else:
        raise ValueError("u must be (T,2,H,W) or (B,T,2,H,W)")

    if w.dim() == 3:  # (T,H,W)
        w_ = w.unsqueeze(0)
        squeeze_w = True
    elif w.dim() == 4:
        w_ = w
    else:
        raise ValueError("w must be (T,H,W) or (B,T,H,W)")

    if u_.shape[0] != w_.shape[0] or u_.shape[1] != w_.shape[1]:
        raise ValueError(f"Batch/time mismatch: u {u_.shape[:2]} vs w {w_.shape[:2]}")
    if dt is None:
        raise ValueError("dt is required to compute w_t")

    ux = u_[:, :, 0]  # (B,T,H,W)
    uy = u_[:, :, 1]

    # w_t
    wt = time_derivative(w_, dt=dt, scheme="central")  # (B,T,H,W)

    # grad w (spectral, per time slice)
    # We'll vectorize over (B,T) by merging dims.
    B, T, H, W = w_.shape
    w_flat = w_.reshape(B * T, H, W)
    dw_dx_flat, dw_dy_flat = spectral_grad_2d(w_flat, Lx=Lx, Ly=Ly)
    dw_dx = dw_dx_flat.reshape(B, T, H, W)
    dw_dy = dw_dy_flat.reshape(B, T, H, W)

    adv = ux * dw_dx + uy * dw_dy

    # Laplacian w
    lapw_flat = spectral_laplacian_2d(w_flat, Lx=Lx, Ly=Ly)
    lapw = lapw_flat.reshape(B, T, H, W)

    # g = curl f (if provided)
    if f is None:
        g = torch.zeros_like(w_)
    else:
        # Accept (2,H,W), (T,2,H,W), (B,T,2,H,W)
        if f.dim() == 3:  # (2,H,W)
            f_ = f.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1, -1)
        elif f.dim() == 4:  # (T,2,H,W)
            f_ = f.unsqueeze(0).expand(B, -1, -1, -1, -1)
        elif f.dim() == 5:
            f_ = f
        else:
            raise ValueError("f must be (2,H,W), (T,2,H,W), or (B,T,2,H,W)")

        fx = f_[:, :, 0].reshape(B * T, H, W)
        fy = f_[:, :, 1].reshape(B * T, H, W)
        g_flat = curl_f_to_g(fx, fy, Lx=Lx, Ly=Ly)
        g = g_flat.reshape(B, T, H, W)

    r = wt + adv - float(nu) * lapw - g

    r_out = r.squeeze(0) if (squeeze_w and squeeze_u) or squeeze_w else r

    if return_terms:
        terms = {
            "w_t": wt.squeeze(0) if squeeze_w else wt,
            "advection": adv.squeeze(0) if squeeze_w else adv,
            "laplacian": lapw.squeeze(0) if squeeze_w else lapw,
            "g": g.squeeze(0) if squeeze_w else g,
        }
        return r_out, terms
    return r_out


def rms_residual(r: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Compute RMS over all dims except batch (if present).
    r: (B,T,H,W) or (T,H,W) etc.
    mask: optional same shape as r (or broadcastable) with 0/1 for excluding points.
    """
    x = r
    if mask is not None:
        x = x * mask
        denom = mask.sum().clamp_min(1.0)
        return torch.sqrt((x**2).sum() / denom)
    return torch.sqrt((x**2).mean())


class PDEResidualLoss(_WeightedLoss):
    def __init__(self, nu=1.0 / 500, Lx=2 * np.pi, Ly=2 * np.pi, dt=1.0 / 500, f=None, H=64, W=64):
        super().__init__()
        self.nu = nu
        self.Lx = Lx
        self.Ly = Ly
        self.dt = dt

        if f is None:
            y_grid = torch.linspace(0, Ly, W)
            f_y = -4 * torch.cos(4 * y_grid)
            f_x = torch.zeros_like(f_y)
            # Store as (H, W, 2) for spatial-only version, or can be reshaped for temporal version
            f = torch.stack([f_x, f_y], dim=-1)  # (W, 2) -> but we need (H, W, 2) or (2, H, W)
            # Expand to full spatial grid if needed (assuming forcing only depends on y)
            if f.dim() == 2 and f.shape[0] == W:  # (W, 2)
                f = f.unsqueeze(0).expand(H, -1, -1)  # (H, W, 2)
        self.f = f

    def forward(self, ux: torch.Tensor, uy: torch.Tensor, w: torch.Tensor):
        """Compute PDE residual for inputs at a single time snapshot.

        Parameters
        ----------
        ux : torch.Tensor
            x-component of velocity, shape (B, H, W) or (H, W)
        uy : torch.Tensor
            y-component of velocity, shape (B, H, W) or (H, W)
        w : torch.Tensor
            vorticity field, shape (B, H, W) or (H, W)

        Returns:
        -------
        score : torch.Tensor
            RMS of spatial residual (scalar)
        """
        # Use spatial-only residual function for (B, H, W) inputs
        r = vorticity_residual_spatial_2d(
            ux,
            uy,
            w,
            nu=self.nu,
            f=self.f.to(ux.device),
            Lx=self.Lx,
            Ly=self.Ly,
            return_terms=False,
        )
        score = rms_residual(r)
        return score


class SWESphereSpectralOperators:
    """Spherical-harmonic differential operators on a lat-lon grid."""

    def __init__(self, nlat, nlon, grid="equiangular"):
        if th is None:
            raise ImportError("torch_harmonics is required for spherical SWE residuals")
        self.nlat = nlat
        self.nlon = nlon
        self.grid = grid
        self.sht = th.RealSHT(nlat, nlon, grid=grid)
        self.isht = th.InverseRealSHT(nlat, nlon, grid=grid)
        self.vsht = th.RealVectorSHT(nlat, nlon, grid=grid)
        self.ivsht = th.InverseRealVectorSHT(nlat, nlon, grid=grid)

    def to(self, device):
        self.sht = self.sht.to(device)
        self.isht = self.isht.to(device)
        self.vsht = self.vsht.to(device)
        self.ivsht = self.ivsht.to(device)
        return self

    def _ell(self, like):
        lmax = self.sht.lmax
        return torch.arange(lmax, device=like.device, dtype=like.real.dtype).view(1, lmax, 1)

    def laplacian(self, f, radius=1.0):
        coeff = self.sht(f)
        ell = self._ell(coeff)
        lap_coeff = -(ell * (ell + 1.0) / (radius**2)) * coeff
        return self.isht(lap_coeff)

    def bilaplacian(self, f, radius=1.0):
        coeff = self.sht(f)
        ell = self._ell(coeff)
        eig = ell * (ell + 1.0) / (radius**2)
        return self.isht((eig**2) * coeff)

    def grad(self, f, radius=1.0):
        coeff = self.sht(f)
        zero = torch.zeros_like(coeff)
        # Output ordering is (meridional, zonal) on the unit sphere.
        g = self.ivsht(torch.stack([coeff, zero], dim=1))
        return g / radius

    def div(self, v_meridional, v_zonal, radius=1.0):
        v = torch.stack([v_meridional, v_zonal], dim=1)
        v_coeff = self.vsht(v)
        ell = self._ell(v_coeff[:, 0])
        div_coeff = -(ell * (ell + 1.0) / radius) * v_coeff[:, 0]
        return self.isht(div_coeff)


def _swe_lat_from_grid(grid, nlat, device, dtype):
    if grid is None:
        lat = torch.linspace(-0.5 * math.pi, 0.5 * math.pi, nlat, device=device, dtype=dtype)
        return lat.view(1, nlat, 1)
    # Spherical positional grid is xyz; z = sin(lat).
    z = grid[..., 2]
    return torch.asin(torch.clamp(z, -1.0, 1.0)).view(1, nlat, -1)


def SWE_sphere_trapz_residual_step(
    x,
    y,
    grid=None,
    radius=1.0,
    Omega=0.262512,  # 7.292e-5 / (1/3600)
    nu=2.778163e-6,  # 1e5 * meter^2 / second / 32^2 in script units
    g=5.650774,  # 9.80616 * meter / second in script units
    H=0.001569557,  # 1e4 * meter in script units
    dt=1.0,  # snapshots saved every 1 hour
    h_channel=0,
    u_zonal_channel=1,
    u_meridional_channel=2,
    op=None,
):
    """One-step trapezoidal residuals for viscous SWE on the sphere.

    x, y: (B, nlat, nlon, C) at t_n and t_{n+1}, respectively.
    """
    if x.shape != y.shape:
        raise ValueError(f"x and y must have same shape, got {x.shape} and {y.shape}")
    if x.shape[-1] <= max(h_channel, u_zonal_channel, u_meridional_channel):
        raise ValueError("Input channels do not contain required SWE fields")

    B, nlat, nlon, _ = x.shape
    device = x.device
    dtype = x.dtype
    if op is None:
        op = SWESphereSpectralOperators(nlat, nlon, grid="equiangular").to(device)
    else:
        op = op.to(device)

    h_n = x[..., h_channel]
    u_lon_n = x[..., u_zonal_channel]
    u_lat_n = x[..., u_meridional_channel]
    h_np1 = y[..., h_channel]
    u_lon_np1 = y[..., u_zonal_channel]
    u_lat_np1 = y[..., u_meridional_channel]

    lat = _swe_lat_from_grid(grid, nlat, device=device, dtype=dtype)
    coriolis = 2.0 * Omega * torch.sin(lat)

    def rhs(h, u_lon, u_lat):
        grad_h = op.grad(h, radius=radius)
        grad_h_lat, grad_h_lon = grad_h[:, 0], grad_h[:, 1]

        grad_u_lon = op.grad(u_lon, radius=radius)
        grad_u_lat = op.grad(u_lat, radius=radius)
        du_lon_dlat, du_lon_dlon = grad_u_lon[:, 0], grad_u_lon[:, 1]
        du_lat_dlat, du_lat_dlon = grad_u_lat[:, 0], grad_u_lat[:, 1]

        adv_lon = u_lat * du_lon_dlat + u_lon * du_lon_dlon
        adv_lat = u_lat * du_lat_dlat + u_lon * du_lat_dlon

        bih_u_lon = op.bilaplacian(u_lon, radius=radius)
        bih_u_lat = op.bilaplacian(u_lat, radius=radius)
        bih_h = op.bilaplacian(h, radius=radius)

        div_u = op.div(u_lat, u_lon, radius=radius)
        div_hu = op.div(h * u_lat, h * u_lon, radius=radius)

        # zcross(u) in local (zonal, meridional) components -> (-u_lat, u_lon)
        cor_lon = coriolis * (-u_lat)
        cor_lat = coriolis * (u_lon)

        rhs_u_lon = -(nu * bih_u_lon + g * grad_h_lon + cor_lon + adv_lon)
        rhs_u_lat = -(nu * bih_u_lat + g * grad_h_lat + cor_lat + adv_lat)
        rhs_h = -(nu * bih_h + H * div_u + div_hu)
        return rhs_h, rhs_u_lon, rhs_u_lat

    rhs_h_n, rhs_u_lon_n, rhs_u_lat_n = rhs(h_n, u_lon_n, u_lat_n)
    rhs_h_np1, rhs_u_lon_np1, rhs_u_lat_np1 = rhs(h_np1, u_lon_np1, u_lat_np1)

    res_h = (h_np1 - h_n) - 0.5 * dt * (rhs_h_n + rhs_h_np1)
    res_u_lon = (u_lon_np1 - u_lon_n) - 0.5 * dt * (rhs_u_lon_n + rhs_u_lon_np1)
    res_u_lat = (u_lat_np1 - u_lat_n) - 0.5 * dt * (rhs_u_lat_n + rhs_u_lat_np1)
    return res_h, res_u_lon, res_u_lat


def SWE_sphere_trapz_residual_metrics(x, y, grid=None, op=None, **kwargs):
    """Return absolute and relative RMS residual metrics for one-step spherical SWE."""
    r_h, r_u_lon, r_u_lat = SWE_sphere_trapz_residual_step(x, y, grid=grid, op=op, **kwargs)
    # Average over batch and spatial points.
    m_h = torch.sqrt(torch.mean(r_h**2))
    m_u_lon = torch.sqrt(torch.mean(r_u_lon**2))
    m_u_lat = torch.sqrt(torch.mean(r_u_lat**2))
    h_channel = kwargs.get("h_channel", 0)
    u_zonal_channel = kwargs.get("u_zonal_channel", 1)
    u_meridional_channel = kwargs.get("u_meridional_channel", 2)
    eps = kwargs.get("eps", 1e-8)

    dh = y[..., h_channel] - x[..., h_channel]
    du_lon = y[..., u_zonal_channel] - x[..., u_zonal_channel]
    du_lat = y[..., u_meridional_channel] - x[..., u_meridional_channel]
    dh_ref = torch.sqrt(torch.mean(dh**2))
    du_lon_ref = torch.sqrt(torch.mean(du_lon**2))
    du_lat_ref = torch.sqrt(torch.mean(du_lat**2))

    m_h_rel = m_h / (dh_ref + eps)
    m_u_lon_rel = m_u_lon / (du_lon_ref + eps)
    m_u_lat_rel = m_u_lat / (du_lat_ref + eps)
    return m_h, m_u_lon, m_u_lat, m_h_rel, m_u_lon_rel, m_u_lat_rel


# --------------------------
# Example usage
# --------------------------
# u: (B,T,2,H,W), w: (B,T,H,W), f: (B,T,2,H,W) or None
# nu = 1e-3
# dt = 0.01
# r_w, terms = vorticity_residual_2d(u, w, nu=nu, f=f, dt=dt, Lx=1.0, Ly=1.0, return_terms=True)
# score = rms_residual(r_w)  # scalar


if __name__ == "__main__":
    # x = torch.randn([2, 128, 128, 1, 3])
    # y = torch.randn([2, 128, 128, 1, 3])
    x = torch.randn([2, 6, 6, 1, 3])
    y = torch.randn([2, 6, 6, 1, 1])
    spectrum = spectrum_2d(y)
    # evaluator = Evaluator(temporal=True, griddata=True, component='all', normalizer=None)
    # metrics = evaluator(x, y)
    # print(metrics)
    # myloss = FourierLoss(beta=1)
    # Test the new cumulative energy-based frequency band selection
    print("Testing cumulative energy-based frequency aggregation...")

    # Create test data (N, H, W, C)
    torch.manual_seed(42)
    N, H, W, C = 2, 128, 128, 1
    target = torch.randn([N, H, W, 1, C])
    pred = target + 0.1 * torch.randn_like(target)  # Add some noise

    print(f"Data shape: {target.shape}")
    print(f"Max frequency: {min(H, W) // 2}")

    # Get frequency bands using cumulative energy
    k_low, k_mid, k_high, E_bins_target = get_frequency_bands_from_cumulative_energy(
        target, low_percentile=0.70, high_percentile=0.99
    )
    _, _, _, E_bins_pred = get_frequency_bands_from_cumulative_energy(
        pred, low_percentile=0.80, high_percentile=0.99
    )

    # Aggregate spectral errors by frequency bands
    low_err, mid_err, high_err = aggregate_spectral_energy_by_bands(
        k_low, k_high, np.abs(E_bins_pred - E_bins_target)
    )

    print("\nSpectral error by frequency bands:")
    print(f"Low frequency error (0 to {k_low}): {low_err:.6f}")
    print(f"Mid frequency error ({k_low} to {k_high}): {mid_err:.6f}")
    print(f"High frequency error ({k_high}+): {high_err:.6f}")

    print("Test completed successfully!")
    # for key, value in metrics.items():
    #     print(key, value.shape)
