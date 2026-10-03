"""Tests for the energy balance enforcement in SpectralDivergenceFreeProjector2D."""

import functools
import math
import unittest

import torch

from models.layers import SpectralDivergenceFreeProjector2D


def _make_projector(energy_balance=True, zero_mean=True, **kwargs):
    """Create a projector with Kolmogorov-flow defaults."""
    defaults = {
        "domain_lengths": (2 * math.pi, 2 * math.pi),
        "zero_mean": zero_mean,
        "energy_balance": energy_balance,
        "viscosity": 1.0 / 500,
        "friction": 0.025,
        "dt": 0.015625,
    }
    defaults.update(kwargs)
    return defaults


def _random_divfree_field(B, N, domain_lengths=(2 * math.pi, 2 * math.pi)):
    """Generate a random divergence-free velocity field on [0,2pi]^2."""
    proj = SpectralDivergenceFreeProjector2D(
        domain_lengths=domain_lengths,
        zero_mean=True,
    )
    u = torch.randn(B, 2, N, N)
    return proj(u)


def _kolmogorov_forcing(N, domain_lengths=(2 * math.pi, 2 * math.pi)):
    """Compute the diagonal Kolmogorov forcing of the dataset for an NxN grid."""
    from utils.criterion import get_forcing_vel

    f = get_forcing_vel(
        N,
        amplitude=2 * math.sqrt(2),
        wavenumber=4.0,
        phase=5 * math.pi / 4,
        domain_length=domain_lengths[0],
    )
    return f.squeeze(-1)  # (1, 2, N, N)


def _kolmogorov_forcing_hat(N, domain_lengths=(2 * math.pi, 2 * math.pi)):
    """Compute Kolmogorov forcing FFT for an NxN grid."""
    return torch.fft.fft2(_kolmogorov_forcing(N, domain_lengths), dim=(-2, -1))


def _energy_balance_residual(
    u_new, u_old, f_hat, viscosity, friction, dt, domain_lengths, subgrid_source=0.0
):
    """Recompute the energy balance residual h from scratch (physical space).

    ``subgrid_source`` is the constant closure for the cross-cutoff inflow: the balance
    it closes is ``dE/dt = -nu Z - 2 alpha E + P + S``, so the residual carries ``-dt*S``.
    """
    Nx, Ny = u_new.shape[-2], u_new.shape[-1]
    N2 = Nx * Ny
    area = domain_lengths[0] * domain_lengths[1]
    w = area / N2

    def E(v):
        return 0.5 * w * (v**2).sum(dim=(-3, -2, -1))

    # Enstrophy via Fourier: Z = (w/N^2) sum_k |k|^2 |v_hat|^2
    # Wavenumbers match SpectralDifferentiationBase: k = 2*pi * fftfreq(N, d=L/N)
    dx = domain_lengths[0] / Nx
    dy = domain_lengths[1] / Ny
    kx = (
        2 * math.pi * torch.fft.fftfreq(Nx, d=dx, device=u_new.device, dtype=u_new.dtype)
    ).reshape(Nx, 1)
    ky = (
        2 * math.pi * torch.fft.fftfreq(Ny, d=dy, device=u_new.device, dtype=u_new.dtype)
    ).reshape(1, Ny)
    k_sq = kx**2 + ky**2

    def Z(v):
        v_hat = torch.fft.fft2(v, dim=(-2, -1))
        return (w / N2) * (k_sq * v_hat.abs().square()).sum(dim=(-3, -2, -1))

    def Inj(v):
        v_hat = torch.fft.fft2(v, dim=(-2, -1))
        return (w / N2) * (v_hat.conj() * f_hat).sum(dim=(-3, -2, -1)).real

    E_old, E_new = E(u_old), E(u_new)
    Z_old, Z_new = Z(u_old), Z(u_new)
    Inj_old, Inj_new = Inj(u_old), Inj(u_new)

    h = (
        E_new
        - E_old
        + (dt / 2)
        * (viscosity * (Z_old + Z_new) + 2 * friction * (E_old + E_new) - (Inj_old + Inj_new))
        - dt * subgrid_source
    )
    return h


class SpectralEnergyBalanceTest(unittest.TestCase):
    def test_backward_compat_forward_without_u_old(self):
        """Calling forward(u) without u_old returns the same div-free result."""
        N, B = 32, 2
        proj_no_energy = SpectralDivergenceFreeProjector2D(
            domain_lengths=(2 * math.pi, 2 * math.pi),
            zero_mean=True,
        )
        proj_with_energy = SpectralDivergenceFreeProjector2D(
            **_make_projector(),
        )
        u = torch.randn(B, 2, N, N)

        out_no = proj_no_energy(u)
        out_with = proj_with_energy(u)  # u_old defaults to None

        torch.testing.assert_close(out_no, out_with, atol=1e-6, rtol=1e-6)

    def test_residual_driven_to_zero(self):
        """After projection, the energy balance residual |h| < tol."""
        N, B = 32, 4
        cfg = _make_projector(max_iterations=5, residual_tol=1e-10)
        proj = SpectralDivergenceFreeProjector2D(**cfg)
        proj.set_forcing(_kolmogorov_forcing(N))

        u_old = _random_divfree_field(B, N)
        u_new = _random_divfree_field(B, N)

        u_proj = proj(u_new, u_old=u_old)

        f_hat = _kolmogorov_forcing_hat(N)
        h = _energy_balance_residual(
            u_proj,
            u_old,
            f_hat,
            cfg["viscosity"],
            cfg["friction"],
            cfg["dt"],
            cfg["domain_lengths"],
        )
        # Tolerance is 1e-4 rather than machine precision because the
        # external verification re-FFTs from physical space, which
        # includes Nyquist modes that were zeroed inside the projector.
        self.assertTrue(
            (h.abs() < 1e-4).all(),
            f"Energy residual too large: max |h| = {h.abs().max().item():.2e}",
        )

    def test_closed_residual_driven_to_zero(self):
        """With a subgrid source, the CLOSED residual |h - dt*S| is what goes to zero.

        ``source`` is far larger than the 3.2e-04 the E3 arm carries, and deliberately so:
        the external verification re-FFTs from physical space in float32, which puts a
        floor of about 1e-4 under any residual it can measure. A source whose ``dt*S``
        sits below that floor is invisible to this test, and the test then passes with
        the closure deleted. At ``dt = 0.015625`` a source of 6.4 puts ``dt*S`` at 0.1,
        three orders of magnitude clear of the floor, so the two residuals separate.
        """
        N, B = 32, 4
        torch.manual_seed(20260901)
        source = 6.4
        cfg = _make_projector(max_iterations=20, residual_tol=1e-12, subgrid_source=source)
        proj = SpectralDivergenceFreeProjector2D(**cfg)
        proj.set_forcing(_kolmogorov_forcing(N))

        u_old = _random_divfree_field(B, N)
        u_new = _random_divfree_field(B, N)

        u_proj = proj(u_new, u_old=u_old)

        f_hat = _kolmogorov_forcing_hat(N)
        residual = functools.partial(
            _energy_balance_residual,
            u_proj,
            u_old,
            f_hat,
            cfg["viscosity"],
            cfg["friction"],
            cfg["dt"],
            cfg["domain_lengths"],
        )
        offset = cfg["dt"] * source
        closed = residual(subgrid_source=source)
        unclosed = residual(subgrid_source=0.0)
        self.assertTrue(
            (closed.abs() < 1e-4).all(),
            f"Closed residual too large: max |h| = {closed.abs().max().item():.2e}",
        )
        # The layer solved the closed equation, not the un-closed one: what it leaves
        # standing in the un-closed residual is the whole constant. Asserted on the
        # layer's own output, so deleting the constant from the layer fails it.
        torch.testing.assert_close(
            unclosed,
            torch.full_like(unclosed, offset),
            atol=1e-4,
            rtol=1e-3,
        )

    def test_zero_subgrid_source_leaves_the_layer_alone(self):
        """The default is 0.0, and 0.0 changes nothing while a non-zero source does.

        The plan asked that ``subgrid_source: 0`` reproduce the pre-change layer bitwise.
        That holds because the only new arithmetic is ``h - dt*S``, and ``x - 0.0`` is
        exact in IEEE 754; it was also checked out of band against cee07f2a, where the
        two trees agree to max abs diff 0.0 in float32 and float64. What this test can
        pin from inside one tree is the pair either side of that claim: the default is
        0.0, passing 0.0 explicitly is bit for bit the same, and a non-zero source is not.
        """
        N, B = 32, 3
        torch.manual_seed(20260901)
        u_old = _random_divfree_field(B, N)
        u_new = _random_divfree_field(B, N)

        default = SpectralDivergenceFreeProjector2D(**_make_projector(max_iterations=20))
        explicit_zero = SpectralDivergenceFreeProjector2D(
            **_make_projector(max_iterations=20, subgrid_source=0.0)
        )
        closed = SpectralDivergenceFreeProjector2D(
            **_make_projector(max_iterations=20, subgrid_source=6.4)
        )
        self.assertEqual(default.subgrid_source, 0.0)
        for proj in (default, explicit_zero, closed):
            proj.set_forcing(_kolmogorov_forcing(N))

        baseline = default(u_new, u_old=u_old)
        self.assertTrue(
            torch.equal(baseline, explicit_zero(u_new, u_old=u_old)),
            "a zero subgrid source must leave the existing layer bitwise unchanged",
        )
        self.assertFalse(
            torch.equal(baseline, closed(u_new, u_old=u_old)),
            "a non-zero subgrid source must move the layer, or the flag is not wired in",
        )

    def test_divergence_preserved_after_energy_correction(self):
        """Output is both divergence-free AND energy-balanced."""
        N, B = 32, 2
        proj = SpectralDivergenceFreeProjector2D(**_make_projector())
        proj.set_forcing(_kolmogorov_forcing(N))

        u_old = _random_divfree_field(B, N)
        u_new = _random_divfree_field(B, N)

        u_proj = proj(u_new, u_old=u_old)

        # Check divergence
        Lx, Ly = 2 * math.pi, 2 * math.pi
        kx = (2 * math.pi / Lx) * torch.fft.fftfreq(N).reshape(N, 1)
        ky = (2 * math.pi / Ly) * torch.fft.fftfreq(N).reshape(1, N)
        u_hat = torch.fft.fft2(u_proj, dim=(-2, -1))
        div_hat = 1j * kx * u_hat[:, 0] + 1j * ky * u_hat[:, 1]
        div = torch.fft.ifft2(div_hat).real
        max_div = div.abs().max().item()
        self.assertLess(
            max_div, 1e-5, f"Divergence too large after energy correction: {max_div:.2e}"
        )

    def test_zero_mean_preserved(self):
        """With zero_mean=True, energy correction preserves zero spatial mean."""
        N, B = 32, 2
        proj = SpectralDivergenceFreeProjector2D(**_make_projector(zero_mean=True))
        proj.set_forcing(_kolmogorov_forcing(N))

        u_old = _random_divfree_field(B, N)
        u_new = _random_divfree_field(B, N)

        u_proj = proj(u_new, u_old=u_old)
        mean_vel = u_proj.mean(dim=(-2, -1))
        self.assertLess(
            mean_vel.abs().max().item(),
            1e-6,
            "Spatial mean velocity should be zero after projection",
        )

    def test_idempotent_on_balanced_pair(self):
        """If u_old, u_new already satisfy h=0, projector returns u_new unchanged."""
        N, B = 32, 2
        cfg = _make_projector()
        proj = SpectralDivergenceFreeProjector2D(**cfg)
        proj.set_forcing(_kolmogorov_forcing(N))

        # First project to get a pair that satisfies the constraint
        u_old = _random_divfree_field(B, N)
        u_new_raw = _random_divfree_field(B, N)
        u_new_balanced = proj(u_new_raw, u_old=u_old)

        # Second projection should be a no-op
        u_new_again = proj(u_new_balanced, u_old=u_old)
        torch.testing.assert_close(
            u_new_balanced,
            u_new_again,
            atol=1e-6,
            rtol=1e-6,
        )

    def test_forcing_buffer_device_transfer(self):
        """Forcing buffer follows module device transfers."""
        N = 16
        proj = SpectralDivergenceFreeProjector2D(**_make_projector())
        proj.set_forcing(_kolmogorov_forcing(N))

        self.assertIsNotNone(proj.forcing_hat)
        self.assertEqual(proj.forcing_hat.shape, (1, 2, N, N))

        # Move to same device (cpu) — should not error
        proj = proj.to("cpu")
        self.assertEqual(proj.forcing_hat.device.type, "cpu")

    def test_no_energy_correction_without_forcing(self):
        """If forcing is not set, energy balance is skipped (div-free only)."""
        N, B = 32, 2
        proj = SpectralDivergenceFreeProjector2D(**_make_projector())
        # Do NOT call set_forcing

        u_old = _random_divfree_field(B, N)
        u_new = torch.randn(B, 2, N, N)

        # Should not raise, should just do div-free projection
        u_proj = proj(u_new, u_old=u_old)

        # Verify it's div-free
        kx = torch.fft.fftfreq(N).reshape(N, 1) * 2 * math.pi
        ky = torch.fft.fftfreq(N).reshape(1, N) * 2 * math.pi
        u_hat = torch.fft.fft2(u_proj, dim=(-2, -1))
        div = torch.fft.ifft2(1j * kx * u_hat[:, 0] + 1j * ky * u_hat[:, 1]).real
        self.assertLess(div.abs().max().item(), 1e-5)


if __name__ == "__main__":
    unittest.main()
