"""Residual tests: the recorded velocity-path baseline and manufactured solutions."""

import math
import unittest

import torch

from utils.criterion import (
    FDM_NS_vorticity,
    INT_NS_vorticity,
    PINO_loss3d,
    PINO_loss3d_physics,
    PINO_loss3d_vel,
    ResidualScales,
    build_forcing,
    build_forcing_for_data,
    get_forcing_vel,
    physics_from_config,
    residual_has_scale,
    residual_scales_from_truth,
    velocity_from_vorticity,
)


def _physics(**overrides):
    """Physics for the Kolmogorov dataset, with per-test overrides."""
    data_config = {
        "nu": 1.0 / 500,
        "alpha": 1e-8,
        "dt": 0.015625,
        "domain_length": 2 * math.pi,
        "nx": 64,
        "ny": 64,
        "formulation": "velocity",
        "forcing": KOLMOGOROV_DIAG,
    }
    data_config.update(overrides)
    return physics_from_config(data_config)


def _forcing_k_squared(physics):
    """|k|^2 of the single mode the forcing drives."""
    k = 2 * math.pi * float(physics.forcing["wavenumber"]) / physics.domain_length
    return 2 * k**2 if physics.forcing["type"] == "kolmogorov_diag" else k**2


def _steady_velocity_state(physics, forcing, nt):
    """The laminar steady state u = f / (nu*|k|^2 + alpha), p = 0, repeated nt times.

    Both forcings here are single-mode fields with u.grad(u) = 0, so this is an
    exact solution of the momentum equation with zero pressure.
    """
    k_sq = _forcing_k_squared(physics)
    u_amplitude = forcing[0, :, :, :, 0] / (physics.nu * k_sq + physics.alpha)
    state = torch.zeros(1, 3, forcing.shape[2], forcing.shape[3], nt, dtype=forcing.dtype)
    state[:, :2] = u_amplitude.unsqueeze(0).unsqueeze(-1)
    return state


# Physics of the Kolmogorov dataset, as hard-coded in criterion.py before issue #15.
NU = 1.0 / 500
ALPHA = 1e-8
DT = 0.015625
L = 2 * math.pi

KOLMOGOROV_DIAG = {
    "type": "kolmogorov_diag",
    "amplitude": 2 * math.sqrt(2),
    "wavenumber": 4.0,
    "phase": 5 * math.pi / 4,
}
PINO_COS4Y = {"type": "pino_cos4y", "amplitude": 1.0, "wavenumber": 4.0}
NO_FORCING = {"type": "none"}

# Recorded on origin/main (commit 18e4031) with torch.manual_seed(0),
# u = torch.randn(2, 3, 32, 32, 3, dtype=torch.float64), u0 = u[..., 0],
# forcing = get_forcing_vel(32); see the issue #15 PR for the recording script.
# The two continuity entries were re-recorded for issue #16, which scores
# continuity on the predicted time levels only: 154.12815551221746 ->
# 153.29866334227907 and 1.000837964950793 -> 0.998141159649627. The momentum
# entries are unchanged, the scales here being measured on u itself.
BASELINE = {
    "integral": {
        "loss_ic": 0.0,
        "loss_cont": 153.29866334227907,
        "loss_momx": 1.9712803946306297,
        "loss_momy": 2.0123191800450764,
        "loss_cont_rel": 0.998141159649627,
        "loss_momx_rel": 1.001682739090838,
        "loss_momy_rel": 1.0010707712180982,
    },
    "fdm": {
        "loss_ic": 0.0,
        "loss_cont": 153.29866334227907,
        "loss_momx": 8644.3329586708296,
        "loss_momy": 8714.6997368789471,
        "loss_cont_rel": 0.998141159649627,
        "loss_momx_rel": 1.0104036609226057,
        "loss_momy_rel": 1.014707891188763,
    },
}
LOSS_NAMES = [
    "loss_ic",
    "loss_cont",
    "loss_momx",
    "loss_momy",
    "loss_cont_rel",
    "loss_momx_rel",
    "loss_momy_rel",
]


class VelocityResidualRegressionTest(unittest.TestCase):
    def test_old_defaults_reproduce_recorded_losses(self):
        torch.manual_seed(0)
        S = 32
        u = torch.randn(2, 3, S, S, 3, dtype=torch.float64)
        u0 = u[..., 0]
        forcing = get_forcing_vel(
            S,
            amplitude=2 * math.sqrt(2),
            wavenumber=4.0,
            phase=5 * math.pi / 4,
            domain_length=L,
        ).to(torch.float64)

        for method, expected in BASELINE.items():
            losses = PINO_loss3d_vel(
                u,
                u0,
                forcing,
                nu=NU,
                alpha=ALPHA,
                t_interval=DT,
                domain_length=L,
                scales=residual_scales_from_truth(u, _physics(), DT, time_method=method),
                time_method=method,
                return_relative=True,
            )
            for name, value in zip(LOSS_NAMES, losses, strict=True):
                with self.subTest(method=method, loss=name):
                    self.assertAlmostEqual(
                        value.item(), expected[name], delta=1e-12 * max(1.0, expected[name])
                    )


class VelocityManufacturedSolutionTest(unittest.TestCase):
    """The velocity residual vanishes on exact solutions of the configured physics."""

    def _assert_steady_state_is_a_solution(self, physics):
        S = physics.nx
        forcing = build_forcing(physics, S, dtype=torch.float64)
        state = _steady_velocity_state(physics, forcing, nt=3)

        _, _, loss_momx, loss_momy, loss_cont_rel, _, _ = PINO_loss3d_vel(
            state,
            state[..., 0],
            forcing,
            nu=physics.nu,
            alpha=physics.alpha,
            t_interval=2 * physics.dt,
            domain_length=physics.domain_length,
            scales=ResidualScales(cont=1.0, momx=1.0, momy=1.0),
            return_relative=True,
        )

        # The momentum deficit is dt*f in size, so scale the residual by it.
        momentum_scale = physics.dt * forcing.square().mean().sqrt().item()
        self.assertLess(loss_cont_rel.item(), 1e-10)
        self.assertLess(math.sqrt(loss_momx.item()) / momentum_scale, 1e-10)
        self.assertLess(math.sqrt(loss_momy.item()) / momentum_scale, 1e-10)

    def test_steady_kolmogorov_diag_solution_has_no_residual(self):
        self._assert_steady_state_is_a_solution(_physics(forcing=KOLMOGOROV_DIAG))

    def test_steady_pino_cos4y_solution_has_no_residual(self):
        self._assert_steady_state_is_a_solution(_physics(forcing=PINO_COS4Y))

    def test_taylor_green_residual_is_second_order_in_dt(self):
        """With no forcing the trapezoidal residual falls as dt^2 relative to du."""
        physics = _physics(forcing=NO_FORCING)
        S = physics.nx
        forcing = build_forcing(physics, S, dtype=torch.float64)
        decay = 2 * physics.nu + physics.alpha

        grid = torch.tensor(
            [2 * math.pi * i / S for i in range(S)],
            dtype=torch.float64,
        )
        x = grid.reshape(S, 1)
        y = grid.reshape(1, S)

        def taylor_green(dt, nt=3):
            state = torch.zeros(1, 3, S, S, nt, dtype=torch.float64)
            for n in range(nt):
                envelope = math.exp(-decay * n * dt)
                state[0, 0, :, :, n] = torch.sin(x) * torch.cos(y) * envelope
                state[0, 1, :, :, n] = -torch.cos(x) * torch.sin(y) * envelope
                # p = +1/4 (cos 2x + cos 2y): the sign that balances u.grad(u)
                # for this velocity convention (u = sin x cos y).
                state[0, 2, :, :, n] = 0.25 * (torch.cos(2 * x) + torch.cos(2 * y)) * envelope**2
            return state

        def momentum_residual(dt):
            state = taylor_green(dt)
            losses = PINO_loss3d_vel(
                state,
                state[..., 0],
                forcing,
                nu=physics.nu,
                alpha=physics.alpha,
                t_interval=2 * dt,
                domain_length=physics.domain_length,
                scales=residual_scales_from_truth(state, physics, 2 * dt),
                return_relative=True,
            )
            return losses[5].item()

        coarse = momentum_residual(0.1)
        fine = momentum_residual(0.05)
        self.assertLess(fine, coarse)
        self.assertAlmostEqual(coarse / fine, 4.0, delta=0.5)


class VorticityManufacturedSolutionTest(unittest.TestCase):
    """The vorticity residual vanishes on exact solutions, at 64^2 with 4x refinement."""

    def test_steady_solutions_have_no_residual(self):
        """w = f / (nu*|k|^2 + alpha) solves the vorticity equation: u.grad(w) = 0."""
        for forcing_cfg in (PINO_COS4Y, KOLMOGOROV_DIAG):
            for time_method in ("fdm", "integral"):
                with self.subTest(forcing=forcing_cfg["type"], time_method=time_method):
                    physics = _physics(formulation="vorticity", forcing=forcing_cfg)
                    S = physics.nx
                    upsample = physics.residual_upsample

                    forcing_coarse = build_forcing(physics, S, dtype=torch.float64)
                    scale = physics.nu * _forcing_k_squared(physics) + physics.alpha
                    w = (forcing_coarse / scale).repeat(1, 1, 1, 3)

                    forcing_fine = build_forcing(physics, upsample * S, dtype=torch.float64)
                    _, loss_f = PINO_loss3d(
                        w,
                        w[..., 0],
                        forcing_fine,
                        nu=physics.nu,
                        alpha=physics.alpha,
                        t_interval=2 * physics.dt,
                        domain_length=physics.domain_length,
                        time_method=time_method,
                        upsample=upsample,
                    )
                    self.assertLess(loss_f.item(), 1e-10)

    def test_the_paper_forcing_fixes_the_sign_of_the_steady_vorticity(self):
        """w = -cos(4y)/(4 nu): negative where the forcing -4 cos(4y) is negative."""
        physics = _physics(formulation="vorticity", forcing=PINO_COS4Y)
        forcing = build_forcing(physics, physics.nx, dtype=torch.float64)
        w = forcing / (physics.nu * _forcing_k_squared(physics) + physics.alpha)
        self.assertAlmostEqual(w[0, 0, 0, 0].item(), -1.0 / (4 * physics.nu), places=4)

    def test_taylor_green_vorticity_residual_is_second_order_in_dt(self):
        physics = _physics(formulation="vorticity", forcing=NO_FORCING)
        S = physics.nx
        decay = 2 * physics.nu + physics.alpha

        grid = torch.tensor([2 * math.pi * i / S for i in range(S)], dtype=torch.float64)
        x = grid.reshape(S, 1)
        y = grid.reshape(1, S)

        def residual_rms(dt, residual_fn, nt=3):
            w = torch.zeros(1, S, S, nt, dtype=torch.float64)
            for n in range(nt):
                w[0, :, :, n] = 2 * torch.sin(x) * torch.sin(y) * math.exp(-decay * n * dt)
            residual = residual_fn(
                w,
                nu=physics.nu,
                alpha=physics.alpha,
                t_interval=(nt - 1) * dt,
                domain_length=physics.domain_length,
            )
            # The trapezoidal residual is an increment; scale it to a rate to compare orders.
            scale = dt if residual_fn is INT_NS_vorticity else 1.0
            return residual.square().mean().sqrt().item() / scale

        for residual_fn in (FDM_NS_vorticity, INT_NS_vorticity):
            with self.subTest(residual=residual_fn.__name__):
                coarse = residual_rms(0.4, residual_fn)
                fine = residual_rms(0.2, residual_fn)
                self.assertLess(fine, coarse)
                self.assertAlmostEqual(coarse / fine, 4.0, delta=0.5)


class VorticityUnforcedGuardTest(unittest.TestCase):
    def test_zero_forcing_is_rejected_rather_than_scored_as_nan(self):
        physics = _physics(formulation="vorticity", forcing=NO_FORCING)
        forcing = build_forcing(physics, physics.nx * physics.residual_upsample)
        w = torch.randn(1, physics.nx, physics.ny, 2)
        with self.assertRaises(ValueError):
            PINO_loss3d(
                w,
                w[..., 0],
                forcing,
                nu=physics.nu,
                alpha=physics.alpha,
                t_interval=physics.dt,
                domain_length=physics.domain_length,
                upsample=physics.residual_upsample,
            )


class VorticityTrainingPathTest(unittest.TestCase):
    """The one-step pair the training loop builds runs through the vorticity residual."""

    def test_dispatch_reports_the_residual_and_backpropagates(self):
        physics = _physics(formulation="vorticity", forcing=PINO_COS4Y)
        forcing = build_forcing(physics, physics.nx * physics.residual_upsample)
        u = torch.randn(2, 1, physics.nx, physics.ny, 2, requires_grad=True)

        losses = PINO_loss3d_physics(u, forcing, physics, t_interval=physics.dt)
        self.assertEqual(len(losses), 7)
        residual = losses[4]
        self.assertGreater(residual.item(), 0.0)
        self.assertEqual(losses[5].item(), 0.0)
        self.assertEqual(losses[6].item(), 0.0)

        residual.backward()
        grad = u.grad
        assert grad is not None
        self.assertTrue(torch.isfinite(grad).all())


class ForcingEquivalenceTest(unittest.TestCase):
    def test_pino_cos4y_velocity_forcing_curls_to_the_vorticity_forcing(self):
        """curl of f = (sin(4y), 0) is the paper's vorticity forcing -4*cos(4y)."""
        S = 64
        velocity_forcing = build_forcing(_physics(forcing=PINO_COS4Y), S)
        vorticity_forcing = build_forcing(_physics(forcing=PINO_COS4Y, formulation="vorticity"), S)

        k = torch.fft.fftfreq(S, d=1.0 / S)
        fx_hat = torch.fft.fft2(velocity_forcing[0, 0, :, :, 0])
        fy_hat = torch.fft.fft2(velocity_forcing[0, 1, :, :, 0])
        curl = torch.fft.ifft2(1j * k.reshape(S, 1) * fy_hat - 1j * k.reshape(1, S) * fx_hat).real

        self.assertTrue(torch.allclose(curl, vorticity_forcing[0, :, :, 0], atol=1e-4))
        self.assertAlmostEqual(vorticity_forcing.abs().max().item(), 4.0, places=5)


if __name__ == "__main__":
    unittest.main()


class ResidualScaleTest(unittest.TestCase):
    """The relative residuals divide by scales measured on the ground truth."""

    def _taylor_green_truth(self, physics, nt=2):
        """(u, 2u) for the divergence-free field u = (sin x cos y, -cos x sin y)."""
        S = physics.nx
        grid = torch.tensor([2 * math.pi * i / S for i in range(S)], dtype=torch.float64)
        x = grid.reshape(S, 1)
        y = grid.reshape(1, S)
        state = torch.zeros(1, 3, S, S, nt, dtype=torch.float64)
        for n in range(nt):
            state[0, 0, :, :, n] = (n + 1) * torch.sin(x) * torch.cos(y)
            state[0, 1, :, :, n] = -(n + 1) * torch.cos(x) * torch.sin(y)
        return state

    def test_scales_match_the_analytic_truth_statistics(self):
        physics = _physics()
        truth = self._taylor_green_truth(physics)
        scales = residual_scales_from_truth(truth, physics, t_interval=physics.dt)

        # dux_dx^2 + duy_dy^2 = 2(n+1)^2 cos^2 x cos^2 y, whose domain mean is
        # (n+1)^2/2; averaged over the two levels and rooted, sqrt(1.25).
        self.assertAlmostEqual(scales.cont, math.sqrt(1.25), places=12)
        # The one-step increment is exactly u itself: RMS = sqrt(1/4).
        self.assertAlmostEqual(scales.momx, 0.5, places=12)
        self.assertAlmostEqual(scales.momy, 0.5, places=12)

    def test_the_relative_residual_divides_by_the_supplied_scale(self):
        physics = _physics()
        torch.manual_seed(0)
        u = torch.randn(2, 3, 32, 32, 2, dtype=torch.float64)
        forcing = build_forcing(physics, 32, dtype=torch.float64)
        scales = residual_scales_from_truth(u, physics, t_interval=physics.dt)
        doubled = ResidualScales(cont=2 * scales.cont, momx=2 * scales.momx, momy=2 * scales.momy)

        _, loss_cont, _, _, cont_rel, momx_rel, _ = PINO_loss3d_vel(
            u,
            u[..., 0],
            forcing,
            nu=physics.nu,
            alpha=physics.alpha,
            t_interval=physics.dt,
            domain_length=physics.domain_length,
            scales=scales,
        )
        _, _, _, _, cont_rel_half, momx_rel_half, _ = PINO_loss3d_vel(
            u,
            u[..., 0],
            forcing,
            nu=physics.nu,
            alpha=physics.alpha,
            t_interval=physics.dt,
            domain_length=physics.domain_length,
            scales=doubled,
        )

        self.assertAlmostEqual(
            cont_rel.item() * scales.cont / math.sqrt(loss_cont.item()), 1.0, places=7
        )
        # The eps guard on the denominator keeps this from being exact.
        self.assertAlmostEqual(cont_rel_half.item(), 0.5 * cont_rel.item(), places=8)
        self.assertAlmostEqual(momx_rel_half.item(), 0.5 * momx_rel.item(), places=8)


class ContinuityLevelTest(unittest.TestCase):
    """Continuity is scored on the predicted level only, not on the truth input."""

    def test_a_divergent_input_level_does_not_enter_the_continuity_loss(self):
        physics = _physics()
        S = physics.nx
        forcing = build_forcing(physics, S, dtype=torch.float64)
        grid = torch.tensor([2 * math.pi * i / S for i in range(S)], dtype=torch.float64)
        x = grid.reshape(S, 1)
        y = grid.reshape(1, S)

        state = torch.zeros(1, 3, S, S, 2, dtype=torch.float64)
        # Level 0 (the ground-truth input) is strongly divergent...
        state[0, 0, :, :, 0] = torch.sin(x) * torch.ones_like(y)
        # ...level 1 (the prediction) is divergence-free.
        state[0, 0, :, :, 1] = torch.sin(y) * torch.ones_like(x)
        state[0, 1, :, :, 1] = torch.sin(x) * torch.ones_like(y)

        scales = residual_scales_from_truth(state, physics, t_interval=physics.dt)
        _, loss_cont, _, _, cont_rel, _, _ = PINO_loss3d_vel(
            state,
            state[..., 0],
            forcing,
            nu=physics.nu,
            alpha=physics.alpha,
            t_interval=physics.dt,
            domain_length=physics.domain_length,
            scales=scales,
        )
        self.assertLess(loss_cont.item(), 1e-24)
        self.assertLess(cont_rel.item(), 1e-10)


class VelocityFromVorticityTest(unittest.TestCase):
    """The spectral inversion the residual and the spectra share."""

    def test_the_curl_of_the_recovered_velocity_is_the_vorticity(self):
        S = 32
        torch.manual_seed(0)
        # The round trip is exact on the range of the discrete curl, which the E0
        # and E1 datasets are projected onto: no mean mode, no Nyquist row or
        # column, the modes d/dx and d/dy annihilate.
        w_h = torch.fft.fft2(torch.randn(2, S, S, 3, dtype=torch.float64), dim=[1, 2])
        w_h[:, 0, 0] = 0
        w_h[:, S // 2] = 0
        w_h[:, :, S // 2] = 0
        w = torch.fft.ifft2(w_h, dim=[1, 2]).real

        ux, uy = velocity_from_vorticity(w, domain_length=2 * math.pi)

        k = torch.fft.fftfreq(S, d=1.0 / S, dtype=torch.float64)
        duy_dx = torch.fft.ifft2(
            1j * k.reshape(1, S, 1, 1) * torch.fft.fft2(uy, dim=[1, 2]), dim=[1, 2]
        ).real
        dux_dy = torch.fft.ifft2(
            1j * k.reshape(1, 1, S, 1) * torch.fft.fft2(ux, dim=[1, 2]), dim=[1, 2]
        ).real
        self.assertLess((duy_dx - dux_dy - w).abs().max().item(), 1e-10)

    def test_the_recovered_velocity_is_divergence_free(self):
        S = 32
        torch.manual_seed(0)
        w = torch.randn(1, S, S, 1, dtype=torch.float64)  # any field, Nyquist included
        ux, uy = velocity_from_vorticity(w, domain_length=2 * math.pi)

        k = torch.fft.fftfreq(S, d=1.0 / S, dtype=torch.float64)
        div = torch.fft.ifft2(
            1j * k.reshape(1, S, 1, 1) * torch.fft.fft2(ux, dim=[1, 2])
            + 1j * k.reshape(1, 1, S, 1) * torch.fft.fft2(uy, dim=[1, 2]),
            dim=[1, 2],
        ).real
        self.assertLess(div.abs().max().item(), 1e-12)


class ResidualHasScaleTest(unittest.TestCase):
    """Whether the configured residual has a denominator to be reported against.

    The vorticity residual is scored relative to the forcing, following PINO, so an
    unforced vorticity run has nothing to divide by and no residual to report. The
    velocity residuals divide by truth statistics instead (#16) and are always defined.
    """

    @staticmethod
    def _make(formulation, forcing):
        return physics_from_config(
            {
                "nu": 1e-3,
                "alpha": 0.0,
                "dt": 0.1,
                "domain_length": 2 * math.pi,
                "nx": 16,
                "ny": 16,
                "formulation": formulation,
                "forcing": forcing,
            }
        )

    @staticmethod
    def _field(physics):
        return build_forcing_for_data(physics, (16, 16))

    def test_an_unforced_vorticity_run_has_no_scale(self):
        physics = self._make("vorticity", {"type": "none"})
        self.assertFalse(residual_has_scale(physics, self._field(physics)))

    def test_a_forced_vorticity_run_has_a_scale(self):
        physics = self._make("vorticity", {"type": "pino_cos4y", "amplitude": 1.0, "wavenumber": 4})
        self.assertTrue(residual_has_scale(physics, self._field(physics)))

    def test_a_named_forcing_of_zero_amplitude_has_no_scale(self):
        # The question is asked of the field, not of the forcing's name, so that it is
        # the same question PINO_loss3d asks and cannot drift from it.
        physics = self._make("vorticity", {"type": "pino_cos4y", "amplitude": 0.0, "wavenumber": 4})
        self.assertTrue(torch.all(self._field(physics) == 0))
        self.assertFalse(residual_has_scale(physics, self._field(physics)))

    def test_an_unforced_velocity_run_still_has_a_scale(self):
        # E3's own case: the coarse field carries no forcing, and the velocity
        # residuals are still defined because they are normalised by the truth.
        physics = self._make("velocity", {"type": "none"})
        self.assertTrue(residual_has_scale(physics, self._field(physics)))
