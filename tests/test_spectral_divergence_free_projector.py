import math
import unittest

import torch

from models.layers import SpectralDivergenceFreeProjector2D
from utils.criterion import max_abs_divergence_2d_velocity
from utils.spectral_test_utils import evaluate_error


class SpectralDivergenceFreeProjector2DTest(unittest.TestCase):
    def test_is_idempotent_on_already_projected_field(self):
        torch.manual_seed(0)

        for grid_shape in [(32, 32), (31, 33)]:
            with self.subTest(grid_shape=grid_shape):
                projector = SpectralDivergenceFreeProjector2D(
                    domain_lengths=(1.0, 1.0),
                    dtype=torch.float64,
                )
                velocity = torch.randn(1, 2, *grid_shape, dtype=torch.float64)
                projected_once = projector(velocity)
                projected_twice = projector(projected_once)

                max_err, rmse = evaluate_error(projected_twice, projected_once)
                self.assertLess(max_err, 1e-12)
                self.assertLess(rmse, 1e-12)

    def test_projects_random_field_to_near_zero_divergence(self):
        torch.manual_seed(0)

        for grid_shape in [(32, 32), (31, 33)]:
            with self.subTest(grid_shape=grid_shape):
                projector = SpectralDivergenceFreeProjector2D(
                    domain_lengths=(1.0, 1.0),
                    dtype=torch.float64,
                )
                velocity = torch.randn(2, 2, *grid_shape, dtype=torch.float64)
                projected = projector(velocity).permute(0, 2, 3, 1)
                div_max = max_abs_divergence_2d_velocity(
                    projected,
                    domain_lengths=(1.0, 1.0),
                )
                self.assertLess(div_max.max().item(), 1e-5)

    def test_preserves_constant_velocity_field(self):
        for grid_shape in [(16, 16), (15, 17)]:
            with self.subTest(grid_shape=grid_shape):
                projector = SpectralDivergenceFreeProjector2D(
                    domain_lengths=(1.0, 1.0),
                    dtype=torch.float64,
                )
                velocity = torch.zeros(1, 2, *grid_shape, dtype=torch.float64)
                velocity[:, 0] = 1.25
                velocity[:, 1] = -0.75

                projected = projector(velocity)
                max_err, rmse = evaluate_error(projected, velocity)
                self.assertLess(max_err, 1e-12)
                self.assertLess(rmse, 1e-12)

    def test_full_fft_wavenumbers_match_pde_residual_convention(self):
        """Verify that the projector's wavenumbers are integer-valued full-FFT
        wavenumbers, matching the convention used by the PDE residual code."""
        projector = SpectralDivergenceFreeProjector2D(
            domain_lengths=(2 * math.pi, 2 * math.pi),
            dtype=torch.float64,
        )
        grid_shape = (8, 6)

        kx = projector._wavenumber(0, grid_shape=grid_shape, device="cpu", dtype=torch.float64)
        ky = projector._wavenumber(1, grid_shape=grid_shape, device="cpu", dtype=torch.float64)

        nx, ny = grid_shape
        expected_kx = torch.cat(
            [
                torch.arange(0, nx // 2, dtype=torch.float64),
                torch.arange(-nx // 2, 0, dtype=torch.float64),
            ]
        )
        expected_ky = torch.cat(
            [
                torch.arange(0, ny // 2, dtype=torch.float64),
                torch.arange(-ny // 2, 0, dtype=torch.float64),
            ]
        )

        self.assertEqual(kx.shape, (nx, 1))
        self.assertEqual(ky.shape, (1, ny))
        self.assertTrue(torch.allclose(kx.squeeze(), expected_kx))
        self.assertTrue(torch.allclose(ky.squeeze(), expected_ky))

    def test_projected_field_is_divergence_free_under_pde_residual_operator(self):
        """Verify that the projected field has zero divergence when measured
        using the same full-FFT spectral derivative as the PDE residual code.

        The projector zeroes Nyquist modes for even-sized grids because their
        projection direction is ill-defined under fftfreq.  The PDE residual
        operator (ifft2(i*k*fft2(u)).real) assigns the Nyquist wavenumber a
        sign, so a small residual can appear at those modes.  We therefore
        allow a tolerance of 1e-6 rather than machine epsilon.
        """
        torch.manual_seed(42)

        for grid_shape in [(32, 32), (31, 33)]:
            with self.subTest(grid_shape=grid_shape):
                nx, ny = grid_shape
                Lx, Ly = 2 * math.pi, 2 * math.pi
                projector = SpectralDivergenceFreeProjector2D(
                    domain_lengths=(Lx, Ly),
                    dtype=torch.float64,
                )

                velocity = torch.randn(2, 2, nx, ny, dtype=torch.float64)
                projected = projector(velocity)

                ux = projected[:, 0]
                uy = projected[:, 1]

                # Use fftfreq to build wavenumbers, matching the PDE residual
                # convention exactly (works for both even and odd N).
                dx, dy = Lx / nx, Ly / ny
                k_x = (2 * math.pi * torch.fft.fftfreq(nx, d=dx, dtype=torch.float64)).reshape(
                    1, nx, 1
                )
                k_y = (2 * math.pi * torch.fft.fftfreq(ny, d=dy, dtype=torch.float64)).reshape(
                    1, 1, ny
                )

                ux_h = torch.fft.fft2(ux, dim=(1, 2))
                uy_h = torch.fft.fft2(uy, dim=(1, 2))
                dux_dx = torch.fft.ifft2(1j * k_x * ux_h, dim=(1, 2)).real
                duy_dy = torch.fft.ifft2(1j * k_y * uy_h, dim=(1, 2)).real

                div = dux_dx + duy_dy
                self.assertLess(div.abs().max().item(), 1e-6)


if __name__ == "__main__":
    unittest.main()
