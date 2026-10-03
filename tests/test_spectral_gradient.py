import unittest

import torch

from models.layers import SpectralGradient
from utils.spectral_test_utils import build_wavenumbers, evaluate_error


def generate_field_and_gradient(grid_shape, device="cpu", dtype=torch.float64):
    u = torch.randn(*grid_shape, device=device, dtype=dtype)
    u_hat = torch.fft.fftn(u)
    k = build_wavenumbers(grid_shape, device=device, dtype=dtype)

    grads = []

    for ki in k:
        grad_hat = 1j * ki * u_hat
        grad = torch.fft.ifftn(grad_hat).real
        grads.append(grad)

    return u, torch.stack(grads, dim=0)


class SpectralGradientExactTest(unittest.TestCase):
    def test_matches_spectral_reference_in_3d(self):
        torch.manual_seed(0)

        device = "cpu"
        dtype = torch.float64
        domain_lengths = (1.0, 1.0, 1.0)
        cases = [(64, 64, 64), (32, 32, 32)]

        grad_layer = SpectralGradient(
            grid_shape=None,
            domain_lengths=domain_lengths,
            device=device,
            dtype=dtype,
        )

        for grid_shape in cases:
            with self.subTest(grid_shape=grid_shape):
                u, true_grad = generate_field_and_gradient(
                    grid_shape,
                    device=device,
                    dtype=dtype,
                )

                u_field = u.unsqueeze(0).unsqueeze(0)
                grad_spec = grad_layer(u_field).squeeze(0)

                for i in range(len(grid_shape)):
                    max_err, rmse = evaluate_error(grad_spec[i], true_grad[i])
                    self.assertLess(max_err, 1e-10)
                    self.assertLess(rmse, 1e-10)


if __name__ == "__main__":
    unittest.main()
