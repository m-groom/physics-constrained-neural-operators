import unittest

import torch

from models.layers import SpectralCurl
from utils.spectral_test_utils import build_wavenumbers, evaluate_error


def generate_vector_field_and_curl(grid_shape, device="cpu", dtype=torch.float64):
    d = len(grid_shape)
    u = torch.randn(d, *grid_shape, device=device, dtype=dtype)
    u_hat = torch.fft.fftn(u, dim=tuple(range(-d, 0)))
    k = build_wavenumbers(grid_shape, device=device, dtype=dtype)

    deriv_hat = []

    for ki in k:
        ki = ki.reshape(*([1] * (u_hat.ndim - d)), *ki.shape)
        deriv_hat.append(1j * u_hat * ki)

    if d == 2:
        curl_hat = torch.stack([deriv_hat[0][1] - deriv_hat[1][0]], dim=0)
    elif d == 3:
        curl_hat = torch.stack(
            [
                deriv_hat[1][2] - deriv_hat[2][1],
                deriv_hat[2][0] - deriv_hat[0][2],
                deriv_hat[0][1] - deriv_hat[1][0],
            ],
            dim=0,
        )
    else:
        comps = []

        for i in range(d):
            for j in range(i + 1, d):
                comps.append(deriv_hat[i][j] - deriv_hat[j][i])

        curl_hat = torch.stack(comps, dim=0)

    curl = torch.fft.ifftn(curl_hat, dim=tuple(range(-d, 0))).real

    return u, curl


class SpectralCurlExactTest(unittest.TestCase):
    def test_matches_spectral_reference_across_dimensions(self):
        torch.manual_seed(0)

        device = "cpu"
        dtype = torch.float64
        cases = [(64, 64), (32, 32, 32), (12, 12, 12, 12)]

        for grid_shape in cases:
            with self.subTest(grid_shape=grid_shape):
                domain_lengths = tuple(1.0 for _ in grid_shape)
                curl_layer = SpectralCurl(
                    grid_shape=None,
                    domain_lengths=domain_lengths,
                    device=device,
                    dtype=dtype,
                )

                u, true_curl = generate_vector_field_and_curl(
                    grid_shape,
                    device=device,
                    dtype=dtype,
                )

                u_field = u.unsqueeze(0)

                curl_spec = curl_layer(u_field).squeeze(0)

                for i in range(true_curl.shape[0]):
                    max_err, rmse = evaluate_error(curl_spec[i], true_curl[i])
                    self.assertLess(max_err, 1e-10)
                    self.assertLess(rmse, 1e-10)

    def test_reuses_layer_across_3d_grid_shapes(self):
        torch.manual_seed(0)

        device = "cpu"
        dtype = torch.float64
        domain_lengths = (1.0, 1.0, 1.0)
        cases = [(64, 64, 64), (32, 32, 32)]

        curl_layer = SpectralCurl(
            grid_shape=None,
            domain_lengths=domain_lengths,
            device=device,
            dtype=dtype,
        )

        for grid_shape in cases:
            with self.subTest(grid_shape=grid_shape):
                u, true_curl = generate_vector_field_and_curl(
                    grid_shape,
                    device=device,
                    dtype=dtype,
                )

                curl_spec = curl_layer(u.unsqueeze(0)).squeeze(0)

                for i in range(true_curl.shape[0]):
                    max_err, rmse = evaluate_error(curl_spec[i], true_curl[i])
                    self.assertLess(max_err, 1e-10)
                    self.assertLess(rmse, 1e-10)


if __name__ == "__main__":
    unittest.main()
