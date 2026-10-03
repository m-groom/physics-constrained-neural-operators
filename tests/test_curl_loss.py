"""The curl-matching loss term: analytic value, exactness on truth, and its k-weighting.

The term exists to give a velocity-formulation loss the wavenumber weighting a vorticity
loss has for free. Because `omega_hat = i (kx v_hat - ky u_hat)`, a velocity error at
wavenumber k contributes to the vorticity error in proportion to k, so the band above the
FNO's spectral cutoff -- invisible to the velocity relative L2 (#42) -- is weighted by the
curl term instead. The last test is that statement made falsifiable.
"""

import math
import unittest

import torch

from utils.criterion import curl_rel_l2_2d

S = 64
L = 2 * math.pi


def grid(side=S, lengths=(L, L), dtype=torch.float64):
    """Periodic grid, `(x, y)` each `(side, side)`, x along axis 0 -- SpectralCurl's
    convention, and `max_abs_divergence_2d_velocity`'s."""
    x = torch.arange(side, dtype=dtype) * (lengths[0] / side)
    y = torch.arange(side, dtype=dtype) * (lengths[1] / side)
    return torch.meshgrid(x, y, indexing="ij")


def taylor_green(amplitude, dtype=torch.float64):
    """Velocity of the stream function `amplitude * cos(x) * cos(y)`.

    With `u = d(psi)/dy` and `v = -d(psi)/dx` the field is divergence-free by
    construction, and its vorticity is analytic (see the tests below).
    """
    x, y = grid(dtype=dtype)
    u = -amplitude * torch.cos(x) * torch.sin(y)
    v = amplitude * torch.sin(x) * torch.cos(y)
    return torch.stack([u, v], dim=-1).unsqueeze(0)


def shear_at_wavenumber(k, amplitude=1.0, dtype=torch.float64):
    """Divergence-free velocity `(0, amplitude * cos(k x))`, whose vorticity is
    `-k * amplitude * sin(k x)`.

    The velocity L2 norm is independent of `k` and the vorticity norm is proportional
    to it, which is exactly the weighting the curl term is being added for.
    """
    x, _ = grid(dtype=dtype)
    u = torch.zeros_like(x)
    v = amplitude * torch.cos(k * x)
    return torch.stack([u, v], dim=-1).unsqueeze(0)


class AnalyticCurlTest(unittest.TestCase):
    """The term reproduces a relative L2 that can be worked out on paper."""

    def test_a_halved_streamfunction_gives_a_relative_error_of_one_half(self):
        # psi = A cos(x) cos(y) has vorticity -lap(psi) = 2A cos(x) cos(y), so a
        # prediction built from A = 1/2 against a truth built from A = 1 has
        # ||w_pred - w_truth|| / ||w_truth|| = ||-A cos cos|| / ||2A cos cos|| = 1/2.
        pred = taylor_green(amplitude=0.5)
        truth = taylor_green(amplitude=1.0)
        self.assertAlmostEqual(float(curl_rel_l2_2d(pred, truth, (L, L))), 0.5, places=10)

    def test_the_curl_of_a_pure_shear_scales_with_its_wavenumber(self):
        # (0, cos(k x)) has vorticity -k sin(k x). Against a zero-vorticity truth the
        # relative L2 is undefined, so compare two shears at the same k instead: a
        # prediction of half the amplitude again gives exactly 1/2.
        pred = shear_at_wavenumber(4, amplitude=0.5)
        truth = shear_at_wavenumber(4, amplitude=1.0)
        self.assertAlmostEqual(float(curl_rel_l2_2d(pred, truth, (L, L))), 0.5, places=10)


class ExactDataTest(unittest.TestCase):
    """The term vanishes when the prediction is the data."""

    def test_the_term_is_zero_on_an_exact_prediction(self):
        truth = taylor_green(amplitude=1.0)
        self.assertLess(float(curl_rel_l2_2d(truth.clone(), truth, (L, L))), 1e-12)

    def test_the_term_is_zero_on_an_exact_prediction_of_a_random_field(self):
        torch.manual_seed(0)
        truth = torch.randn(3, S, S, 2, dtype=torch.float64)
        self.assertLess(float(curl_rel_l2_2d(truth.clone(), truth, (L, L))), 1e-12)

    def test_a_float32_exact_prediction_is_zero_to_single_precision(self):
        torch.manual_seed(0)
        truth = torch.randn(2, S, S, 2)
        self.assertLess(float(curl_rel_l2_2d(truth.clone(), truth, (L, L))), 1e-6)


class WavenumberWeightingTest(unittest.TestCase):
    """The hypothesis the term is being added to test, stated as an assertion.

    Two predictions carry the *same* velocity error norm, one at k = 2 and one at
    k = 20. The velocity relative L2 cannot tell them apart; the curl term ranks the
    high-wavenumber one ten times worse, in proportion to the ratio of wavenumbers.
    """

    def setUp(self):
        self.truth = taylor_green(amplitude=1.0)

    def _error_norms(self, k, epsilon=1e-2):
        perturbation = shear_at_wavenumber(k, amplitude=epsilon)
        pred = self.truth + perturbation
        velocity_error = float(torch.linalg.vector_norm(perturbation))
        return velocity_error, float(curl_rel_l2_2d(pred, self.truth, (L, L)))

    def test_the_two_perturbations_carry_the_same_velocity_error(self):
        low, high = self._error_norms(2)[0], self._error_norms(20)[0]
        self.assertAlmostEqual(low, high, places=10)

    def test_the_curl_term_weights_the_high_wavenumber_error_by_its_wavenumber(self):
        low = self._error_norms(2)[1]
        high = self._error_norms(20)[1]
        self.assertAlmostEqual(high / low, 10.0, delta=0.2)


class ContractTest(unittest.TestCase):
    """The shape contract, so a wrong-shaped state fails loudly rather than silently."""

    def test_a_state_that_is_not_a_two_component_velocity_is_refused(self):
        field = torch.randn(2, S, S, 3)
        with self.assertRaises(ValueError):
            curl_rel_l2_2d(field, field, (L, L))

    def test_a_rectangular_domain_uses_both_lengths(self):
        # Two fields that are NOT scalar multiples of one another, so the answer cannot
        # come out right by the ratio cancelling. The truth varies along x only and the
        # prediction along y only, at one wavelength each:
        #
        #   truth u = (0, sin(2 pi x / Lx))   -> w_t = (2 pi / Lx) cos(2 pi x / Lx)
        #   pred  u = (-sin(2 pi y / Ly), 0)  -> w_p = (2 pi / Ly) cos(2 pi y / Ly)
        #
        # The two vorticities are orthogonal (each integrates to zero over its own full
        # period), so ||w_p - w_t||^2 = ||w_p||^2 + ||w_t||^2, and their norms are in the
        # ratio Lx / Ly. The relative L2 is therefore sqrt(1 + (Lx / Ly)^2) -- a number
        # that moves when either length moves, which is what pins the wiring.
        for lengths in ((L, L), (2 * L, L), (L, 2 * L)):
            lx, ly = lengths
            x, y = grid(lengths=lengths)
            zero = torch.zeros_like(x)
            truth = torch.stack([zero, torch.sin(2 * math.pi * x / lx)], dim=-1).unsqueeze(0)
            pred = torch.stack([-torch.sin(2 * math.pi * y / ly), zero], dim=-1).unsqueeze(0)
            with self.subTest(lengths=lengths):
                self.assertAlmostEqual(
                    float(curl_rel_l2_2d(pred, truth, lengths)),
                    math.sqrt(1 + (lx / ly) ** 2),
                    places=8,
                )

    def test_the_domain_lengths_are_not_ignored(self):
        # The guard on the test above: the same fields scored on a square domain give a
        # different number, so passing the lengths through is doing work.
        x, y = grid(lengths=(2 * L, L))
        zero = torch.zeros_like(x)
        truth = torch.stack([zero, torch.sin(2 * math.pi * x / (2 * L))], dim=-1).unsqueeze(0)
        pred = torch.stack([-torch.sin(2 * math.pi * y / L), zero], dim=-1).unsqueeze(0)
        self.assertNotAlmostEqual(
            float(curl_rel_l2_2d(pred, truth, (2 * L, L))),
            float(curl_rel_l2_2d(pred, truth, (L, L))),
            places=3,
        )


if __name__ == "__main__":
    unittest.main()
