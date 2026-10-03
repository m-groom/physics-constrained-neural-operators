import unittest

import torch

from models.layers import FunctionalMaskConstraint


def toy_field_2d(x, y):
    return torch.sin(2 * torch.pi * x) * torch.cos(3 * torch.pi * y) + 0.25 * torch.cos(
        5 * torch.pi * x * y
    )


def ellipse_mask(positions, a=0.35, b=0.2, center=(0.5, 0.5)):
    x = positions[0]
    y = positions[1]
    cx, cy = center
    return ((x - cx) / a) ** 2 + ((y - cy) / b) ** 2 - 1.0


def tilted_strip_mask(positions, slope=0.8, intercept=0.1, width=0.05):
    x = positions[0]
    y = positions[1]
    return y - slope * x - intercept - width


def radial_mask(positions, radius=0.25, center=(0.5, 0.5)):
    x = positions[0]
    y = positions[1]
    cx, cy = center
    return (x - cx) ** 2 + (y - cy) ** 2 - radius**2


class FunctionalMaskConstraintTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.device = "cpu"
        self.dtype = torch.float64
        self.domain_lengths = (1.0, 1.0)

    def build_field(self, grid_shape):
        x = torch.arange(grid_shape[0], device=self.device, dtype=self.dtype) * (
            self.domain_lengths[0] / grid_shape[0]
        )
        y = torch.arange(grid_shape[1], device=self.device, dtype=self.dtype) * (
            self.domain_lengths[1] / grid_shape[1]
        )
        x, y = torch.meshgrid(x, y, indexing="ij")
        field = toy_field_2d(x, y).unsqueeze(0).unsqueeze(0)
        return x, y, field

    def check_mask(self, mask_function, grid_shape, boundary_points):
        layer = FunctionalMaskConstraint(
            grid_shape=None,
            domain_lengths=self.domain_lengths,
            mask_function=mask_function,
            device=self.device,
            dtype=self.dtype,
        )

        _, _, field = self.build_field(grid_shape)
        masked = layer(field).squeeze(0).squeeze(0)

        residuals = []

        for ix, iy in boundary_points:
            residuals.append(masked[ix, iy].abs().item())

        residual = max(residuals)
        return residual

    def test_ellipse_mask_vanishes_on_boundary(self):
        grid_shape = (160, 160)
        boundary_points = [
            (24, 80),
            (136, 80),
            (80, 48),
            (80, 112),
        ]
        residual = self.check_mask(ellipse_mask, grid_shape, boundary_points)
        self.assertLess(residual, 1e-12, f"ellipse boundary residual={residual}")

    def test_tilted_strip_mask_vanishes_on_boundary(self):
        grid_shape = (160, 160)
        boundary_points = [
            (20, 40),
            (80, 88),
            (120, 120),
        ]
        residual = self.check_mask(tilted_strip_mask, grid_shape, boundary_points)
        self.assertLess(residual, 1e-12, f"strip boundary residual={residual}")

    def test_radial_mask_vanishes_on_boundary(self):
        grid_shape = (160, 160)
        boundary_points = [
            (120, 80),
            (40, 80),
            (80, 120),
            (80, 40),
        ]
        residual = self.check_mask(radial_mask, grid_shape, boundary_points)
        self.assertLess(residual, 1e-12, f"radial boundary residual={residual}")

    def test_reuses_mask_layer_across_grid_shapes(self):
        layer = FunctionalMaskConstraint(
            grid_shape=None,
            domain_lengths=self.domain_lengths,
            mask_function=ellipse_mask,
            device=self.device,
            dtype=self.dtype,
        )

        cases = [
            ((160, 160), [(24, 80), (136, 80), (80, 48), (80, 112)]),
            ((200, 200), [(30, 100), (170, 100), (100, 60), (100, 140)]),
        ]

        for grid_shape, boundary_points in cases:
            with self.subTest(grid_shape=grid_shape):
                _, _, field = self.build_field(grid_shape)
                masked = layer(field).squeeze(0).squeeze(0)

                residuals = [masked[ix, iy].abs().item() for ix, iy in boundary_points]
                residual = max(residuals)
                self.assertLess(residual, 1e-12, f"ellipse boundary residual={residual}")


if __name__ == "__main__":
    unittest.main()
