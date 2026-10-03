import unittest

import torch

from models.fno import FNO2d
from utils.criterion import max_abs_divergence_2d_velocity


class FNOOutputConstraintTest(unittest.TestCase):
    def test_apply_output_constraint_only_updates_velocity_channels(self):
        model = FNO2d(
            in_dim=7,
            out_dim=3,
            modes1=[2, 2],
            modes2=[2, 2],
            fc_dim=8,
            layers=[8, 8, 8],
            act="gelu",
            output_constraint={
                "enabled": True,
                "type": "divergence_free",
                "space": "physical",
                "velocity_channels": [0, 1],
                "domain_lengths": [1.0, 1.0],
            },
        )
        mean = torch.zeros(3, 1, 1)
        std = torch.ones(3, 1, 1)
        model.set_output_normalizer(mean, std)

        pred = torch.randn(2, 16, 16, 3)
        constrained = model.apply_output_constraint(pred)

        self.assertEqual(constrained.shape, pred.shape)
        self.assertTrue(torch.allclose(constrained[..., 2], pred[..., 2]))

        div_max = max_abs_divergence_2d_velocity(
            constrained[..., :2],
            domain_lengths=(1.0, 1.0),
        )
        self.assertLess(div_max.max().item(), 1e-4)

    def test_apply_output_constraint_accepts_final_residual_state(self):
        model = FNO2d(
            in_dim=7,
            out_dim=3,
            modes1=[2, 2],
            modes2=[2, 2],
            fc_dim=8,
            layers=[8, 8, 8],
            act="gelu",
            output_constraint={
                "enabled": True,
                "type": "divergence_free",
                "space": "physical",
                "velocity_channels": [0, 1],
                "domain_lengths": [1.0, 1.0],
            },
        )
        model.set_output_normalizer(torch.zeros(3, 1, 1), torch.ones(3, 1, 1))

        x = torch.randn(2, 16, 16, 3)
        delta = torch.randn(2, 16, 16, 3)
        final_state = x + delta
        constrained = model.apply_output_constraint(final_state)

        self.assertEqual(constrained.shape, final_state.shape)
        div_max = max_abs_divergence_2d_velocity(
            constrained[..., :2],
            domain_lengths=(1.0, 1.0),
        )
        self.assertLess(div_max.max().item(), 1e-4)


if __name__ == "__main__":
    unittest.main()
