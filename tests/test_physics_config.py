"""The physics block of the run configuration is the single source of truth."""

import math
import unittest
from pathlib import Path

import torch
import yaml

from models.fno import FNO2d
from utils.criterion import (
    build_forcing,
    build_forcing_for_data,
    check_model_channels,
    physics_from_config,
)

KOLMOGOROV_DIAG = {
    "type": "kolmogorov_diag",
    "amplitude": 2 * math.sqrt(2),
    "wavenumber": 4,
    "phase": 5 * math.pi / 4,
}
DATA_CONFIG = {
    "nu": 0.002,
    "alpha": 1e-8,
    "dt": 0.015625,
    "domain_length": 2 * math.pi,
    "nx": 128,
    "ny": 128,
    "formulation": "velocity",
    "forcing": KOLMOGOROV_DIAG,
}


def _constraint(**energy_balance):
    return {
        "enabled": True,
        "type": "divergence_free",
        "space": "physical",
        "velocity_channels": [0, 1],
        "energy_balance": {"enabled": True, **energy_balance},
    }


def _fno(physics, output_constraint):
    return FNO2d(
        in_dim=7,
        out_dim=3,
        modes1=[2, 2],
        modes2=[2, 2],
        fc_dim=8,
        layers=[8, 8, 8],
        act="gelu",
        output_constraint=output_constraint,
        physics=physics,
    )


class PhysicsFromConfigTest(unittest.TestCase):
    def test_every_required_key_must_be_present(self):
        for key in DATA_CONFIG:
            data_config = {k: v for k, v in DATA_CONFIG.items() if k != key}
            with self.subTest(missing=key), self.assertRaises(KeyError) as raised:
                physics_from_config(data_config)
            self.assertIn(key, str(raised.exception))

    def test_unknown_formulation_or_forcing_is_rejected(self):
        with self.assertRaises(ValueError):
            physics_from_config({**DATA_CONFIG, "formulation": "streamfunction"})
        with self.assertRaises(ValueError):
            physics_from_config({**DATA_CONFIG, "forcing": {"type": "kolmogorov"}})
        with self.assertRaises(ValueError):
            physics_from_config({**DATA_CONFIG, "forcing": "kolmogorov_diag"})

    def test_forcing_parameters_must_be_present(self):
        for key in ("amplitude", "wavenumber", "phase"):
            forcing = {k: v for k, v in KOLMOGOROV_DIAG.items() if k != key}
            with self.subTest(missing=key), self.assertRaises(KeyError) as raised:
                physics_from_config({**DATA_CONFIG, "forcing": forcing})
            self.assertIn(key, str(raised.exception))

    def test_residual_upsample_defaults_to_four(self):
        self.assertEqual(physics_from_config(DATA_CONFIG).residual_upsample, 4)
        self.assertEqual(
            physics_from_config({**DATA_CONFIG, "residual_upsample": 2}).residual_upsample, 2
        )


# Every configuration of a campaign resolves to that campaign's one physics block:
# E1 (``e1_noise``) is the PINO Kolmogorov flow, E2 (``e3_final``) the truncated
# stochastically forced turbulence, whose forcing lies entirely beyond the cutoff.
CAMPAIGNS = {
    "e1_noise": (8, ("velocity", 0.002, 0.0, 0.015625, (64, 64), "pino_cos4y")),
    "e3_final": (9, ("velocity", 9.3e-06, 0.003, 0.4, (64, 64), "none")),
}


class ShippedConfigsTest(unittest.TestCase):
    def test_every_config_carries_a_valid_physics_block(self):
        config_dir = Path(__file__).resolve().parent.parent / "experiments" / "configs"
        for campaign, (count, expected) in CAMPAIGNS.items():
            configs = sorted((config_dir / campaign).glob("*.yaml"))
            self.assertEqual(len(configs), count, campaign)
            for path in configs:
                with self.subTest(config=f"{campaign}/{path.name}"):
                    data_config = yaml.safe_load(path.read_text())["data"]
                    physics = physics_from_config(data_config)
                    resolved = (
                        physics.formulation,
                        physics.nu,
                        physics.alpha,
                        physics.dt,
                        (physics.nx, physics.ny),
                        physics.forcing["type"],
                    )
                    self.assertEqual(resolved, expected)


class BuildForcingTest(unittest.TestCase):
    def test_shapes_follow_the_formulation(self):
        velocity = build_forcing(physics_from_config(DATA_CONFIG), 32)
        self.assertEqual(tuple(velocity.shape), (1, 2, 32, 32, 1))

        vorticity_config = {
            **DATA_CONFIG,
            "formulation": "vorticity",
            "forcing": {"type": "pino_cos4y", "amplitude": 1.0, "wavenumber": 4},
        }
        vorticity = build_forcing(physics_from_config(vorticity_config), 32)
        self.assertEqual(tuple(vorticity.shape), (1, 32, 32, 1))

    def test_no_forcing_is_zero(self):
        physics = physics_from_config({**DATA_CONFIG, "forcing": {"type": "none"}})
        self.assertEqual(build_forcing(physics, 32).abs().max().item(), 0.0)

    def test_every_forcing_has_a_form_in_both_formulations(self):
        for forcing in (KOLMOGOROV_DIAG, {"type": "pino_cos4y", "amplitude": 1.0, "wavenumber": 4}):
            for formulation, shape in (
                ("velocity", (1, 2, 32, 32, 1)),
                ("vorticity", (1, 32, 32, 1)),
            ):
                with self.subTest(forcing=forcing["type"], formulation=formulation):
                    physics = physics_from_config(
                        {**DATA_CONFIG, "formulation": formulation, "forcing": forcing}
                    )
                    self.assertEqual(tuple(build_forcing(physics, 32).shape), shape)


class ForcingForDataTest(unittest.TestCase):
    def test_velocity_forcing_matches_the_data_grid(self):
        physics = physics_from_config({**DATA_CONFIG, "nx": 64, "ny": 64})
        forcing = build_forcing_for_data(physics, (64, 64))
        self.assertEqual(tuple(forcing.shape), (1, 2, 64, 64, 1))

    def test_vorticity_forcing_lands_on_the_refined_grid(self):
        physics = physics_from_config(
            {
                **DATA_CONFIG,
                "nx": 64,
                "ny": 64,
                "formulation": "vorticity",
                "forcing": {"type": "pino_cos4y", "amplitude": 1.0, "wavenumber": 4},
            }
        )
        forcing = build_forcing_for_data(physics, (64, 64))
        self.assertEqual(tuple(forcing.shape), (1, 256, 256, 1))

    def test_data_resolution_must_match_the_configuration(self):
        physics = physics_from_config(DATA_CONFIG)
        with self.assertRaises(ValueError) as raised:
            build_forcing_for_data(physics, (64, 64))
        self.assertIn("128", str(raised.exception))


class StateChannelsTest(unittest.TestCase):
    def test_the_formulation_fixes_the_model_width(self):
        velocity = physics_from_config(DATA_CONFIG)
        vorticity = physics_from_config(
            {
                **DATA_CONFIG,
                "formulation": "vorticity",
                "forcing": {"type": "pino_cos4y", "amplitude": 1.0, "wavenumber": 4},
            }
        )
        check_model_channels(velocity, 3)
        check_model_channels(vorticity, 1)
        with self.assertRaises(ValueError):
            check_model_channels(vorticity, 3)
        with self.assertRaises(ValueError):
            check_model_channels(velocity, 1)


class EnergyBalanceProjectorConfigTest(unittest.TestCase):
    def test_projector_takes_viscosity_friction_and_dt_from_the_data_block(self):
        physics = physics_from_config(DATA_CONFIG)
        model = _fno(physics, _constraint())

        projector = model.output_projector
        self.assertEqual(projector.viscosity, physics.nu)
        # The friction is the data's alpha = 1e-8, not the 0.025 of the old config block.
        self.assertEqual(projector.friction, 1e-8)
        self.assertEqual(projector.dt, physics.dt)
        self.assertEqual(projector.domain_lengths, (physics.domain_length,) * 2)

    def test_physics_overrides_in_the_constraint_block_are_rejected(self):
        physics = physics_from_config(DATA_CONFIG)
        for stale in ("viscosity", "friction", "dt", "forcing"):
            with self.subTest(key=stale), self.assertRaises(ValueError) as raised:
                _fno(physics, _constraint(**{stale: 0.025}))
            self.assertIn(stale, str(raised.exception))

    def test_the_friction_from_the_data_block_is_the_one_that_acts(self):
        """A pair balanced at alpha = 1e-8 is left alone; at 0.025 it is moved hard."""
        N = 16
        physics = physics_from_config({**DATA_CONFIG, "nx": N, "ny": N})
        wrong = physics_from_config({**DATA_CONFIG, "nx": N, "ny": N, "alpha": 0.025})

        projectors = []
        for candidate in (physics, wrong):
            model = _fno(candidate, _constraint(max_iterations=20))
            model.set_energy_forcing(build_forcing(candidate, N).squeeze(-1))
            projectors.append(model.output_projector)
        from_data, from_the_old_block = projectors

        torch.manual_seed(0)
        u_old = from_data(torch.randn(2, 2, N, N))
        balanced = from_data(torch.randn(2, 2, N, N), u_old=u_old)

        stays = (from_data(balanced, u_old=u_old) - balanced).norm().item()
        moves = (from_the_old_block(balanced, u_old=u_old) - balanced).norm().item()
        self.assertLess(stays, 1e-5 * balanced.norm().item())
        self.assertGreater(moves, 1e3 * stays)

    def test_subgrid_source_reaches_the_projector_and_defaults_to_zero(self):
        """The closure constant is a constraint-block scalar, zero unless a config sets it."""
        physics = physics_from_config(DATA_CONFIG)
        self.assertEqual(_fno(physics, _constraint()).output_projector.subgrid_source, 0.0)
        closed = _fno(physics, _constraint(subgrid_source=3.2e-04))
        self.assertEqual(closed.output_projector.subgrid_source, 3.2e-04)

    def test_energy_balance_requires_physics(self):
        with self.assertRaises(ValueError):
            _fno(None, _constraint())

    def test_forcing_must_be_a_field(self):
        model = _fno(physics_from_config(DATA_CONFIG), _constraint())
        with self.assertRaises(TypeError):
            model.set_energy_forcing(128)
        model.set_energy_forcing(torch.zeros(1, 2, 16, 16))
        self.assertIsNotNone(model.output_projector.forcing_hat)


if __name__ == "__main__":
    unittest.main()
