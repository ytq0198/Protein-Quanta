import unittest

import torch

from protein_quanta.anchored_velocity_residual import AnchoredGatedVelocityDynamics
from protein_quanta.velocity_equivariant_dynamics import VelocityEquivariantAcceleration


class AnchoredVelocityResidualTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        anchor = VelocityEquivariantAcceleration(hidden_dim=8)
        residual = VelocityEquivariantAcceleration(
            hidden_dim=8,
            bounded_damping_max=0.05,
            normalize_velocity_invariants=True,
        )
        self.model = AnchoredGatedVelocityDynamics(
            anchor, residual, hidden_dim=8, maximum_gate=0.25,
        )
        self.position = torch.randn(4, 3)
        self.velocity = torch.randn(4, 3)
        self.condition = (
            torch.tensor([6, 7, 8, 6]),
            torch.tensor([0, 0, 1, 1]),
            torch.tensor([12.0, 14.0, 16.0, 12.0]),
            torch.empty(0),
            torch.randn(6, 3),
            torch.empty(0),
            torch.tensor([1, 2, 3, 4, 5, 6]),
            torch.tensor([0, 0, 0, 1, 1, 1]),
        )

    @staticmethod
    def _orthogonal(reflection=False):
        matrix = torch.randn(3, 3)
        matrix, _ = torch.linalg.qr(matrix)
        desired = -1 if reflection else 1
        if torch.sign(torch.linalg.det(matrix)).item() != desired:
            matrix[:, -1] = -matrix[:, -1]
        return matrix

    def test_initial_output_exactly_matches_frozen_anchor(self):
        state = (self.velocity, self.position)
        expected = self.model.anchor(0, state, self.condition)[0]
        actual = self.model(0, state, self.condition)[0]
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(all(
            not parameter.requires_grad for parameter in self.model.anchor.parameters()
        ))

    def test_initial_scale_gradient_is_finite_nonzero(self):
        acceleration = self.model(
            0, (self.velocity, self.position), self.condition
        )[0]
        gradient = torch.autograd.grad(
            acceleration.square().mean(), self.model.residual_scale_logit
        )[0]
        self.assertTrue(torch.isfinite(gradient))
        self.assertGreater(abs(gradient.item()), 0)

    def test_gate_is_small_and_bounded(self):
        gate = self.model.gate_coefficient(self.condition[0], self.velocity)
        self.assertTrue((gate >= 0).all())
        self.assertTrue((gate <= 0.25).all())
        torch.testing.assert_close(gate, torch.full_like(gate, 0.0125))

    def test_joint_orthogonal_translation_equivariance(self):
        for reflection in (False, True):
            matrix = self._orthogonal(reflection)
            translation = torch.tensor([1.2, -0.5, 2.0])
            reference = self.model(
                0, (self.velocity, self.position), self.condition
            )[0]
            transformed_condition = list(self.condition)
            transformed_condition[4] = self.condition[4] @ matrix.T + translation
            transformed = self.model(
                0,
                (
                    self.velocity @ matrix.T,
                    self.position @ matrix.T + translation,
                ),
                tuple(transformed_condition),
            )[0]
            torch.testing.assert_close(
                transformed, reference @ matrix.T, atol=2e-6, rtol=2e-6
            )

    def test_invalid_gate_configuration_is_rejected(self):
        anchor = VelocityEquivariantAcceleration(hidden_dim=8)
        residual = VelocityEquivariantAcceleration(hidden_dim=8)
        with self.assertRaises(ValueError):
            AnchoredGatedVelocityDynamics(anchor, residual, maximum_gate=0)
        with self.assertRaises(ValueError):
            AnchoredGatedVelocityDynamics(
                anchor, residual, initial_gate_fraction=1
            )


if __name__ == "__main__":
    unittest.main()
