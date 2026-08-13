import unittest

import torch

from protein_quanta.velocity_equivariant_dynamics import (
    VelocityEquivariantAcceleration,
)


class VelocityEquivariantDynamicsTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.model = VelocityEquivariantAcceleration(hidden_dim=8)
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
        rotation, _ = torch.linalg.qr(matrix)
        desired = -1 if reflection else 1
        if torch.sign(torch.linalg.det(rotation)).item() != desired:
            rotation[:, -1] = -rotation[:, -1]
        return rotation

    def _assert_equivariant(self, matrix):
        translation = torch.tensor([1.5, -2.0, 0.75])
        acceleration, _ = self.model(
            0, (self.velocity, self.position), self.condition
        )
        transformed_condition = list(self.condition)
        transformed_condition[4] = self.condition[4] @ matrix.T + translation
        transformed_acceleration, _ = self.model(
            0,
            (
                self.velocity @ matrix.T,
                self.position @ matrix.T + translation,
            ),
            tuple(transformed_condition),
        )
        error = (
            transformed_acceleration - acceleration @ matrix.T
        ).abs().max()
        self.assertLess(error.item(), 2e-6)

    def test_joint_rotation_translation_is_equivariant(self):
        self._assert_equivariant(self._orthogonal(reflection=False))

    def test_joint_reflection_translation_is_equivariant(self):
        self._assert_equivariant(self._orthogonal(reflection=True))

    def test_acceleration_is_velocity_sensitive(self):
        acceleration, _ = self.model(
            0, (self.velocity, self.position), self.condition
        )
        zero_acceleration, _ = self.model(
            0, (torch.zeros_like(self.velocity), self.position), self.condition
        )
        self.assertGreater(
            torch.linalg.vector_norm(acceleration - zero_acceleration).item(),
            1e-6,
        )

    def test_protein_condition_is_not_mutated(self):
        before = self.condition[4].clone()
        self.model(0, (self.velocity, self.position), self.condition)
        self.assertTrue(torch.equal(before, self.condition[4]))

    def test_loss_has_finite_nonzero_parameter_and_velocity_gradients(self):
        velocity = self.velocity.clone().requires_grad_(True)
        acceleration, _ = self.model(0, (velocity, self.position), self.condition)
        loss = acceleration.square().mean()
        gradients = torch.autograd.grad(
            loss, tuple(self.model.parameters()) + (velocity,)
        )
        for gradient in gradients:
            self.assertTrue(torch.isfinite(gradient).all())
        parameter_norm = torch.sqrt(
            sum(gradient.square().sum() for gradient in gradients[:-1])
        )
        self.assertGreater(parameter_norm.item(), 0)
        self.assertGreater(torch.linalg.vector_norm(gradients[-1]).item(), 0)

    def test_bounded_damping_is_small_bounded_and_dissipative(self):
        model = VelocityEquivariantAcceleration(
            hidden_dim=8,
            bounded_damping_max=0.05,
            normalize_velocity_invariants=True,
            initial_damping_fraction=0.05,
        )
        feature = model.ligand_embedding(self.condition[0])
        speed_squared = self.velocity.square().sum(dim=-1, keepdim=True)
        coefficient = model.damping_coefficient(feature, speed_squared)
        damping_acceleration = -model.scale * coefficient * self.velocity
        damping_power = (damping_acceleration * self.velocity).sum(dim=-1)
        self.assertTrue((coefficient >= 0).all())
        self.assertTrue((coefficient <= 0.05).all())
        self.assertTrue((damping_power <= 0).all())
        self.assertTrue(torch.allclose(
            coefficient,
            torch.full_like(coefficient, 0.0025),
            atol=1e-7,
        ))

    def test_bounded_normalized_variant_remains_equivariant(self):
        self.model = VelocityEquivariantAcceleration(
            hidden_dim=8,
            bounded_damping_max=0.05,
            normalize_velocity_invariants=True,
        )
        self._assert_equivariant(self._orthogonal(reflection=False))
        self._assert_equivariant(self._orthogonal(reflection=True))

    def test_invalid_bounded_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            VelocityEquivariantAcceleration(bounded_damping_max=0)
        with self.assertRaises(ValueError):
            VelocityEquivariantAcceleration(speed_squared_scale=0)
        with self.assertRaises(ValueError):
            VelocityEquivariantAcceleration(initial_damping_fraction=1)


if __name__ == "__main__":
    unittest.main()
