import unittest

import torch

from protein_quanta.dense_equivariant_dynamics import DenseEquivariantAcceleration


class DenseEquivariantDynamicsTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(8)
        self.model = DenseEquivariantAcceleration(hidden_dim=8)
        self.position = torch.randn(4, 3)
        self.velocity = torch.randn(4, 3)
        self.condition = (
            torch.tensor([6, 7, 8, 6]), torch.tensor([0, 0, 1, 1]),
            torch.ones(4), torch.empty(0), torch.randn(6, 3), torch.empty(0),
            torch.tensor([1, 2, 3, 4, 5, 6]), torch.tensor([0, 0, 0, 1, 1, 1]),
        )

    def test_joint_rigid_transform_is_equivariant(self):
        matrix = torch.randn(3, 3)
        rotation, _ = torch.linalg.qr(matrix)
        if torch.linalg.det(rotation) < 0:
            rotation[:, -1] = -rotation[:, -1]
        translation = torch.tensor([1.5, -2.0, 0.75])
        acceleration, _ = self.model(0, (self.velocity, self.position), self.condition)
        transformed_condition = list(self.condition)
        transformed_condition[4] = self.condition[4] @ rotation.T + translation
        transformed_acceleration, _ = self.model(
            0,
            (self.velocity @ rotation.T, self.position @ rotation.T + translation),
            tuple(transformed_condition),
        )
        error = (transformed_acceleration - acceleration @ rotation.T).abs().max()
        self.assertLess(error.item(), 1e-6)

    def test_protein_condition_is_not_mutated(self):
        before = self.condition[4].clone()
        self.model(0, (self.velocity, self.position), self.condition)
        self.assertTrue(torch.equal(before, self.condition[4]))

    def test_loss_has_finite_nonzero_parameter_gradient(self):
        acceleration, _ = self.model(0, (self.velocity, self.position), self.condition)
        loss = acceleration.square().mean()
        gradients = torch.autograd.grad(loss, tuple(self.model.parameters()))
        norm = torch.sqrt(sum(gradient.square().sum() for gradient in gradients))
        self.assertTrue(torch.isfinite(norm))
        self.assertGreater(norm.item(), 0)


if __name__ == "__main__":
    unittest.main()
