import random
import unittest
from types import SimpleNamespace

import torch

from protein_quanta.neuralmd_multiscale import (
    neuralmd_condition,
    neuralmd_ode_rollout,
    sampled_multiscale_neuralmd_loss,
)


class _ConstantAcceleration(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.1))

    def forward(self, time, state, condition):
        velocity, position = state
        protein_centroid = condition[4].mean(dim=0, keepdim=True)
        return self.scale * (protein_centroid - position), velocity


def _euler_odeint(model, state, times, condition, method, options):
    velocity, position = state
    velocities = [velocity]
    positions = [position]
    for left, right in zip(times[:-1], times[1:]):
        dt = right - left
        acceleration, derivative_position = model(left, (velocity, position), condition)
        velocity = velocity + dt * acceleration
        position = position + dt * derivative_position
        velocities.append(velocity)
        positions.append(position)
    return torch.stack(velocities), torch.stack(positions)


class NeuralMDMultiscaleTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        frames, atoms = 50, 3
        self.batch = SimpleNamespace(
            ligand_trajectory_pos=torch.randn(atoms, frames, 3),
            ligand_x=torch.ones(atoms, dtype=torch.long),
            batch_ligand=torch.zeros(atoms, dtype=torch.long),
            ligand_mass=torch.ones(atoms),
            protein_pos=torch.randn(6, 3),
            mask_n=torch.tensor([True, False, False, True, False, False]),
            mask_ca=torch.tensor([False, True, False, False, True, False]),
            mask_c=torch.tensor([False, False, True, False, False, True]),
            protein_backbone_residue=torch.ones(2),
            batch_residue=torch.zeros(2, dtype=torch.long),
        )
        self.model = _ConstantAcceleration()

    def test_condition_keeps_protein_coordinates_separate_and_unchanged(self):
        before = self.batch.protein_pos.clone()
        condition = neuralmd_condition(self.batch)
        neuralmd_ode_rollout(
            self.model, _euler_odeint, self.batch, start=3, horizon=5
        )
        self.assertTrue(torch.equal(before, self.batch.protein_pos))
        self.assertTrue(torch.equal(condition[4], before[self.batch.mask_ca]))

    def test_sampled_loss_is_differentiable(self):
        loss, metadata = sampled_multiscale_neuralmd_loss(
            self.model,
            _euler_odeint,
            self.batch,
            horizons=(5, 10, 20, 40),
            rng=random.Random(42),
        )
        gradient = torch.autograd.grad(loss, self.model.scale)[0]
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(gradient))
        self.assertIn(metadata["horizon"], (5, 10, 20, 40))

    def test_terminal_error_reaches_initial_position(self):
        initial = self.batch.ligand_trajectory_pos[:, 0, :].clone().requires_grad_(True)
        _, positions = neuralmd_ode_rollout(
            self.model,
            _euler_odeint,
            self.batch,
            start=0,
            horizon=40,
            initial_position=initial,
        )
        terminal_loss = positions[-1].square().mean()
        gradient = torch.autograd.grad(terminal_loss, initial)[0]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(torch.linalg.vector_norm(gradient).item(), 0)


if __name__ == "__main__":
    unittest.main()
