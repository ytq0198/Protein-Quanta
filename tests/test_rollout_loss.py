import random
import unittest

import torch

from protein_quanta.rollout_loss import (
    combined_rollout_loss,
    multiscale_coordinate_smooth_l1,
    sample_rollout_segment,
    validate_horizons,
)


class RolloutLossTests(unittest.TestCase):
    def test_preregistered_horizons_and_segments_are_legal(self):
        horizons = validate_horizons((5, 10, 20, 40), frame_count=100)
        rng = random.Random(42)
        observed = set()
        for _ in range(200):
            start, end, horizon = sample_rollout_segment(100, horizons, rng)
            observed.add(horizon)
            self.assertEqual(end - start, horizon)
            self.assertGreaterEqual(start, 0)
            self.assertLess(end, 100)
        self.assertEqual(observed, set(horizons))

    def test_loss_is_mean_normalized_across_horizon(self):
        truth_short = torch.zeros(5, 2, 3)
        truth_long = torch.zeros(40, 2, 3)
        prediction_short = torch.ones_like(truth_short)
        prediction_long = torch.ones_like(truth_long)
        short = multiscale_coordinate_smooth_l1(prediction_short, truth_short)
        long = multiscale_coordinate_smooth_l1(prediction_long, truth_long)
        self.assertEqual(short.item(), long.item())

    def test_zero_weight_is_exact_baseline_in_value_and_gradient(self):
        parameter = torch.nn.Parameter(torch.tensor(1.5))
        local = (parameter - 0.25).square()
        auxiliary = (parameter + 4.0).square()
        baseline_gradient = torch.autograd.grad(local, parameter, retain_graph=True)[0]
        candidate = combined_rollout_loss(local, auxiliary, coefficient=0)
        candidate_gradient = torch.autograd.grad(candidate, parameter)[0]
        self.assertIs(candidate, local)
        self.assertEqual(candidate_gradient.item(), baseline_gradient.item())

    def test_terminal_error_backpropagates_through_all_euler_steps(self):
        initial = torch.tensor([[[-1.0, 0.5, 2.0]]], requires_grad=True)
        state = initial
        for _ in range(40):
            state = state + 0.02 * (-0.1 * state)
        terminal_loss = state.square().mean()
        gradient = torch.autograd.grad(terminal_loss, initial)[0]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(torch.linalg.vector_norm(gradient).item(), 0)

    def test_validation_rejects_ambiguous_horizon_sets(self):
        for invalid in ((), (0, 5), (5, 5), (100,)):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    validate_horizons(invalid, frame_count=100)


if __name__ == "__main__":
    unittest.main()
