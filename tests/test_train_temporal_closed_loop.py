import unittest

import torch

from protein_quanta.temporal_models import TemporalFeatureForecaster
from scripts.train_temporal_closed_loop import aggregate_and_decide, differentiable_rollout


def _scenarios(t1, t2, t3):
    return {
        name: {"standardized_rmse": value, "finite": True}
        for name, value in zip(("T1", "T2", "T3"), (t1, t2, t3))
    }


class ClosedLoopTrainingTests(unittest.TestCase):
    def test_rollout_retains_gradient_through_generated_states(self):
        model = TemporalFeatureForecaster(4, "transformer", hidden_dim=8, layers=1, transformer_heads=2)
        observed = torch.randn(2, 5, 4)
        prediction = differentiable_rollout(model, observed, 3)
        self.assertEqual(tuple(prediction.shape), (2, 3, 4))
        prediction.square().mean().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_promotion_requires_long_horizon_mechanism(self):
        results = []
        references = []
        for seed in (0, 42, 123):
            references.append(
                {"architecture": "transformer", "seed": seed, "best": {"scenarios": _scenarios(1.0, 1.0, 1.0)}}
            )
            results.append(
                {"seed": seed, "final": {"scenarios": _scenarios(1.0, 0.9, 0.8)}}
            )
        _, decision = aggregate_and_decide(results, {"results": references})
        self.assertTrue(decision["passed"])

    def test_t3_under_five_percent_is_no_go(self):
        results = []
        references = []
        for seed in (0, 42, 123):
            references.append(
                {"architecture": "transformer", "seed": seed, "best": {"scenarios": _scenarios(1.0, 1.0, 1.0)}}
            )
            results.append(
                {"seed": seed, "final": {"scenarios": _scenarios(0.9, 0.9, 0.96)}}
            )
        _, decision = aggregate_and_decide(results, {"results": references})
        self.assertFalse(decision["passed"])


if __name__ == "__main__":
    unittest.main()
