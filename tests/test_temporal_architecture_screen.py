import unittest

import numpy as np
import torch

from scripts.train_temporal_architecture_screen import (
    _promotion,
    _scenario_metrics,
    _weighted_rmse,
)
from scripts.train_temporal_multiseed_rope import aggregate_results


def _scenarios(value, static=2.0):
    return {
        scenario: {
            "standardized_rmse": value,
            "structure_rmse": value,
            "step_rmse": value,
            "static_standardized_rmse": static,
            "finite": True,
        }
        for scenario in ("T1", "T2", "T3")
    }


class TemporalArchitectureScreenTests(unittest.TestCase):
    def test_weighted_rmse_uses_competition_scenario_weights(self):
        summary = _scenarios(0.0)
        summary["T1"]["standardized_rmse"] = 1.0
        summary["T2"]["standardized_rmse"] = 2.0
        summary["T3"]["standardized_rmse"] = 3.0
        self.assertAlmostEqual(_weighted_rmse(summary), 1.7)

    def test_promotion_requires_long_horizon_margin_and_static_control(self):
        results = [
            {"architecture": "mlp", "best": {"weighted_validation_rmse": 1.0, "scenarios": _scenarios(1.0)}},
            {"architecture": "gru", "best": {"weighted_validation_rmse": 0.8, "scenarios": _scenarios(0.8)}},
        ]
        decision = _promotion(results)
        self.assertTrue(decision["gru"]["passed"])
        self.assertFalse(decision["mlp"]["passed"])

    def test_static_feature_control_uses_zero_physical_step(self):
        class ZeroModel:
            def eval(self):
                return self

            def rollout(self, observed, horizon):
                return torch.zeros(
                    observed.shape[0], horizon, observed.shape[-1]
                )

        normalized = np.zeros((1, 100, 12), dtype=np.float32)
        normalized[:, :, 8:] = -3.0
        normalized[:, 1, 8:] = 0.0
        summary = _scenario_metrics(
            ZeroModel(), normalized, torch.device("cpu"), np.full(4, -3.0)
        )
        self.assertEqual(summary["T1"]["static_standardized_rmse"], 0.0)

    def test_multiseed_aggregation_uses_paired_seeds(self):
        results = []
        for seed, mlp_value, gru_value in ((0, 1.0, 0.7), (42, 1.1, 0.8), (123, 0.9, 1.0)):
            for architecture, value in (("mlp", mlp_value), ("gru", gru_value)):
                results.append(
                    {
                        "architecture": architecture,
                        "seed": seed,
                        "parameter_count": 10,
                        "best": {
                            "weighted_validation_rmse": value,
                            "scenarios": _scenarios(value),
                        },
                    }
                )
        aggregate, decision = aggregate_results(results)
        self.assertEqual(aggregate["gru"]["beats_mlp_seed_count"], 2)
        self.assertTrue(decision["gru"]["passed"])


if __name__ == "__main__":
    unittest.main()
