import unittest

from scripts.train_temporal_architecture_screen import _promotion, _weighted_rmse


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


if __name__ == "__main__":
    unittest.main()
