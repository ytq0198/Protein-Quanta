import unittest

from scripts.summarize_updated_temporal_results import summarize


def _result(architecture, seed, values):
    return {
        "architecture": architecture,
        "seed": seed,
        "parameter_count": 10,
        "best": {
            "weighted_validation_rmse": 999.0,
            "scenarios": {
                name: {
                    "standardized_rmse": value,
                    "structure_rmse": value,
                    "step_rmse": value,
                    "static_standardized_rmse": 2.0,
                }
                for name, value in zip(("T1", "T2", "T3"), values)
            },
        },
    }


class UpdatedTemporalSummaryTests(unittest.TestCase):
    def test_ignores_historical_weighted_field(self):
        rows = []
        for seed in (0, 42, 123):
            rows.append(_result("mlp", seed, (1.0, 1.0, 1.0)))
            rows.append(_result("transformer", seed, (0.9, 0.8, 0.7)))
        report = summarize({"protocol": {"test_accessed": False}, "results": rows})
        self.assertAlmostEqual(
            report["aggregate"]["transformer"]["macro_scenario_rmse"]["mean"],
            0.8,
        )
        self.assertTrue(report["promotion"]["transformer"]["passed"])
        self.assertFalse(report["protocol"]["test_accessed"])


if __name__ == "__main__":
    unittest.main()
