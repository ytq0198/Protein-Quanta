import unittest

from scripts.evaluate_uncertainty_gate import (
    _balanced_accuracy,
    _promotion_gates,
)


class UncertaintyGateEvaluationTests(unittest.TestCase):
    def test_balanced_accuracy_averages_class_recalls(self):
        self.assertEqual(
            _balanced_accuracy([0, 0, 1, 1], [0, 1, 1, 1]),
            0.75,
        )

    def test_promotion_requires_all_metric_and_classifier_guards(self):
        changes = {
            "T1": {
                "coordinate_rmse_percent": 1.0,
                "matching_percent": -1.0,
                "stability_points": 0.5,
                "rmsf_percent": 1.0,
            },
            "T2": {
                "coordinate_rmse_percent": -1.0,
                "matching_percent": -2.0,
                "stability_points": 0.2,
                "rmsf_percent": -1.0,
            },
            "T3": {
                "coordinate_rmse_percent": 0.0,
                "matching_percent": 0.0,
                "stability_points": 0.0,
                "rmsf_percent": 0.0,
            },
        }

        passing = _promotion_gates(
            changes, selected_strong_count=10, record_count=30, balanced_accuracy=0.7
        )
        failing = _promotion_gates(
            changes, selected_strong_count=10, record_count=30, balanced_accuracy=0.59
        )

        self.assertTrue(all(passing.values()))
        self.assertFalse(failing["balanced_accuracy_at_least_0_60"])


if __name__ == "__main__":
    unittest.main()
