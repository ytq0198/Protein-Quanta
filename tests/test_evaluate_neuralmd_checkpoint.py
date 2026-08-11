import unittest

from scripts.evaluate_neuralmd_checkpoint import _aggregate_comparisons


class NeuralMDEvaluationHelpersTests(unittest.TestCase):
    def test_aggregate_comparisons_averages_scalar_metrics_only(self):
        samples = [
            {
                "comparison": {
                    "neuralmd": {
                        "diagnostic_status": "proxy; not official score",
                        "coordinate_rmse_angstrom": 1.0,
                        "matching_by_frame_angstrom": [1.0, 2.0],
                    },
                    "static": {
                        "diagnostic_status": "proxy; not official score",
                        "coordinate_rmse_angstrom": 3.0,
                        "matching_by_frame_angstrom": [3.0, 4.0],
                    },
                }
            },
            {
                "comparison": {
                    "neuralmd": {
                        "diagnostic_status": "proxy; not official score",
                        "coordinate_rmse_angstrom": 5.0,
                        "matching_by_frame_angstrom": [5.0, 6.0],
                    },
                    "static": {
                        "diagnostic_status": "proxy; not official score",
                        "coordinate_rmse_angstrom": 7.0,
                        "matching_by_frame_angstrom": [7.0, 8.0],
                    },
                }
            },
        ]

        summary = _aggregate_comparisons(samples)

        self.assertEqual(summary["neuralmd"]["coordinate_rmse_angstrom"], 3.0)
        self.assertEqual(summary["static"]["coordinate_rmse_angstrom"], 5.0)
        self.assertNotIn("matching_by_frame_angstrom", summary["neuralmd"])

    def test_aggregate_comparisons_rejects_empty_samples(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            _aggregate_comparisons([])


if __name__ == "__main__":
    unittest.main()
