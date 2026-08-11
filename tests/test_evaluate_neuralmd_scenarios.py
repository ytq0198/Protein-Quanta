import unittest

from scripts.evaluate_neuralmd_scenarios import (
    REPORT_STATUS,
    _aggregate_scenarios,
    _validate_sample_ids,
)


class ScenarioEvaluationHelperTests(unittest.TestCase):
    def test_aggregates_each_scenario_and_model_independently(self):
        samples = [
            {
                "scenarios": {
                    "T1": {"comparison": {"neuralmd": {"rmse": 1.0}, "static": {"rmse": 3.0}}},
                    "T2": {"comparison": {"neuralmd": {"rmse": 5.0}, "static": {"rmse": 7.0}}},
                }
            },
            {
                "scenarios": {
                    "T1": {"comparison": {"neuralmd": {"rmse": 3.0}, "static": {"rmse": 5.0}}},
                    "T2": {"comparison": {"neuralmd": {"rmse": 7.0}, "static": {"rmse": 9.0}}},
                }
            },
        ]

        summary = _aggregate_scenarios(samples)

        self.assertEqual(summary["T1"]["neuralmd"]["rmse"], 2.0)
        self.assertEqual(summary["T1"]["static"]["rmse"], 4.0)
        self.assertEqual(summary["T2"]["neuralmd"]["rmse"], 6.0)

    def test_rejects_duplicate_sample_ids(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            _validate_sample_ids(["10GS", "10GS"])

    def test_report_status_cannot_be_mistaken_for_official_score(self):
        self.assertEqual(
            REPORT_STATUS,
            "competition-aligned proxy; not official score",
        )


if __name__ == "__main__":
    unittest.main()
