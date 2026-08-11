import unittest

from scripts.plot_scenario_baselines import _metric_values


class ScenarioBaselinePlotTests(unittest.TestCase):
    def test_extracts_models_in_scenario_order(self):
        summary = {
            "T1": {"neuralmd": {"rmse": 1}, "static": {"rmse": 2}},
            "T2": {"neuralmd": {"rmse": 3}, "static": {"rmse": 4}},
            "T3": {"neuralmd": {"rmse": 5}, "static": {"rmse": 6}},
        }

        values = _metric_values(summary, "rmse")

        self.assertEqual(values["neuralmd"], [1.0, 3.0, 5.0])
        self.assertEqual(values["static"], [2.0, 4.0, 6.0])


if __name__ == "__main__":
    unittest.main()
