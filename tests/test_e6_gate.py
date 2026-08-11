import unittest

from protein_quanta.e6_gate import evaluate_e6_gate


class E6GateTests(unittest.TestCase):
    def setUp(self):
        self.baseline = {
            "T1": {"neuralmd": {"coordinate_rmse_angstrom": 1.0}},
            "T2": {
                "neuralmd": {
                    "matching_mean_angstrom": 1.0,
                    "stability_mean_percent": 80.0,
                }
            },
            "T3": {
                "neuralmd": {
                    "matching_mean_angstrom": 1.0,
                    "stability_mean_percent": 70.0,
                    "rmsf_mae_angstrom": 1.0,
                }
            },
        }

    def test_passes_with_t3_matching_improvement(self):
        candidate = {
            "T1": {"neuralmd": {"coordinate_rmse_angstrom": 1.02}},
            "T2": {
                "neuralmd": {
                    "matching_mean_angstrom": 1.02,
                    "stability_mean_percent": 79.0,
                }
            },
            "T3": {
                "neuralmd": {
                    "matching_mean_angstrom": 0.95,
                    "stability_mean_percent": 69.0,
                    "rmsf_mae_angstrom": 1.05,
                }
            },
        }

        result = evaluate_e6_gate(
            self.baseline, candidate, collision_relative_regression=0.02
        )

        self.assertTrue(result["passed"])
        self.assertTrue(result["checks"]["t3_geometry_improvement"])

    def test_fails_if_collision_regression_is_material(self):
        candidate = {
            "T1": {"neuralmd": {"coordinate_rmse_angstrom": 1.0}},
            "T2": {
                "neuralmd": {
                    "matching_mean_angstrom": 1.0,
                    "stability_mean_percent": 80.0,
                }
            },
            "T3": {
                "neuralmd": {
                    "matching_mean_angstrom": 0.9,
                    "stability_mean_percent": 72.0,
                    "rmsf_mae_angstrom": 1.0,
                }
            },
        }

        result = evaluate_e6_gate(
            self.baseline, candidate, collision_relative_regression=0.021
        )

        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["collision_regression"])


if __name__ == "__main__":
    unittest.main()
