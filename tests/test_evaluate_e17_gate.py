import unittest

from scripts.evaluate_e17_gate import evaluate_gate


def _geo(value=1.0):
    return {
        "summary": {
            scenario: {
                "neuralmd": {
                    "coordinate_rmse_angstrom": value,
                    "matching_mean_angstrom": value,
                    "stability_mean_percent": 80.0,
                }
            }
            for scenario in ("T1", "T2", "T3")
        }
    }


def _dyn(candidate_amplitude=0.2):
    values = {
        "rmsf_profile_mae_angstrom": 1.0,
        "rg_wasserstein_angstrom": 1.0,
        "pair_distance_wasserstein_angstrom": 1.0,
        "step_displacement_wasserstein_angstrom": 1.0,
    }
    return {
        "summary": {
            scenario: {
                "epoch5": {**values, "step_amplitude_ratio": 0.1},
                "candidate": {**values, "step_amplitude_ratio": candidate_amplitude},
            }
            for scenario in ("T1", "T2", "T3")
        }
    }


def _phys(value=1.0, extreme=0.0):
    return {
        "summary": {
            scenario: {
                "neuralmd": {
                    "bond_length_mae_angstrom": value,
                    "extreme_bond_event_percent": extreme,
                }
            }
            for scenario in ("T1", "T2", "T3")
        }
    }


class E17GateTests(unittest.TestCase):
    def test_joint_improvement_passes(self):
        result = evaluate_gate(_geo(), _geo(), _dyn(), _phys(), _phys())
        self.assertTrue(result["passed"])

    def test_any_mandatory_failure_rejects_candidate(self):
        candidate = _geo(1.1)
        result = evaluate_gate(_geo(), candidate, _dyn(), _phys(), _phys())
        self.assertFalse(result["passed"])
        self.assertIn("coordinate_rmse_ratio:T1", result["failed_checks"])


if __name__ == "__main__":
    unittest.main()
