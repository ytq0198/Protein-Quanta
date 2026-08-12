import unittest

import numpy as np

from protein_quanta.anchoring import (
    anchored_residual_rollout,
    validate_scenario_betas,
)


class AnchoredResidualTests(unittest.TestCase):
    def setUp(self):
        self.history = np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            ]
        )
        self.prediction = np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.9, 0.0, 0.0], [1.9, 0.0, 0.0]],
                [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
                [[3.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            ]
        )

    def test_beta_zero_keeps_neural_forecast_and_observed_history(self):
        result = anchored_residual_rollout(
            self.prediction, self.history, beta=0
        )

        np.testing.assert_allclose(result[:2], self.history)
        np.testing.assert_allclose(result[2:], self.prediction[2:])

    def test_positive_beta_decays_forecast_residual_toward_static_anchor(self):
        result = anchored_residual_rollout(
            self.prediction,
            self.history,
            beta=2,
            decay_scale_frames=2,
        )

        anchor = self.history[-1]
        self.assertLess(
            np.linalg.norm(result[-1] - anchor),
            np.linalg.norm(self.prediction[-1] - anchor),
        )

    def test_negative_beta_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            anchored_residual_rollout(
                self.prediction, self.history, beta=-1
            )

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "atom dimensions"):
            anchored_residual_rollout(
                self.prediction,
                self.history[:, :1],
                beta=1,
            )


class ScenarioBetaPolicyTests(unittest.TestCase):
    def test_complete_policy_is_normalized_to_floats(self):
        policy = validate_scenario_betas(
            {"T1": 8, "T2": 8, "T3": 1},
            ["T1", "T2", "T3"],
        )

        self.assertEqual(policy, {"T1": 8.0, "T2": 8.0, "T3": 1.0})

    def test_missing_or_unknown_scenarios_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing.*T3"):
            validate_scenario_betas(
                {"T1": 8, "T2": 8}, ["T1", "T2", "T3"]
            )
        with self.assertRaisesRegex(ValueError, "unknown.*T4"):
            validate_scenario_betas(
                {"T1": 8, "T2": 8, "T3": 1, "T4": 2},
                ["T1", "T2", "T3"],
            )

    def test_invalid_beta_values_are_rejected(self):
        for invalid in (-1, np.nan):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "finite and non-negative"):
                    validate_scenario_betas(
                        {"T1": invalid, "T2": 8, "T3": 1},
                        ["T1", "T2", "T3"],
                    )

    def test_duplicate_expected_scenarios_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "expected_scenarios.*unique"):
            validate_scenario_betas(
                {"T1": 8, "T2": 8}, ["T1", "T1", "T2"]
            )


if __name__ == "__main__":
    unittest.main()
