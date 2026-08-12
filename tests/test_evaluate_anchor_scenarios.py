import unittest

import numpy as np

from scripts.evaluate_anchor_scenarios import (
    _anchored_scenario_comparison,
    _parse_scenario_betas,
    _resolve_anchor_policy,
)


class AnchoredScenarioComparisonTests(unittest.TestCase):
    def test_beta_zero_matches_neural_prediction(self):
        truth = np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
                [[3.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            ]
        )

        comparison = _anchored_scenario_comparison(
            truth.copy(), truth, beta=0, decay_scale_frames=98, contact_cutoff=1.5
        )

        self.assertEqual(comparison["anchored"]["coordinate_rmse_angstrom"], 0.0)
        self.assertEqual(comparison["neuralmd"]["coordinate_rmse_angstrom"], 0.0)

    def test_scenario_beta_tokens_are_parsed(self):
        self.assertEqual(
            _parse_scenario_betas(["T1=8", "T2=8", "T3=1"]),
            {"T1": 8.0, "T2": 8.0, "T3": 1.0},
        )

    def test_malformed_or_duplicate_tokens_are_rejected(self):
        for tokens, message in [
            (["T1"], "SCENARIO=BETA"),
            (["=8"], "non-empty"),
            (["T1=8", "T1=1"], "duplicate.*T1"),
        ]:
            with self.subTest(tokens=tokens):
                with self.assertRaisesRegex(ValueError, message):
                    _parse_scenario_betas(tokens)

    def test_policy_resolver_dispatches_per_scenario(self):
        policy = _resolve_anchor_policy(
            ["T1", "T2", "T3"],
            beta=None,
            scenario_beta_tokens=["T1=8", "T2=8", "T3=1"],
        )

        self.assertEqual(policy["type"], "scenario-conditioned")
        self.assertEqual(policy["beta_by_scenario"]["T1"], 8.0)
        self.assertEqual(policy["beta_by_scenario"]["T3"], 1.0)

    def test_scalar_and_scenario_policies_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            _resolve_anchor_policy(
                ["T1", "T2", "T3"],
                beta=1,
                scenario_beta_tokens=["T1=8", "T2=8", "T3=1"],
            )

    def test_omitted_policy_preserves_default_scalar_beta_four(self):
        policy = _resolve_anchor_policy(
            ["T1", "T2", "T3"], beta=None, scenario_beta_tokens=None
        )

        self.assertEqual(policy["type"], "scalar")
        self.assertEqual(
            policy["beta_by_scenario"], {"T1": 4.0, "T2": 4.0, "T3": 4.0}
        )


if __name__ == "__main__":
    unittest.main()
