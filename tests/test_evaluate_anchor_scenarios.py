import unittest

import numpy as np

from scripts.evaluate_anchor_scenarios import _anchored_scenario_comparison


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


if __name__ == "__main__":
    unittest.main()
