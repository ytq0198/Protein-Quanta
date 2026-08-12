import unittest

import numpy as np

from scripts.evaluate_bond_aware_phys import _metrics


class EvaluateBondAwarePhysTests(unittest.TestCase):
    def test_metrics_report_truth_relative_bonds_and_nonbonded_clashes(self):
        truth = np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [2.2, 0.0, 0.0], [4.2, 0.0, 0.0]],
            ]
        )
        prediction = truth.copy()
        prediction[:, 3, 0] += 0.2

        result = _metrics(
            prediction,
            truth,
            np.array([[0, 1], [1, 2], [2, 3]]),
            np.full(4, 0.4),
        )

        self.assertGreater(result["bond_length_mae_angstrom"], 0.0)
        self.assertEqual(result["extreme_bond_event_percent"], 0.0)
        self.assertEqual(result["nonbonded_collision_mean_percent"], 0.0)


if __name__ == "__main__":
    unittest.main()
