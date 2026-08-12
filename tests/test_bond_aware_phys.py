import unittest

import numpy as np

from protein_quanta.collision import (
    bond_length_diagnostics,
    nonbonded_intramolecular_collision_rate,
)


class BondAwarePhysTests(unittest.TestCase):
    def test_nonbonded_rate_excludes_one_and_two_hop_neighbours(self):
        trajectory = np.array(
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.1, 0.0, 0.0]]]
        )
        rates = nonbonded_intramolecular_collision_rate(
            trajectory,
            np.full(4, 0.6),
            np.array([[0, 1], [1, 2]]),
        )
        self.assertAlmostEqual(float(rates[0]), 2.0 / 3.0 * 100.0)

    def test_bond_length_diagnostics_are_zero_on_truth(self):
        truth = np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [2.2, 0.0, 0.0]],
            ]
        )
        result = bond_length_diagnostics(truth, truth, np.array([[0, 1], [1, 2]]))
        self.assertEqual(result["bond_length_mae_angstrom"], 0.0)
        self.assertEqual(result["bond_length_violation_percent"], 0.0)
        self.assertEqual(result["extreme_bond_event_percent"], 0.0)

    def test_bond_length_diagnostics_detect_extreme_stretch(self):
        truth = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
        prediction = np.array([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
        result = bond_length_diagnostics(prediction, truth, np.array([[0, 1]]))
        self.assertEqual(result["bond_length_relative_mae"], 1.0)
        self.assertEqual(result["bond_length_violation_percent"], 100.0)
        self.assertEqual(result["extreme_bond_event_percent"], 100.0)

    def test_invalid_bond_indices_are_rejected(self):
        trajectory = np.zeros((1, 2, 3))
        with self.assertRaises(ValueError):
            bond_length_diagnostics(trajectory, trajectory, np.array([[0, 2]]))


if __name__ == "__main__":
    unittest.main()
