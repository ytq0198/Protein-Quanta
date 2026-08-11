import unittest

import numpy as np

from protein_quanta.collision import (
    binding_collision_rate,
    intramolecular_collision_rate,
)


class CollisionMetricTests(unittest.TestCase):
    def test_intramolecular_rate_excludes_self_and_duplicate_pairs(self):
        trajectory = np.array(
            [[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [2.0, 0.0, 0.0]]]
        )

        rate = intramolecular_collision_rate(
            trajectory, covalent_radii=np.array([0.5, 0.5, 0.5])
        )

        np.testing.assert_allclose(rate, [100.0 / 3.0])

    def test_binding_rate_uses_all_ligand_protein_pairs(self):
        ligand = np.array([[[0.0, 0.0, 0.0]]])
        protein = np.array([[0.5, 0.0, 0.0], [2.0, 0.0, 0.0]])

        rate = binding_collision_rate(
            ligand,
            ligand_covalent_radii=np.array([0.5]),
            protein_positions=protein,
            protein_covalent_radii=np.array([0.5, 0.5]),
        )

        np.testing.assert_allclose(rate, [50.0])

    def test_rejects_invalid_radius_shapes(self):
        trajectory = np.zeros((2, 2, 3))
        with self.assertRaises(ValueError):
            intramolecular_collision_rate(trajectory, np.array([0.5]))


if __name__ == "__main__":
    unittest.main()
