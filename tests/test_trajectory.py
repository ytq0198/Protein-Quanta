import unittest

import numpy as np

from protein_quanta.trajectory import Trajectory


class TrajectoryTests(unittest.TestCase):
    def test_trajectory_keeps_sample_identity_and_coordinates(self):
        coordinates = np.zeros((3, 2, 3), dtype=float)

        trajectory = Trajectory(sample_id="1abc", coordinates=coordinates)

        self.assertEqual(trajectory.sample_id, "1abc")
        self.assertEqual(trajectory.unit, "angstrom")
        np.testing.assert_array_equal(trajectory.coordinates, coordinates)
        np.testing.assert_array_equal(trajectory.atom_mask, np.array([True, True]))

    def test_trajectory_rejects_non_finite_coordinates(self):
        coordinates = np.zeros((3, 2, 3), dtype=float)
        coordinates[1, 0, 2] = np.nan

        with self.assertRaisesRegex(ValueError, "coordinates must be finite"):
            Trajectory(sample_id="bad", coordinates=coordinates)

    def test_trajectory_rejects_mask_with_wrong_atom_count(self):
        coordinates = np.zeros((3, 2, 3), dtype=float)

        with self.assertRaisesRegex(ValueError, "atom_mask must have shape"):
            Trajectory(
                sample_id="bad-mask",
                coordinates=coordinates,
                atom_mask=np.array([True]),
            )


if __name__ == "__main__":
    unittest.main()
