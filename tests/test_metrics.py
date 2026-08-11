import unittest

import numpy as np

from protein_quanta.metrics import (
    aligned_rmsd,
    contact_map_agreement,
    coordinate_mae,
    coordinate_rmse,
    distance_matching,
    distance_stability,
    radius_of_gyration_error,
    rmsf_error,
)


class ReconstructionMetricTests(unittest.TestCase):
    def test_coordinate_metrics_match_known_values(self):
        truth = np.zeros((1, 2, 3), dtype=float)
        prediction = np.array([[[1.0, -1.0, 1.0], [0.0, 0.0, 0.0]]])

        self.assertAlmostEqual(coordinate_mae(prediction, truth), 0.5)
        self.assertAlmostEqual(coordinate_rmse(prediction, truth), np.sqrt(0.5))

    def test_coordinate_metrics_apply_atom_mask(self):
        truth = np.zeros((1, 2, 3), dtype=float)
        prediction = np.array([[[1.0, 1.0, 1.0], [100.0, 100.0, 100.0]]])

        mask = np.array([True, False])

        self.assertAlmostEqual(coordinate_mae(prediction, truth, mask), 1.0)
        self.assertAlmostEqual(coordinate_rmse(prediction, truth, mask), 1.0)

    def test_coordinate_metrics_reject_shape_mismatch(self):
        truth = np.zeros((1, 2, 3), dtype=float)
        prediction = np.zeros((2, 2, 3), dtype=float)

        for metric in (coordinate_mae, coordinate_rmse):
            with self.subTest(metric=metric.__name__):
                with self.assertRaisesRegex(ValueError, "matching shapes"):
                    metric(prediction, truth)

    def test_distance_matching_reproduces_pairwise_distance_rmse(self):
        truth = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
        prediction = np.array([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])

        per_frame = distance_matching(prediction, truth)

        np.testing.assert_allclose(per_frame, np.array([np.sqrt(0.5)]))

    def test_distance_stability_reproduces_neuralmd_percentage(self):
        truth = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
        prediction = np.array([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])

        per_frame = distance_stability(prediction, truth, threshold=0.5)

        np.testing.assert_allclose(per_frame, np.array([50.0]))

    def test_aligned_rmsd_removes_rigid_rotation_and_translation(self):
        truth = np.array(
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]
        )
        prediction = np.array(
            [[[3.0, -2.0, 1.0], [3.0, -1.0, 1.0], [2.0, -2.0, 1.0]]]
        )

        np.testing.assert_allclose(aligned_rmsd(prediction, truth), [0.0], atol=1e-12)

    def test_radius_of_gyration_error_detects_expansion(self):
        truth = np.array([[[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
        prediction = np.array([[[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])

        np.testing.assert_allclose(radius_of_gyration_error(prediction, truth), [1.0])

    def test_rmsf_error_compares_per_atom_fluctuation_amplitudes(self):
        truth = np.array(
            [
                [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ]
        )
        prediction = np.array(
            [
                [[-2.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ]
        )

        self.assertAlmostEqual(rmsf_error(prediction, truth), 0.5)

    def test_contact_map_agreement_excludes_diagonal_and_symmetric_duplicates(self):
        truth = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
        prediction = np.array([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])

        np.testing.assert_allclose(
            contact_map_agreement(prediction, truth, cutoff=1.5),
            [0.0],
        )


if __name__ == "__main__":
    unittest.main()
