import unittest

import numpy as np

from protein_quanta.temporal_features import FEATURE_NAMES, invariant_trajectory_features


class TemporalFeatureTests(unittest.TestCase):
    def test_features_are_rigid_motion_invariant(self):
        rng = np.random.default_rng(3)
        trajectory = rng.normal(size=(8, 5, 3))
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        transformed = trajectory @ rotation + np.array([4.0, -2.0, 7.0])
        np.testing.assert_allclose(
            invariant_trajectory_features(trajectory),
            invariant_trajectory_features(transformed),
            atol=1e-10,
        )

    def test_feature_shape_and_first_step_convention(self):
        trajectory = np.arange(6 * 3 * 3, dtype=float).reshape(6, 3, 3)
        features = invariant_trajectory_features(trajectory)
        self.assertEqual(features.shape, (6, len(FEATURE_NAMES)))
        np.testing.assert_array_equal(features[0, 8:], 0.0)


if __name__ == "__main__":
    unittest.main()
