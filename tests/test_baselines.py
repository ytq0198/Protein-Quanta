import numpy as np
import unittest

from protein_quanta.baselines import linear_rollout, static_rollout


class BaselineRolloutTests(unittest.TestCase):
    def setUp(self):
        self.history = np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
                [[0.5, 0.0, 0.0], [1.0, 2.0, 1.0]],
            ]
        )

    def test_static_rollout_repeats_last_observed_frame(self):
        predicted = static_rollout(self.history, horizon=3)

        expected = np.repeat(self.history[-1][None, ...], repeats=3, axis=0)
        np.testing.assert_allclose(predicted, expected)

    def test_linear_rollout_extends_last_observed_velocity(self):
        predicted = linear_rollout(self.history, horizon=3)

        velocity = self.history[-1] - self.history[-2]
        expected = np.stack(
            [self.history[-1] + step * velocity for step in (1, 2, 3)]
        )
        np.testing.assert_allclose(predicted, expected)

    def test_rollouts_reject_non_positive_horizon(self):
        for rollout in (static_rollout, linear_rollout):
            with self.subTest(rollout=rollout.__name__):
                with self.assertRaisesRegex(ValueError, "horizon must be positive"):
                    rollout(self.history, horizon=0)

    def test_linear_rollout_requires_two_observed_frames(self):
        history = np.zeros((1, 4, 3), dtype=float)

        with self.assertRaisesRegex(ValueError, "at least two observed frames"):
            linear_rollout(history, horizon=1)

    def test_rollouts_require_xyz_trajectory_shape(self):
        invalid_history = np.zeros((2, 4, 2), dtype=float)

        for rollout in (static_rollout, linear_rollout):
            with self.subTest(rollout=rollout.__name__):
                with self.assertRaisesRegex(
                    ValueError, r"shape \(frames, atoms, 3\)"
                ):
                    rollout(invalid_history, horizon=1)


if __name__ == "__main__":
    unittest.main()
