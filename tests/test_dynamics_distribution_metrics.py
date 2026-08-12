import unittest

import numpy as np

from protein_quanta.metrics import dynamics_distribution_metrics


class DynamicsDistributionMetricTests(unittest.TestCase):
    def setUp(self):
        time = np.arange(8, dtype=float)[:, None, None]
        atoms = np.array([[[0.0, 0.0, 0.0], [1.0, 0.5, 0.0], [2.0, 0.0, 0.5]]])
        phase = np.array([[[0.0, 1.0, 0.5], [1.0, 0.0, 1.0], [0.5, 1.0, 0.0]]])
        self.truth = atoms + 0.08 * np.sin(time + phase)

    def test_truth_is_distributional_identity(self):
        result = dynamics_distribution_metrics(self.truth, self.truth)

        self.assertEqual(result["rmsf_profile_mae_angstrom"], 0.0)
        self.assertAlmostEqual(result["rmsf_profile_pearson"], 1.0)
        self.assertAlmostEqual(result["rmsf_profile_spearman"], 1.0)
        self.assertEqual(result["rg_wasserstein_angstrom"], 0.0)
        self.assertEqual(result["pair_distance_wasserstein_angstrom"], 0.0)
        self.assertEqual(result["step_displacement_wasserstein_angstrom"], 0.0)
        self.assertAlmostEqual(result["step_amplitude_ratio"], 1.0)
        self.assertEqual(result["velocity_autocorrelation_mae"], 0.0)

    def test_static_control_is_identified_as_dynamics_collapse(self):
        static = np.repeat(self.truth[:1], self.truth.shape[0], axis=0)
        result = dynamics_distribution_metrics(static, self.truth)

        self.assertEqual(result["step_amplitude_ratio"], 0.0)
        self.assertGreater(result["step_displacement_wasserstein_angstrom"], 0.0)
        self.assertGreater(result["velocity_autocorrelation_mae"], 0.0)

    def test_short_or_single_atom_trajectories_are_rejected(self):
        with self.assertRaises(ValueError):
            dynamics_distribution_metrics(self.truth[:2], self.truth[:2])
        with self.assertRaises(ValueError):
            dynamics_distribution_metrics(self.truth[:, :1], self.truth[:, :1])


if __name__ == "__main__":
    unittest.main()
