import unittest

import numpy as np

from scripts.evaluate_naive_baselines import _evaluate


class NaiveBaselineEvaluationTests(unittest.TestCase):
    def test_evaluate_reports_proxy_diagnostics_without_labeling_them_official(self):
        truth = np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]],
            ]
        )

        metrics = _evaluate(truth.copy(), truth, contact_cutoff=1.5)

        self.assertEqual(metrics["diagnostic_status"], "proxy; not official score")
        self.assertAlmostEqual(metrics["aligned_rmsd_mean_angstrom"], 0.0)
        self.assertAlmostEqual(metrics["radius_of_gyration_mae_angstrom"], 0.0)
        self.assertAlmostEqual(metrics["rmsf_mae_angstrom"], 0.0)
        self.assertAlmostEqual(metrics["contact_map_agreement_mean"], 1.0)


if __name__ == "__main__":
    unittest.main()
