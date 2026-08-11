import unittest

from scripts.evaluate_anchor_residual import resolve_selected_beta, select_beta


def _candidate(beta, rmse, matching, stability, rmsf):
    return {
        "beta": beta,
        "summary": {
            "coordinate_rmse_angstrom": rmse,
            "matching_mean_angstrom": matching,
            "stability_mean_percent": stability,
            "rmsf_mae_angstrom": rmsf,
        },
    }


class AnchorResidualSelectionTests(unittest.TestCase):
    def test_selection_rejects_static_collapse_even_if_stability_is_highest(self):
        candidates = [
            _candidate(0.0, 2.0, 0.6, 75.0, 1.0),
            _candidate(0.5, 2.01, 0.5, 80.0, 1.04),
            _candidate(8.0, 1.9, 0.4, 90.0, 1.20),
        ]

        selected = select_beta(candidates)

        self.assertEqual(selected, 0.5)

    def test_selection_maximizes_stability_then_matching_then_lower_beta(self):
        candidates = [
            _candidate(0.0, 2.0, 0.6, 75.0, 1.0),
            _candidate(0.25, 2.0, 0.52, 82.0, 1.02),
            _candidate(0.5, 2.0, 0.50, 82.0, 1.02),
            _candidate(1.0, 2.0, 0.50, 82.0, 1.02),
        ]

        selected = select_beta(candidates)

        self.assertEqual(selected, 0.5)

    def test_selection_requires_beta_zero_reference(self):
        with self.assertRaisesRegex(ValueError, "beta 0"):
            select_beta([_candidate(0.5, 2.0, 0.5, 80.0, 1.0)])

    def test_test_split_uses_frozen_nonzero_beta_without_ranking_metrics(self):
        candidates = [
            _candidate(0.0, 1.0, 0.1, 99.0, 0.1),
            _candidate(4.0, 9.0, 9.0, 1.0, 9.0),
        ]

        selected = resolve_selected_beta(candidates, selection_allowed=False)

        self.assertEqual(selected, 4.0)


if __name__ == "__main__":
    unittest.main()
