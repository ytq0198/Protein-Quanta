import unittest

from scripts.confirm_temporal_multiscale_official_validation import summarize


def _row(seed, t1, t2, t3):
    scenarios = {
        name: {"standardized_rmse": value}
        for name, value in zip(("T1", "T2", "T3"), (t1, t2, t3))
    }
    return {
        "seed": seed,
        "macro_scenario_rmse": (t1 + t2 + t3) / 3,
        "scenarios": scenarios,
    }


class ConfirmationSummaryTests(unittest.TestCase):
    def test_confirmation_requires_t3_direction_and_paired_wins(self):
        baseline = [_row(seed, 1, 1, 1) for seed in (0, 42, 123)]
        candidate = [_row(seed, 1, 0.9, 0.9) for seed in (0, 42, 123)]
        _, decision = summarize(candidate, baseline)
        self.assertTrue(decision["passed"])
        self.assertTrue(decision["T3_strong_marker_passed"])


if __name__ == "__main__":
    unittest.main()
