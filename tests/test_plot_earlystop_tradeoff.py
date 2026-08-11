import unittest

from scripts.plot_earlystop_tradeoff import _relative_changes


class EarlyStopPlotTests(unittest.TestCase):
    def test_computes_percent_and_point_changes(self):
        baseline = {"T1": {"neuralmd": {"error": 2.0, "stability": 80.0}}}
        candidate = {"T1": {"neuralmd": {"error": 1.0, "stability": 82.0}}}

        self.assertEqual(
            _relative_changes(baseline, candidate, "error", point_change=False),
            [-50.0],
        )
        self.assertEqual(
            _relative_changes(baseline, candidate, "stability", point_change=True),
            [2.0],
        )

    def test_supports_an_anchored_candidate(self):
        baseline = {"T1": {"neuralmd": {"error": 2.0}}}
        candidate = {"T1": {"anchored": {"error": 1.0}}}

        self.assertEqual(
            _relative_changes(
                baseline,
                candidate,
                "error",
                point_change=False,
                candidate_model="anchored",
            ),
            [-50.0],
        )


if __name__ == "__main__":
    unittest.main()
