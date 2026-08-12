import unittest

from scripts.plot_earlystop_tradeoff import _panel_limits, _relative_changes


class EarlyStopPlotTests(unittest.TestCase):
    def test_panel_limits_leave_label_space_around_one_sided_changes(self):
        negative = _panel_limits([-4.75, -1.04, 0.0])
        positive = _panel_limits([0.0, 1.30, 0.33])

        self.assertLess(negative[0], -4.75)
        self.assertGreater(negative[1], 0.0)
        self.assertLess(positive[0], 0.0)
        self.assertGreater(positive[1], 1.30)

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

    def test_supports_an_anchored_baseline_and_candidate(self):
        baseline = {"T1": {"anchored": {"error": 2.0}}}
        candidate = {"T1": {"anchored": {"error": 1.0}}}

        self.assertEqual(
            _relative_changes(
                baseline,
                candidate,
                "error",
                point_change=False,
                baseline_model="anchored",
                candidate_model="anchored",
            ),
            [-50.0],
        )


if __name__ == "__main__":
    unittest.main()
