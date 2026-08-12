import unittest

from scripts.plot_uncertainty_gate import _confusion_counts


class UncertaintyGatePlotTests(unittest.TestCase):
    def test_confusion_counts_follow_binary_label_order(self):
        records = [
            {"label": 0, "selected_beta": 1.0},
            {"label": 0, "selected_beta": 8.0},
            {"label": 1, "selected_beta": 1.0},
            {"label": 1, "selected_beta": 8.0},
            {"label": 1, "selected_beta": 8.0},
        ]

        self.assertEqual(
            _confusion_counts(records),
            {"true_negative": 1, "false_positive": 1, "false_negative": 1, "true_positive": 2},
        )


if __name__ == "__main__":
    unittest.main()
