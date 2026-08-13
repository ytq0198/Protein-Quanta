import unittest

from scripts.train_proar_antidrift_proxy import aggregate_and_decide


def _method(t1, t2, t3):
    values = {"T1": t1, "T2": t2, "T3": t3}
    return {
        **{
            name: {"standardized_rmse": value, "finite": True}
            for name, value in values.items()
        },
        "macro_scenario_rmse": sum(values.values()) / 3,
    }


class ProARProxyAggregationTests(unittest.TestCase):
    def test_promotion_requires_paired_wins_and_long_horizon_gain(self):
        rows = []
        for seed, t1 in ((0, 1.00), (42, 1.01), (123, 1.03)):
            rows.append(
                {
                    "seed": seed,
                    "validation": {
                        "one_pass": _method(1.0, 1.0, 1.0),
                        "alternating": _method(t1, 0.9, 0.8),
                    },
                }
            )
        aggregate, decision = aggregate_and_decide(rows)
        self.assertEqual(decision["candidate_seed_wins"], 3)
        self.assertGreaterEqual(decision["T3_relative_improvement"], 0.05)
        self.assertTrue(decision["passed"])
        self.assertLess(
            aggregate["alternating"]["macro_scenario_rmse"]["mean"],
            aggregate["one_pass"]["macro_scenario_rmse"]["mean"],
        )

    def test_short_horizon_regression_blocks_promotion(self):
        rows = [
            {
                "seed": seed,
                "validation": {
                    "one_pass": _method(1.0, 1.0, 1.0),
                    "alternating": _method(1.1, 0.8, 0.7),
                },
            }
            for seed in (0, 42, 123)
        ]
        _, decision = aggregate_and_decide(rows)
        self.assertFalse(decision["passed"])


if __name__ == "__main__":
    unittest.main()
