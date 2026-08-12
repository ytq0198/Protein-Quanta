import unittest

import numpy as np

from protein_quanta.uncertainty_gate import (
    FEATURE_NAMES,
    extract_gate_features,
    fit_logistic_gate,
    grouped_leave_one_out,
    predict_gate_probability,
    strong_anchor_label,
)


class GateFeatureTests(unittest.TestCase):
    def setUp(self):
        self.history = np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
                [[0.1, 0.2, 0.0], [1.1, 0.2, 0.0], [0.1, 2.2, 0.0]],
            ]
        )
        self.prediction = np.concatenate(
            [
                self.history,
                self.history[-1:] + np.array([[[0.2, 0.0, 0.0]]]),
                self.history[-1:] + np.array([[[0.4, 0.1, 0.0]]]),
            ],
            axis=0,
        )

    def test_features_are_rotation_and_translation_invariant(self):
        rotation = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        translation = np.array([3.0, -2.0, 5.0])

        original = extract_gate_features(self.prediction, self.history, "T2")
        transformed = extract_gate_features(
            self.prediction @ rotation + translation,
            self.history @ rotation + translation,
            "T2",
        )

        self.assertEqual(tuple(original), FEATURE_NAMES)
        np.testing.assert_allclose(
            list(original.values()),
            list(transformed.values()),
            atol=1e-12,
        )

    def test_unknown_scenario_and_invalid_shapes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "scenario"):
            extract_gate_features(self.prediction, self.history, "T4")
        with self.assertRaisesRegex(ValueError, "history"):
            extract_gate_features(self.prediction, self.history[:1], "T1")


class GateLabelTests(unittest.TestCase):
    def test_label_requires_both_benefit_and_safety(self):
        baseline = {
            "coordinate_rmse_angstrom": 2.0,
            "matching_mean_angstrom": 1.0,
            "stability_mean_percent": 80.0,
            "rmsf_mae_angstrom": 1.0,
        }
        useful = {
            "coordinate_rmse_angstrom": 2.03,
            "matching_mean_angstrom": 0.9,
            "stability_mean_percent": 81.0,
            "rmsf_mae_angstrom": 1.04,
        }
        unsafe = {**useful, "coordinate_rmse_angstrom": 2.05}

        self.assertTrue(strong_anchor_label(baseline, useful))
        self.assertFalse(strong_anchor_label(baseline, unsafe))


class LogisticGateTests(unittest.TestCase):
    def test_regularized_logistic_gate_learns_a_simple_signal(self):
        features = np.array([[-2.0], [-1.0], [1.0], [2.0]])
        labels = np.array([0, 0, 1, 1])

        model = fit_logistic_gate(features, labels, l2=0.1)
        probabilities = predict_gate_probability(model, features)

        self.assertTrue(np.all(probabilities[:2] < 0.5))
        self.assertTrue(np.all(probabilities[2:] >= 0.5))

    def test_grouped_leave_one_out_never_trains_on_held_out_complex(self):
        records = []
        for group_index, group in enumerate(("A", "B", "C", "D")):
            for scenario_index, scenario in enumerate(("T1", "T2", "T3")):
                records.append(
                    {
                        "sample_id": group,
                        "scenario": scenario,
                        "features": np.array(
                            [group_index, scenario_index, group_index + scenario_index],
                            dtype=float,
                        ),
                        "label": int((group_index + scenario_index) % 2 == 0),
                    }
                )

        result = grouped_leave_one_out(records, l2=10.0)

        self.assertEqual(len(result["predictions"]), 12)
        for fold in result["folds"]:
            self.assertNotIn(fold["held_out_sample_id"], fold["training_sample_ids"])
            self.assertEqual(fold["held_out_record_count"], 3)
        predicted_keys = {
            (row["sample_id"], row["scenario"])
            for row in result["predictions"]
        }
        self.assertEqual(len(predicted_keys), 12)


if __name__ == "__main__":
    unittest.main()
