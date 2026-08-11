import unittest

from protein_quanta.training_log import parse_neuralmd_training_log


class NeuralMDTrainingLogTests(unittest.TestCase):
    def test_parser_attaches_loss_gradient_and_validation_metrics_to_epochs(self):
        log = """
epoch 5
loss pos: 12.50000\tloss velocity: 0.00000\t3.200s
optimizer_stats grad_norm_mean: 0.250000\tgrad_norm_max: 1.500000\tclipped_batches: 2\tskipped_nonfinite_batches: 0
MAE train: 0.00000\t\tval: 2.10000\t\ttest: 2.00000
RMSE train: 0.00000\t\tval: 2.50000\t\ttest: 2.40000
hr MAE train: 0.00000\t\tval: 0.50000\t\ttest: 0.40000
Stability train:0.00000\t\tval: 80.00000\t\ttest: 81.00000
epoch 6
loss pos: 10.00000\tloss velocity: 0.00000\t3.000s
"""

        rows = parse_neuralmd_training_log(log)

        self.assertEqual([row["epoch"] for row in rows], [5, 6])
        self.assertEqual(rows[0]["loss_pos"], 12.5)
        self.assertEqual(rows[0]["grad_norm_max"], 1.5)
        self.assertEqual(rows[0]["clipped_batches"], 2)
        self.assertEqual(rows[0]["val_coordinate_rmse"], 2.5)
        self.assertEqual(rows[0]["val_matching"], 0.5)
        self.assertEqual(rows[0]["val_stability"], 80.0)
        self.assertNotIn("val_stability", rows[1])

    def test_parser_supports_uninstrumented_baseline_logs(self):
        rows = parse_neuralmd_training_log(
            "epoch 48\nloss pos: 4531.00000\tloss velocity: 0.00000\t3.0s\n"
        )

        self.assertEqual(rows, [{"epoch": 48, "loss_pos": 4531.0}])


if __name__ == "__main__":
    unittest.main()
