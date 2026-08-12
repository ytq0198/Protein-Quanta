import unittest

import torch

from protein_quanta.temporal_models import (
    ARCHITECTURES,
    TemporalFeatureForecaster,
    parameter_count,
)


class TemporalModelTests(unittest.TestCase):
    def test_all_cores_share_sequence_contract_and_rollout(self):
        sequence = torch.randn(2, 8, 12)
        for architecture in ARCHITECTURES:
            with self.subTest(architecture=architecture):
                model = TemporalFeatureForecaster(
                    12, architecture, hidden_dim=16, maximum_length=20
                )
                prediction, _ = model(sequence)
                rollout = model.rollout(sequence[:, :3], horizon=4)
                self.assertEqual(prediction.shape, sequence.shape)
                self.assertEqual(rollout.shape, (2, 4, 12))
                self.assertGreater(parameter_count(model), 0)

    def test_unknown_architecture_is_rejected(self):
        with self.assertRaises(ValueError):
            TemporalFeatureForecaster(12, "cnn")


if __name__ == "__main__":
    unittest.main()
