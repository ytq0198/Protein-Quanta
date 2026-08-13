import unittest

import torch

from protein_quanta.temporal_models import (
    ARCHITECTURES,
    TemporalFeatureForecaster,
    apply_rotary_position,
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

    def test_rope_preserves_vector_norm_and_changes_nonzero_positions(self):
        values = torch.randn(2, 4, 7, 4)
        rotated = apply_rotary_position(values)
        torch.testing.assert_close(
            torch.linalg.vector_norm(rotated, dim=-1),
            torch.linalg.vector_norm(values, dim=-1),
        )
        torch.testing.assert_close(rotated[..., 0, :], values[..., 0, :])
        self.assertFalse(torch.equal(rotated[..., 1:, :], values[..., 1:, :]))


if __name__ == "__main__":
    unittest.main()
