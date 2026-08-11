import importlib.util
import tempfile
import unittest
from pathlib import Path


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch

    from scripts.smoke_neuralmd import (
        _checkpoint_architecture,
        _load_checkpoint,
        _rollout_comparison,
    )


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is optional in the lightweight test environment")
class NeuralMDSmokeHelpersTests(unittest.TestCase):
    def test_load_checkpoint_uses_binding_model_state_dict(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "model.pth"
            source = torch.nn.Linear(2, 1)
            with torch.no_grad():
                source.weight.fill_(3.0)
                source.bias.fill_(4.0)
            torch.save({"binding_model": source.state_dict()}, path)
            target = torch.nn.Linear(2, 1)

            metadata = _load_checkpoint(target, path)

        self.assertEqual(metadata["missing_keys"], [])
        self.assertEqual(metadata["unexpected_keys"], [])
        self.assertEqual(metadata["sha256"], metadata["sha256"].lower())
        torch.testing.assert_close(target.weight, source.weight)
        torch.testing.assert_close(target.bias, source.bias)

    def test_load_checkpoint_rejects_missing_binding_model_key(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "model.pth"
            torch.save({"wrong_key": {}}, path)

            with self.assertRaisesRegex(ValueError, "binding_model"):
                _load_checkpoint(torch.nn.Linear(1, 1), path)

    def test_checkpoint_architecture_reads_radial_size_and_velocity_module(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "model.pth"
            torch.save(
                {
                    "binding_model": {
                        "ligand_model.radial_emb.means": torch.zeros(100),
                    }
                },
                path,
            )

            architecture = _checkpoint_architecture(path)

        self.assertEqual(architecture["frame_net_num_radial"], 100)
        self.assertEqual(
            architecture["velocity_refined_value_coefficient"],
            0,
        )

    def test_rollout_comparison_uses_first_two_frames_as_observation(self):
        truth = torch.tensor(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
                [[3.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            ]
        ).numpy()

        comparison = _rollout_comparison(truth.copy(), truth, contact_cutoff=1.5)

        self.assertAlmostEqual(
            comparison["neuralmd"]["coordinate_rmse_angstrom"],
            0.0,
        )
        self.assertAlmostEqual(
            comparison["linear"]["coordinate_rmse_angstrom"],
            0.0,
        )
        self.assertGreater(
            comparison["static"]["coordinate_rmse_angstrom"],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
