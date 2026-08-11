import unittest
from pathlib import Path

from protein_quanta.neuralmd_training import build_training_command


class NeuralMDTrainingCommandTests(unittest.TestCase):
    def test_official_segment_sampling_flag_is_explicit(self):
        command = build_training_command(
            output_dir=Path("/runs/corrected"),
            epochs=100,
            gpu_index=2,
            max_grad_norm=0.0,
        )

        self.assertIn("--no_NeuralMD_Binding_start_with_first_frame", command)
        frame_index = command.index("--NeuralMD_Binding_frame_num")
        self.assertEqual(command[frame_index + 1], "20")
        self.assertIn("--no_MLP_velocity", command)
        self.assertEqual(command[command.index("--device") + 1], "2")
        self.assertEqual(command[command.index("--max_grad_norm") + 1], "0.0")

    def test_training_command_rejects_invalid_runtime_values(self):
        with self.assertRaises(ValueError):
            build_training_command(Path("/runs/x"), epochs=0)
        with self.assertRaises(ValueError):
            build_training_command(Path("/runs/x"), epochs=1, max_grad_norm=-1)


if __name__ == "__main__":
    unittest.main()
