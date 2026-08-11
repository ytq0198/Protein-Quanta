import tempfile
import unittest
from pathlib import Path

try:
    import matplotlib  # noqa: F401

    HAS_MATPLOTLIB = True
except ModuleNotFoundError:
    HAS_MATPLOTLIB = False


@unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib is optional")
class StableTrainingPlotTests(unittest.TestCase):
    def test_render_figure_creates_nonempty_png(self):
        from scripts.plot_stable_training import render_figure

        log = """
epoch 1
loss pos: 12.0\tloss velocity: 0.0\t3.0s
optimizer_stats grad_norm_mean: 0.2\tgrad_norm_max: 1.5\tclipped_batches: 2\tskipped_nonfinite_batches: 0
Stability train:0.0\t\tval: 80.0\t\ttest: 81.0
epoch 2
loss pos: 10.0\tloss velocity: 0.0\t3.0s
optimizer_stats grad_norm_mean: 0.1\tgrad_norm_max: 0.5\tclipped_batches: 0\tskipped_nonfinite_batches: 0
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "training.png"

            render_figure(log, output)

            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
