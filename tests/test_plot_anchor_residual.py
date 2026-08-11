import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MATPLOTLIB_AVAILABLE = importlib.util.find_spec("matplotlib") is not None

if MATPLOTLIB_AVAILABLE:
    from scripts.plot_anchor_residual import render_figure


def _report(split):
    base = {
        "coordinate_rmse_angstrom": 2.0,
        "matching_mean_angstrom": 0.5,
        "stability_mean_percent": 80.0,
        "aligned_rmsd_mean_angstrom": 0.8,
        "radius_of_gyration_mae_angstrom": 0.1,
        "rmsf_mae_angstrom": 2.0,
        "contact_map_agreement_mean": 0.95,
    }
    anchored = {
        "coordinate_rmse_angstrom": 1.9,
        "matching_mean_angstrom": 0.4,
        "stability_mean_percent": 84.0,
        "aligned_rmsd_mean_angstrom": 0.7,
        "radius_of_gyration_mae_angstrom": 0.08,
        "rmsf_mae_angstrom": 1.9,
        "contact_map_agreement_mean": 0.96,
    }
    return {
        "protocol": {"split_label": split},
        "selected_beta": 4.0,
        "candidates": [
            {"beta": 0.0, "summary": base},
            {"beta": 4.0, "summary": anchored},
        ],
    }


@unittest.skipUnless(MATPLOTLIB_AVAILABLE, "matplotlib is optional")
class AnchorResidualPlotTests(unittest.TestCase):
    def test_render_figure_creates_nonempty_png(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            validation = temp / "validation.json"
            test = temp / "test.json"
            output = temp / "tradeoff.png"
            validation.write_text(json.dumps(_report("validation")))
            test.write_text(json.dumps(_report("test")))

            render_figure(validation, test, output)

            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
