import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MATPLOTLIB_AVAILABLE = importlib.util.find_spec("matplotlib") is not None

if MATPLOTLIB_AVAILABLE:
    from scripts.plot_neuralmd_test import render_figures


@unittest.skipUnless(MATPLOTLIB_AVAILABLE, "matplotlib is optional")
class NeuralMDTestPlotTests(unittest.TestCase):
    def test_render_figures_creates_summary_and_per_sample_pngs(self):
        metrics = {
            "coordinate_rmse_angstrom": 1.0,
            "matching_mean_angstrom": 0.5,
            "stability_mean_percent": 80.0,
            "aligned_rmsd_mean_angstrom": 0.8,
            "radius_of_gyration_mae_angstrom": 0.1,
            "rmsf_mae_angstrom": 2.0,
            "contact_map_agreement_mean": 0.96,
        }
        report = {
            "summary": {
                "neuralmd": metrics,
                "static": {key: value * 1.1 for key, value in metrics.items()},
                "linear": {key: value * 10 for key, value in metrics.items()},
            },
            "samples": [
                {
                    "sample_id": "TEST",
                    "comparison": {
                        "neuralmd": metrics,
                        "static": {
                            key: value * 1.1 for key, value in metrics.items()
                        },
                        "linear": {
                            key: value * 10 for key, value in metrics.items()
                        },
                    },
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            input_path = temp / "report.json"
            input_path.write_text(json.dumps(report), encoding="utf-8")

            outputs = render_figures(input_path, temp)

            self.assertEqual(len(outputs), 2)
            self.assertTrue(all(path.exists() and path.stat().st_size for path in outputs))


if __name__ == "__main__":
    unittest.main()
