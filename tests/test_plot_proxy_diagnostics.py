import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MATPLOTLIB_AVAILABLE = importlib.util.find_spec("matplotlib") is not None

if MATPLOTLIB_AVAILABLE:
    from scripts.plot_proxy_diagnostics import render_figures


@unittest.skipUnless(MATPLOTLIB_AVAILABLE, "matplotlib is optional")
class ProxyDiagnosticPlotTests(unittest.TestCase):
    def test_render_figures_creates_summary_and_rollout_pngs(self):
        report = {
            "protocol": {"status": "T1 proxy; not an official competition score"},
            "summary": {
                "static": {
                    "aligned_rmsd_mean_angstrom": 1.0,
                    "radius_of_gyration_mae_angstrom": 0.2,
                    "rmsf_mae_angstrom": 0.5,
                    "contact_map_agreement_mean": 0.9,
                },
                "linear": {
                    "aligned_rmsd_mean_angstrom": 4.0,
                    "radius_of_gyration_mae_angstrom": 3.0,
                    "rmsf_mae_angstrom": 2.0,
                    "contact_map_agreement_mean": 0.4,
                },
            },
            "samples": [
                {
                    "baseline": "static",
                    "aligned_rmsd_by_frame_angstrom": [0.5, 1.0],
                    "radius_of_gyration_error_by_frame_angstrom": [0.1, 0.2],
                },
                {
                    "baseline": "linear",
                    "aligned_rmsd_by_frame_angstrom": [1.0, 4.0],
                    "radius_of_gyration_error_by_frame_angstrom": [0.5, 3.0],
                },
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "report.json"
            input_path.write_text(json.dumps(report), encoding="utf-8")
            output_dir = Path(temp_dir) / "figures"

            outputs = render_figures(input_path, output_dir)

            self.assertEqual(len(outputs), 2)
            for output in outputs:
                self.assertTrue(output.is_file())
                self.assertEqual(output.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
