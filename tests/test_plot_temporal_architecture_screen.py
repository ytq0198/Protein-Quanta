import json
import tempfile
import unittest
from pathlib import Path

from scripts.plot_temporal_architecture_screen import render_figure


class TemporalArchitecturePlotTests(unittest.TestCase):
    def test_render_from_frozen_report(self):
        report_path = (
            Path(__file__).resolve().parents[1]
            / "reports"
            / "reproduction"
            / "temporal_architecture_screen_val.json"
        )
        if not report_path.exists():
            self.skipTest("frozen architecture report not present")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "screen.png"
            render_figure(report, output)
            self.assertGreater(output.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
