import json
from pathlib import Path

from scripts.plot_temporal_multiseed_rope import render_figure


def test_render_multiseed_rope_figure(tmp_path):
    report_path = Path("reports/reproduction/temporal_multiseed_rope_val.json")
    if not report_path.exists():
        return
    output = tmp_path / "figure.png"
    render_figure(json.loads(report_path.read_text(encoding="utf-8")), output)
    assert output.exists()
    assert output.stat().st_size > 10_000
