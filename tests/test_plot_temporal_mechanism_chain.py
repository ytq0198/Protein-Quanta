import json
from pathlib import Path

from scripts.plot_temporal_mechanism_chain import render_figure


def _scenario(value):
    return {name: {"standardized_rmse": value} for name in ("T1", "T2", "T3")}


def test_render_temporal_mechanism_chain(tmp_path):
    architectures = {
        "results": [
            {"architecture": name, "best": {"scenarios": _scenario(value)}}
            for name, value in (("transformer", 1.0), ("transformer_rope", 0.9))
        ]
    }
    noise = {"aggregate": {"scenarios": {name: {"noise": {"mean": 1.1}} for name in ("T1", "T2", "T3")}}}
    proar = {"aggregate": {"alternating": {"scenarios": {name: {"mean": 1.2} for name in ("T1", "T2", "T3")}}}}
    closed = {"aggregate": {"scenarios": {name: {"closed_loop": {"mean": 0.8}} for name in ("T1", "T2", "T3")}}}
    output = tmp_path / "mechanism.png"
    render_figure(architectures, noise, proar, closed, output)
    assert output.exists()
    assert output.stat().st_size > 1000
