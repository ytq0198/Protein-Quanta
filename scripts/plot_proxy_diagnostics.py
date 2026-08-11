"""Render publication-ready figures from the proxy-diagnostic JSON report."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {"static": "#2C7FB8", "linear": "#D95F0E"}


def _load_report(path):
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    status = report.get("protocol", {}).get("status", "")
    if "not an official competition score" not in status:
        raise ValueError("input must be explicitly labeled as a non-official proxy")
    return report


def _mean_curve(samples, baseline, key):
    curves = [
        np.asarray(sample[key], dtype=float)
        for sample in samples
        if sample["baseline"] == baseline
    ]
    if not curves:
        raise ValueError(f"no {baseline} samples found")
    lengths = {curve.shape for curve in curves}
    if len(lengths) != 1:
        raise ValueError(f"{baseline} curves must have matching shapes")
    return np.mean(np.stack(curves), axis=0)


def _save_summary(report, path):
    metrics = (
        ("aligned_rmsd_mean_angstrom", "Aligned RMSD", "A"),
        ("radius_of_gyration_mae_angstrom", "Radius-of-gyration MAE", "A"),
        ("rmsf_mae_angstrom", "RMSF MAE", "A"),
        ("contact_map_agreement_mean", "Contact-map agreement", "%"),
    )
    baselines = ("static", "linear")
    figure, axes = plt.subplots(2, 2, figsize=(9.2, 6.6))
    for axis, (key, title, unit) in zip(axes.flat, metrics):
        values = [report["summary"][name][key] for name in baselines]
        if unit == "%":
            values = [100.0 * value for value in values]
        bars = axis.bar(
            [name.title() for name in baselines],
            values,
            color=[COLORS[name] for name in baselines],
            width=0.62,
        )
        axis.set_title(title, fontsize=11, weight="bold")
        axis.set_ylabel(unit)
        axis.grid(axis="y", alpha=0.25)
        axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "MISATO tiny: Static vs Linear proxy diagnostics",
        fontsize=14,
        weight="bold",
    )
    figure.text(
        0.5,
        0.01,
        "20 complexes; frames 0-1 observed, frames 2-99 predicted. "
        "Project-defined proxies, not official competition scores.",
        ha="center",
        fontsize=8.5,
        color="#444444",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.95))
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _save_rollout(report, path):
    samples = report["samples"]
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.1))
    curve_specs = (
        ("aligned_rmsd_by_frame_angstrom", "Aligned RMSD", "A"),
        (
            "radius_of_gyration_error_by_frame_angstrom",
            "Radius-of-gyration error",
            "A",
        ),
    )
    for axis, (key, title, unit) in zip(axes, curve_specs):
        for baseline in ("static", "linear"):
            curve = _mean_curve(samples, baseline, key)
            forecast_frames = np.arange(2, 2 + curve.size)
            axis.plot(
                forecast_frames,
                curve,
                label=baseline.title(),
                color=COLORS[baseline],
                linewidth=2.2,
            )
        axis.set_title(title, fontsize=11, weight="bold")
        axis.set_xlabel("Trajectory frame")
        axis.set_ylabel(unit)
        axis.grid(alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False)
    figure.suptitle(
        "Error accumulation across the 98-frame rollout",
        fontsize=14,
        weight="bold",
    )
    figure.text(
        0.5,
        -0.01,
        "Mean over 20 MISATO tiny complexes. Project-defined proxies, "
        "not official competition scores.",
        ha="center",
        fontsize=8.5,
        color="#444444",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.94))
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def render_figures(input_path, output_dir):
    report = _load_report(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = (
        output_dir / "misato_tiny_baseline_summary.png",
        output_dir / "misato_tiny_rollout_diagnostics.png",
    )
    _save_summary(report, outputs[0])
    _save_rollout(report, outputs[1])
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    for output in render_figures(args.input, args.output_dir):
        print(output)


if __name__ == "__main__":
    main()
