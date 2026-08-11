"""Render report-ready NeuralMD versus Static proxy-diagnostic figures."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LOWER_IS_BETTER = {
    "Coord. RMSE": "coordinate_rmse_angstrom",
    "Distance match": "matching_mean_angstrom",
    "Aligned RMSD": "aligned_rmsd_mean_angstrom",
    "Rg error": "radius_of_gyration_mae_angstrom",
    "RMSF error": "rmsf_mae_angstrom",
}
HIGHER_IS_BETTER = {
    "Stability": "stability_mean_percent",
    "Contact agreement": "contact_map_agreement_mean",
}


def _improvement_percent(neuralmd, static, metric, higher_is_better=False):
    denominator = abs(float(static[metric]))
    if denominator == 0:
        return 0.0
    direction = 1.0 if higher_is_better else -1.0
    return direction * (float(neuralmd[metric]) - float(static[metric])) / denominator * 100


def render_figures(input_path, output_dir):
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = json.loads(input_path.read_text(encoding="utf-8"))
    neuralmd = report["summary"]["neuralmd"]
    static = report["summary"]["static"]

    labels = list(LOWER_IS_BETTER) + list(HIGHER_IS_BETTER)
    improvements = [
        _improvement_percent(neuralmd, static, metric)
        for metric in LOWER_IS_BETTER.values()
    ] + [
        _improvement_percent(neuralmd, static, metric, higher_is_better=True)
        for metric in HIGHER_IS_BETTER.values()
    ]
    colors = ["#2676b8" if value >= 0 else "#d75b4e" for value in improvements]
    fig, axis = plt.subplots(figsize=(11, 5.8))
    positions = np.arange(len(labels))
    axis.barh(positions, improvements, color=colors, alpha=0.9)
    axis.axvline(0, color="#333333", linewidth=1)
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlabel("NeuralMD improvement over Static (%)")
    axis.set_title("MISATO-100 official test split: proxy diagnostic comparison")
    for y, value in zip(positions, improvements):
        offset = 0.35 if value >= 0 else -0.35
        axis.text(
            value + offset,
            y,
            f"{value:+.1f}%",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=9,
        )
    axis.text(
        0.01,
        -0.16,
        "Positive means NeuralMD is better. Project-defined proxies; not an official competition score.",
        transform=axis.transAxes,
        fontsize=9,
        color="#555555",
    )
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    summary_path = output_dir / "neuralmd_test_static_improvement.png"
    fig.savefig(summary_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    sample_ids = [sample["sample_id"] for sample in report["samples"]]
    neural_rmse = [
        sample["comparison"]["neuralmd"]["coordinate_rmse_angstrom"]
        for sample in report["samples"]
    ]
    static_rmse = [
        sample["comparison"]["static"]["coordinate_rmse_angstrom"]
        for sample in report["samples"]
    ]
    neural_stability = [
        sample["comparison"]["neuralmd"]["stability_mean_percent"]
        for sample in report["samples"]
    ]
    static_stability = [
        sample["comparison"]["static"]["stability_mean_percent"]
        for sample in report["samples"]
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    annotation_offsets = [
        (5, 6), (5, -12), (5, 6), (-24, 6), (5, -12),
        (5, 6), (-24, -12), (5, 6), (5, -12), (-24, 6),
    ]
    for axis, x_values, y_values, title, unit in (
        (
            axes[0], static_rmse, neural_rmse,
            "Coordinate RMSE (lower is better)", "Angstrom",
        ),
        (
            axes[1], static_stability, neural_stability,
            "Distance stability (higher is better)", "Percent",
        ),
    ):
        lower = min(x_values + y_values)
        upper = max(x_values + y_values)
        margin = max((upper - lower) * 0.08, 0.05)
        axis.plot(
            [lower - margin, upper + margin],
            [lower - margin, upper + margin],
            linestyle="--",
            color="#888888",
            linewidth=1,
        )
        axis.scatter(x_values, y_values, s=55, color="#2676b8")
        for sample_id, x_value, y_value, offset in zip(
            sample_ids, x_values, y_values, annotation_offsets
        ):
            axis.annotate(sample_id, (x_value, y_value), xytext=offset,
                          textcoords="offset points", fontsize=8)
        axis.set_xlabel(f"Static ({unit})")
        axis.set_ylabel(f"NeuralMD ({unit})")
        axis.set_title(title)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Per-complex behavior on 10 held-out MISATO-100 complexes")
    fig.text(
        0.5, 0.01,
        "Each point is one complex; dashed line denotes equal performance. Proxy diagnostics only.",
        ha="center", fontsize=9, color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    samples_path = output_dir / "neuralmd_test_per_complex.png"
    fig.savefig(samples_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return [summary_path, samples_path]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    for output in render_figures(args.input, args.output_dir):
        print(output)


if __name__ == "__main__":
    main()
