"""Visualize grouped OOF uncertainty-gate evidence."""

import argparse
import json
from pathlib import Path

import numpy as np


def _confusion_counts(records):
    counts = {
        "true_negative": 0,
        "false_positive": 0,
        "false_negative": 0,
        "true_positive": 0,
    }
    for record in records:
        truth = int(record["label"])
        prediction = int(record["selected_beta"] == 8.0)
        key = {
            (0, 0): "true_negative",
            (0, 1): "false_positive",
            (1, 0): "false_negative",
            (1, 1): "true_positive",
        }[(truth, prediction)]
        counts[key] += 1
    return counts


def main():
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    changes = report["relative_to_global_beta1"]
    confusion = _confusion_counts(report["records"])

    scenarios = ("T1", "T2", "T3")
    metrics = (
        ("coordinate_rmse_percent", "Coordinate RMSE (%)"),
        ("matching_percent", "Matching (%)"),
        ("stability_points", "Stability (points)"),
        ("rmsf_percent", "RMSF MAE (%)"),
    )
    colors = ("#287CA3", "#C78D35", "#73874D", "#B65C7A")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=170)
    x = np.arange(len(scenarios))
    width = 0.18
    for index, ((metric, label), color) in enumerate(zip(metrics, colors)):
        values = [changes[scenario][metric] for scenario in scenarios]
        bars = axes[0].bar(
            x + (index - 1.5) * width,
            values,
            width,
            label=label,
            color=color,
            edgecolor="#333333",
            linewidth=0.4,
        )
        axes[0].bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
    axes[0].axhline(0, color="#333333", linewidth=0.8)
    axes[0].set_xticks(x, scenarios)
    axes[0].set_title("OOF metric change vs global beta=1", weight="semibold")
    axes[0].set_ylabel("Relative change; errors below zero are better")
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")

    matrix = np.array(
        [
            [confusion["true_negative"], confusion["false_positive"]],
            [confusion["false_negative"], confusion["true_positive"]],
        ]
    )
    image = axes[1].imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, matrix.max()))
    for row in range(2):
        for column in range(2):
            axes[1].text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                fontsize=16,
                color="white" if matrix[row, column] > matrix.max() / 2 else "#222222",
            )
    axes[1].set_xticks([0, 1], ["Select beta=1", "Select beta=8"])
    axes[1].set_yticks([0, 1], ["Unsafe/not useful", "Safe + useful"])
    axes[1].set_xlabel("OOF gate decision")
    axes[1].set_ylabel("Validation label")
    axes[1].set_title(
        f"OOF classification (balanced accuracy {report['oof_classification']['balanced_accuracy']:.2f})",
        weight="semibold",
    )
    figure.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04, label="Records")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "Low-capacity uncertainty gate: grouped leave-one-complex-out validation",
        fontsize=14,
        weight="bold",
    )
    figure.text(
        0.5,
        0.01,
        "30 validation records (10 complexes × 3 scenarios); internal proxy diagnostics, not official score. Test split not accessed.",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    figure.tight_layout(rect=(0.01, 0.05, 0.99, 0.92))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()

