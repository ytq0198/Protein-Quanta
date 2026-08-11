"""Plot the frozen competition-scenario validation baseline."""

import argparse
import json
from pathlib import Path

import numpy as np


SCENARIOS = ("T1", "T2", "T3")
MODELS = ("neuralmd", "static")


def _metric_values(summary, metric):
    return {
        model: [float(summary[scenario][model][metric]) for scenario in SCENARIOS]
        for model in MODELS
    }


def main():
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8"))

    panels = (
        ("coordinate_rmse_angstrom", "Coordinate RMSE (A)"),
        ("matching_mean_angstrom", "Pair-distance RMSE (A)"),
        ("stability_mean_percent", "Distance stability (%)"),
        ("rmsf_mae_angstrom", "RMSF MAE (A)"),
        ("coordinate_rmse_slope_angstrom_per_frame", "RMSE growth (A/frame)"),
    )
    colors = {"neuralmd": "#1976A3", "static": "#C58A35"}
    labels = {"neuralmd": "NeuralMD", "static": "Static"}
    figure, axes = plt.subplots(2, 3, figsize=(13.5, 7.6), dpi=160)
    x = np.arange(len(SCENARIOS))
    width = 0.34
    for axis, (metric, title) in zip(axes.flat, panels):
        values = _metric_values(report["summary"], metric)
        for offset, model in zip((-width / 2, width / 2), MODELS):
            bars = axis.bar(
                x + offset,
                values[model],
                width,
                color=colors[model],
                label=labels[model],
            )
            axis.bar_label(bars, fmt="%.3g", padding=2, fontsize=8)
        axis.set_title(title, fontsize=11, weight="semibold")
        axis.set_xticks(x, SCENARIOS)
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    axes.flat[-1].axis("off")
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.943),
        ncol=2,
        frameon=False,
    )
    figure.suptitle(
        "Competition-aligned validation proxies (10 MISATO complexes)",
        y=0.992,
        fontsize=15,
        weight="bold",
    )
    figure.text(
        0.5,
        0.012,
        "Raw diagnostic metrics; not the organizer's normalized competition score.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    figure.tight_layout(rect=(0.02, 0.04, 0.98, 0.88))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
