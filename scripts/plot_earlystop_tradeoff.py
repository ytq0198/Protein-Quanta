"""Visualize the frozen epoch-5 checkpoint against the published checkpoint."""

import argparse
import json
from pathlib import Path

import numpy as np


def _relative_changes(
    baseline,
    candidate,
    metric,
    point_change=False,
    baseline_model="neuralmd",
    candidate_model="neuralmd",
):
    changes = []
    for scenario in baseline:
        reference = float(baseline[scenario][baseline_model][metric])
        value = float(candidate[scenario][candidate_model][metric])
        changes.append(value - reference if point_change else 100.0 * (value / reference - 1.0))
    return changes


def _panel_limits(values):
    values = np.asarray(values, dtype=float)
    low = min(float(values.min()), 0.0)
    high = max(float(values.max()), 0.0)
    span = high - low
    padding = max(0.08 * span, 0.08)
    return low - padding, high + padding


def main():
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("--published-val", type=Path, required=True)
    parser.add_argument("--earlystop-val", type=Path, required=True)
    parser.add_argument("--published-test", type=Path, required=True)
    parser.add_argument("--earlystop-test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-model", default="neuralmd")
    parser.add_argument("--baseline-model", default="neuralmd")
    parser.add_argument(
        "--title",
        default="Scenario-aware early stopping: epoch 5 vs published checkpoint",
    )
    args = parser.parse_args()

    def summary(path):
        return json.loads(path.read_text(encoding="utf-8"))["summary"]

    baseline_val, earlystop_val = summary(args.published_val), summary(args.earlystop_val)
    baseline_test, earlystop_test = summary(args.published_test), summary(args.earlystop_test)
    panels = (
        ("coordinate_rmse_angstrom", "Coordinate RMSE change (%)", False),
        ("matching_mean_angstrom", "Pair-distance RMSE change (%)", False),
        ("stability_mean_percent", "Distance stability change (points)", True),
        ("rmsf_mae_angstrom", "RMSF MAE change (%)", False),
    )
    scenarios = list(baseline_val)
    x = np.arange(len(scenarios))
    width = 0.34
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.5), dpi=160)
    for axis, (metric, title, point_change) in zip(axes.flat, panels):
        validation = _relative_changes(
            baseline_val,
            earlystop_val,
            metric,
            point_change,
            baseline_model=args.baseline_model,
            candidate_model=args.candidate_model,
        )
        test = _relative_changes(
            baseline_test,
            earlystop_test,
            metric,
            point_change,
            baseline_model=args.baseline_model,
            candidate_model=args.candidate_model,
        )
        left = axis.bar(x - width / 2, validation, width, label="Validation", color="#1976A3")
        right = axis.bar(x + width / 2, test, width, label="Frozen test", color="#C58A35")
        axis.bar_label(left, fmt="%.2f", padding=2, fontsize=8)
        axis.bar_label(right, fmt="%.2f", padding=2, fontsize=8)
        axis.axhline(0, color="#333333", linewidth=0.8)
        axis.set_ylim(*_panel_limits(validation + test))
        axis.set_xticks(x, scenarios)
        axis.set_title(title, fontsize=11, weight="semibold")
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=2,
        frameon=False,
    )
    figure.suptitle(
        args.title,
        fontsize=14,
        weight="bold",
        y=0.992,
    )
    figure.text(
        0.5,
        0.01,
        "Validation selected the checkpoint; test was evaluated once after freezing. Proxy metrics, not official score.",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    figure.tight_layout(rect=(0.02, 0.04, 0.98, 0.87))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
