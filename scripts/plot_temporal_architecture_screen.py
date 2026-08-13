"""Plot the invariant temporal architecture screen."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def render_figure(report, output):
    results = report["results"]
    names = [row["architecture"].upper() for row in results]
    scenarios = ("T1", "T2", "T3")
    values = np.array(
        [
            [row["best"]["scenarios"][scenario]["standardized_rmse"] for scenario in scenarios]
            for row in results
        ]
    )
    static = np.array(
        [results[0]["best"]["scenarios"][scenario]["static_standardized_rmse"] for scenario in scenarios]
    )
    weighted = np.array(
        [
            row["best"].get(
                "macro_scenario_rmse", row["best"].get("weighted_validation_rmse")
            )
            for row in results
        ]
    )

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    x = np.arange(len(names))
    width = 0.22
    colors = ("#375a7f", "#e28e2c", "#59a14f")
    for index, (scenario, color) in enumerate(zip(scenarios, colors)):
        axes[0].bar(x + (index - 1) * width, values[:, index], width, label=scenario, color=color)
        axes[0].axhline(static[index], color=color, linestyle="--", alpha=0.55)
    axes[0].set_xticks(x, names)
    axes[0].set_ylabel("Standardized rollout RMSE (lower is better)")
    axes[0].set_title("Scenario-wise validation screen")
    axes[0].legend(frameon=False, ncol=3)

    bar_colors = ["#4e79a7" if name != "TRANSFORMER" else "#f28e2b" for name in names]
    axes[1].bar(x, weighted, color=bar_colors)
    axes[1].set_xticks(x, names, rotation=20)
    axes[1].set_ylabel("0.5 T1 + 0.3 T2 + 0.2 T3 RMSE")
    axes[1].set_title("Competition-weighted architecture proxy")
    for index, value in enumerate(weighted):
        axes[1].text(index, value + 0.006, f"{value:.3f}", ha="center", fontsize=9)
    axes[1].set_ylim(0, max(weighted) * 1.16)
    figure.suptitle(
        "Invariant temporal-core screen — architecture evidence, not official competition scores",
        fontsize=12,
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render_figure(json.loads(args.report.read_text(encoding="utf-8")), args.output)


if __name__ == "__main__":
    main()
