"""Visualize the fixed-epoch multi-seed temporal comparison."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DISPLAY_NAMES = {
    "mlp": "MLP",
    "rnn": "RNN",
    "lstm": "LSTM",
    "gru": "GRU",
    "transformer": "Transformer",
    "transformer_rope": "Transformer + RoPE",
}


def render_figure(report, output):
    aggregate = report["aggregate"]
    architectures = list(aggregate)
    names = [DISPLAY_NAMES[name] for name in architectures]
    means = np.array(
        [
            aggregate[name].get(
                "macro_scenario_rmse", aggregate[name].get("weighted_validation_rmse")
            )["mean"]
            for name in architectures
        ]
    )
    stds = np.array(
        [
            aggregate[name].get(
                "macro_scenario_rmse", aggregate[name].get("weighted_validation_rmse")
            )["sample_std"]
            for name in architectures
        ]
    )
    seed_values = [
        aggregate[name].get(
            "macro_scenario_rmse", aggregate[name].get("weighted_validation_rmse")
        )["values"]
        for name in architectures
    ]

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    colors = ["#8f9aa6", "#8f9aa6", "#8f9aa6", "#8f9aa6", "#276fbf", "#e07a1f"]
    x = np.arange(len(names))
    for index, (values, color) in enumerate(zip(seed_values, colors)):
        axes[0].errorbar(
            index,
            means[index],
            yerr=stds[index],
            fmt="none",
            ecolor=color,
            elinewidth=2,
            capsize=4,
            zorder=1,
        )
        jitter = np.linspace(-0.08, 0.08, len(values))
        axes[0].scatter(index + jitter, values, s=35, color=color, edgecolor="white", zorder=3)
        axes[0].scatter(index, means[index], marker="D", s=55, color=color, edgecolor="black", linewidth=0.5, zorder=4)
    axes[0].axhline(means[0], color="#404040", linestyle="--", linewidth=1, alpha=0.7)
    axes[0].set_xticks(x, names, rotation=22, ha="right")
    axes[0].set_ylabel("Equal-macro T1/T2/T3 RMSE (lower is better)")
    axes[0].set_title("Stability across three fixed seeds")
    axes[0].text(0.02, 0.02, "Dots: seeds 0/42/123 · diamonds: mean · bars: sample SD", transform=axes[0].transAxes, fontsize=8)

    scenarios = ("T1", "T2", "T3")
    selected = ("transformer", "transformer_rope")
    scenario_x = np.arange(len(scenarios))
    marker = {"transformer": "o", "transformer_rope": "s"}
    selected_colors = {"transformer": "#276fbf", "transformer_rope": "#e07a1f"}
    mlp = np.array(
        [aggregate["mlp"]["scenarios"][scenario]["standardized_rmse"]["mean"] for scenario in scenarios]
    )
    for architecture in selected:
        candidate = np.array(
            [aggregate[architecture]["scenarios"][scenario]["standardized_rmse"]["mean"] for scenario in scenarios]
        )
        relative = 100.0 * (candidate / mlp - 1.0)
        axes[1].plot(
            scenario_x,
            relative,
            marker=marker[architecture],
            linewidth=2,
            markersize=7,
            color=selected_colors[architecture],
            label=DISPLAY_NAMES[architecture],
        )
        for index, value in enumerate(relative):
            axes[1].annotate(f"{value:+.1f}%", (index, value), xytext=(0, 7 if value >= 0 else -13), textcoords="offset points", ha="center", fontsize=8)
    axes[1].axhline(0, color="#404040", linestyle="--", linewidth=1)
    axes[1].set_ylim(-30, 29)
    axes[1].set_xticks(scenario_x, scenarios)
    axes[1].set_ylabel("RMSE change versus MLP (%)")
    axes[1].set_title("Scenario-wise change relative to MLP")
    axes[1].legend(frameon=False)

    figure.suptitle(
        "Multi-seed temporal audit — 12D invariant proxy, not official competition scores",
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
