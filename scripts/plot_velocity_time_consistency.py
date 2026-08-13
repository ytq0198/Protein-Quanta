"""Plot the development-only velocity/time unit audit."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8"))
    summary = report["summary"]
    scenarios = ("T1", "T2", "T3")
    labels = ("T1 10→10", "T2 80→20", "T3 20→80")
    protocols = (
        ("upstream_convention", "Upstream: time /100, velocity ×1", "#3567A8", ""),
        ("unit_consistent", "Unit-consistent: time /100, velocity ×100", "#D47A2C", "//"),
    )
    x = np.arange(len(scenarios))
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.6))

    for offset, (key, label, color, hatch) in zip((-width / 2, width / 2), protocols):
        amplitude = [summary[key][name]["step_amplitude_ratio"] for name in scenarios]
        bars = axes[0].bar(
            x + offset, amplitude, width, label=label, color=color,
            edgecolor="#263238", linewidth=0.7, hatch=hatch,
        )
        axes[0].bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
        rmse = [summary[key][name]["coordinate_rmse_angstrom"] for name in scenarios]
        bars = axes[1].bar(
            x + offset, rmse, width, label=label, color=color,
            edgecolor="#263238", linewidth=0.7, hatch=hatch,
        )
        axes[1].bar_label(bars, fmt="%.2f", padding=3, fontsize=9)

    axes[0].axhline(1.0, color="#263238", linestyle="--", linewidth=1.2, label="Ideal amplitude = 1")
    axes[0].set_title("Step-amplitude ratio")
    axes[0].set_ylabel("Predicted / true mean step amplitude")
    axes[0].set_ylim(0, 1.18)
    axes[1].set_title("Constant-velocity forecast error")
    axes[1].set_ylabel("Coordinate RMSE (Å)")
    axes[1].set_ylim(bottom=0)
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.grid(axis="y", color="#D8DEE5", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, legend_labels, loc="upper center", ncol=3,
        frameon=False, bbox_to_anchor=(0.5, 0.925), fontsize=9,
    )
    fig.suptitle(
        "Velocity–time unit audit on 64 development complexes",
        fontsize=15, y=0.985,
    )
    fig.text(
        0.5, 0.018,
        "Zero-acceleration analytic control; T1/T2/T3 follow the updated guide. "
        "Amplitude recovery is not a performance gain: long-horizon constant velocity drifts.",
        ha="center", fontsize=9, color="#455A64",
    )
    fig.subplots_adjust(top=0.78, bottom=0.13, left=0.07, right=0.985, wspace=0.10)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
