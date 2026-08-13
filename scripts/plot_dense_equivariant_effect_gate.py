"""Render paired-seed and scenario summaries from a completed effect gate."""

import argparse
import json
from pathlib import Path


def load_plot_rows(report):
    gate = report["gate"]
    paired = gate["paired_seed_macro"]
    scenarios = []
    for name in ("T1", "T2", "T3"):
        control = gate["mean_summary"]["control"][name]
        candidate = gate["mean_summary"]["candidate"][name]
        scenarios.append({
            "scenario": name,
            "control_coordinate_rmse": control["coordinate_rmse_angstrom"],
            "candidate_coordinate_rmse": candidate["coordinate_rmse_angstrom"],
            "control_rmse_slope": control["rmse_slope_angstrom_per_frame"],
            "candidate_rmse_slope": candidate["rmse_slope_angstrom_per_frame"],
        })
    return paired, scenarios


def render(report, output):
    import matplotlib.pyplot as plt
    import numpy as np

    paired, scenarios = load_plot_rows(report)
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))

    seed_labels = [f'seed {row["seed"]}' for row in paired]
    relative_ppm = [
        1e6 * (row["candidate"] - row["control"]) / row["control"]
        for row in paired
    ]
    axes[0].bar(
        seed_labels, relative_ppm, color="#D77A36", edgecolor="#6B3A17"
    )
    axes[0].axhline(1.0, color="#4A5560", linestyle="--", linewidth=1.3)
    axes[0].text(
        len(seed_labels) - 0.5, 1.02, "1 ppm negligible-effect boundary",
        ha="right", va="bottom", color="#4A5560", fontsize=8,
    )
    axes[0].set_ylim(0, 1.15)
    axes[0].set_ylabel("Candidate worsening (ppm of baseline RMSE)")
    axes[0].set_title("Paired macro RMSE change")
    axes[0].grid(axis="y", alpha=0.25)
    for index, value in enumerate(relative_ppm):
        axes[0].text(index, value + 0.025, f"{value:.4f}", ha="center", fontsize=8)

    indices = np.arange(len(scenarios))
    width = 0.36
    axes[1].bar(
        indices - width / 2,
        [row["control_coordinate_rmse"] for row in scenarios],
        width,
        label="Local ODE",
        color="#7A8A99",
    )
    axes[1].bar(
        indices + width / 2,
        [row["candidate_coordinate_rmse"] for row in scenarios],
        width,
        label="+ multiscale",
        color="#D77A36",
    )
    axes[1].set_xticks(indices, [row["scenario"] for row in scenarios])
    axes[1].set_ylabel("Coordinate RMSE (Å)")
    axes[1].set_title("Updated-guide scenarios")
    axes[1].set_ylim(bottom=0)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False)

    status = report["status"].replace("_", " ")
    figure.suptitle(f"Dense equivariant multiscale effect gate — {status}")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    render(report, args.output)


if __name__ == "__main__":
    main()
