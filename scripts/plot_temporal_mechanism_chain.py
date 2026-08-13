"""Plot the corrected-guide temporal mechanism evidence chain."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _scenario_means_from_results(report, architecture):
    rows = [row for row in report["results"] if row.get("architecture") == architecture]
    return np.array(
        [
            np.mean([row["best"]["scenarios"][name]["standardized_rmse"] for row in rows])
            for name in ("T1", "T2", "T3")
        ]
    )


def render_figure(architecture_report, noise_report, proar_report, closed_loop_report, output):
    reference = _scenario_means_from_results(architecture_report, "transformer")
    methods = {
        "Transformer": reference,
        "RoPE": _scenario_means_from_results(architecture_report, "transformer_rope"),
        "iid noise": np.array(
            [noise_report["aggregate"]["scenarios"][name]["noise"]["mean"] for name in ("T1", "T2", "T3")]
        ),
        "alternating": np.array(
            [proar_report["aggregate"]["alternating"]["scenarios"][name]["mean"] for name in ("T1", "T2", "T3")]
        ),
        "closed-loop": np.array(
            [closed_loop_report["aggregate"]["scenarios"][name]["closed_loop"]["mean"] for name in ("T1", "T2", "T3")]
        ),
    }
    colors = {
        "Transformer": "#4b5563",
        "RoPE": "#d97706",
        "iid noise": "#dc2626",
        "alternating": "#7c3aed",
        "closed-loop": "#059669",
    }
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    x = np.arange(3)
    width = 0.15
    for index, (name, values) in enumerate(methods.items()):
        axes[0].bar(x + (index - 2) * width, values, width=width, label=name, color=colors[name])
    axes[0].set_xticks(x, ("T1 10→10", "T2 80→20", "T3 20→80"))
    axes[0].set_ylabel("Standardized rollout RMSE (lower is better)")
    axes[0].set_title("Corrected-guide scenario results")
    axes[0].legend(frameon=False, ncol=2, fontsize=8)

    comparison_names = ("RoPE", "iid noise", "alternating", "closed-loop")
    changes = np.array([100.0 * (methods[name] / reference - 1.0) for name in comparison_names])
    for scenario_index, scenario in enumerate(("T1", "T2", "T3")):
        axes[1].plot(
            np.arange(len(comparison_names)),
            changes[:, scenario_index],
            marker=("o", "s", "^")[scenario_index],
            linewidth=2,
            label=scenario,
        )
    axes[1].axhline(0, color="#4b5563", linestyle="--", linewidth=1)
    axes[1].set_xticks(np.arange(len(comparison_names)), comparison_names, rotation=18, ha="right")
    axes[1].set_ylabel("RMSE change versus Transformer (%)")
    axes[1].set_title("Only direct closed-loop training improves all mean scenarios")
    axes[1].legend(frameon=False)
    figure.suptitle("Temporal mechanism audit — MISATO-100 validation proxy, not official scores")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture-report", type=Path, required=True)
    parser.add_argument("--noise-report", type=Path, required=True)
    parser.add_argument("--proar-report", type=Path, required=True)
    parser.add_argument("--closed-loop-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    render_figure(
        load(args.architecture_report),
        load(args.noise_report),
        load(args.proar_report),
        load(args.closed_loop_report),
        args.output,
    )


if __name__ == "__main__":
    main()
