"""Plot validation/test trade-offs for a frozen anchor-residual beta."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRICS = (
    ("Coord. RMSE", "coordinate_rmse_angstrom", False),
    ("Matching", "matching_mean_angstrom", False),
    ("Stability", "stability_mean_percent", True),
    ("Aligned RMSD", "aligned_rmsd_mean_angstrom", False),
    ("Rg error", "radius_of_gyration_mae_angstrom", False),
    ("RMSF error", "rmsf_mae_angstrom", False),
    ("Contact agree.", "contact_map_agreement_mean", True),
)


def _selected_and_reference(report):
    selected_beta = report["selected_beta"]
    reference = next(
        candidate for candidate in report["candidates"] if candidate["beta"] == 0
    )
    selected = next(
        candidate
        for candidate in report["candidates"]
        if candidate["beta"] == selected_beta
    )
    return reference["summary"], selected["summary"], selected_beta


def _improvements(report):
    reference, selected, selected_beta = _selected_and_reference(report)
    values = []
    for _, metric, higher_is_better in METRICS:
        baseline = float(reference[metric])
        candidate = float(selected[metric])
        direction = 1.0 if higher_is_better else -1.0
        values.append(direction * (candidate - baseline) / abs(baseline) * 100)
    return values, selected_beta


def render_figure(validation_path, test_path, output_path):
    validation = json.loads(Path(validation_path).read_text(encoding="utf-8"))
    test = json.loads(Path(test_path).read_text(encoding="utf-8"))
    validation_values, selected_beta = _improvements(validation)
    test_values, test_beta = _improvements(test)
    if selected_beta != test_beta:
        raise ValueError("validation and test must use the same frozen beta")

    labels = [label for label, _, _ in METRICS]
    positions = np.arange(len(labels))
    width = 0.36
    fig, axis = plt.subplots(figsize=(12, 6.2))
    validation_bars = axis.bar(
        positions - width / 2,
        validation_values,
        width,
        label="Validation (selection)",
        color="#2676b8",
    )
    test_bars = axis.bar(
        positions + width / 2,
        test_values,
        width,
        label="Test (frozen once)",
        color="#e39a37",
    )
    axis.axhline(0, color="#333333", linewidth=1)
    axis.set_xticks(positions, labels, rotation=18, ha="right")
    axis.set_ylabel("Improvement over NeuralMD beta=0 (%)")
    axis.set_title(
        f"Static-anchor residual trade-off (validation-selected beta={selected_beta:g})"
    )
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    for bars in (validation_bars, test_bars):
        for bar in bars:
            value = bar.get_height()
            axis.annotate(
                f"{value:+.1f}",
                (bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 3 if value >= 0 else -13),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
    axis.text(
        0.01,
        -0.22,
        "Positive means improvement. Validation selected beta; test did not rescan. "
        "Project proxy diagnostics, not official competition scores.",
        transform=axis.transAxes,
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("validation", type=Path)
    parser.add_argument("test", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(render_figure(args.validation, args.test, args.output))


if __name__ == "__main__":
    main()
