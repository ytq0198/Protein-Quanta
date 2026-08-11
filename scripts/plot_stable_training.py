"""Plot loss, gradient intervention and validation stability for E3."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from protein_quanta.training_log import parse_neuralmd_training_log


def render_figure(log_text, output_path):
    rows = parse_neuralmd_training_log(log_text)
    if not rows:
        raise ValueError("training log contains no epochs")

    epochs = [row["epoch"] for row in rows]
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)

    axes[0, 0].plot(
        epochs, [row.get("loss_pos", float("nan")) for row in rows], color="#315f72"
    )
    axes[0, 0].set(title="Training position loss", xlabel="Epoch", ylabel="MSE")

    gradient_rows = [row for row in rows if "grad_norm_max" in row]
    axes[0, 1].semilogy(
        [row["epoch"] for row in gradient_rows],
        [max(row["grad_norm_max"], 1e-12) for row in gradient_rows],
        color="#b05a3c",
    )
    axes[0, 1].axhline(1.0, color="#3f7f62", linestyle="--", label="clip=1")
    axes[0, 1].set(title="Pre-clip maximum gradient norm", xlabel="Epoch", ylabel="L2 norm")
    axes[0, 1].legend()

    axes[1, 0].bar(
        [row["epoch"] for row in gradient_rows],
        [row["clipped_batches"] for row in gradient_rows],
        color="#d3a342",
    )
    axes[1, 0].set(title="Clipped batches", xlabel="Epoch", ylabel="Count")

    validation_rows = [row for row in rows if "val_stability" in row]
    axes[1, 1].plot(
        [row["epoch"] for row in validation_rows],
        [row["val_stability"] for row in validation_rows],
        marker="o",
        color="#6b4c8a",
    )
    axes[1, 1].set(
        title="Validation stability (upstream metric)",
        xlabel="Epoch",
        ylabel="Stability (%)",
    )

    for axis in axes.flat:
        axis.grid(alpha=0.2)
    figure.suptitle("NeuralMD seed 42 — global gradient clipping at 1.0")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render_figure(args.log.read_text(encoding="utf-8"), args.output)


if __name__ == "__main__":
    main()
