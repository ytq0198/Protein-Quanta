"""Run the canonical MISATO-100 NeuralMD training configuration."""

import argparse
import os
from pathlib import Path
import shlex
import subprocess

from protein_quanta.neuralmd_training import build_training_command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("epochs", type=int)
    parser.add_argument("gpu_index", type=int, nargs="?", default=0)
    parser.add_argument("--max-grad-norm", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pair-loss-coefficient", type=float, default=0.0)
    parser.add_argument("--pair-loss-beta", type=float, default=0.5)
    parser.add_argument("--displacement-loss-coefficient", type=float, default=0.0)
    parser.add_argument("--displacement-loss-beta", type=float, default=0.5)
    parser.add_argument("--multiscale-rollout-coefficient", type=float, default=0.0)
    parser.add_argument("--multiscale-rollout-beta", type=float, default=0.5)
    parser.add_argument(
        "--multiscale-rollout-horizons", type=int, nargs="+", default=(5, 10, 20, 40)
    )
    parser.add_argument("--save-every-epoch", type=int, default=0)
    parser.add_argument("--calibration-batches", type=int, default=0)
    parser.add_argument("--calibration-output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    project_root = Path(
        os.environ.get("PROTEIN_QUANTA_ROOT", "/mnt/localDisk3/weizian/Protein-Quanta")
    )
    neuralmd_root = Path(
        os.environ.get("NEURALMD_ROOT", "/mnt/localDisk3/weizian/external/NeuralMD")
    )
    command = build_training_command(
        output_dir=args.output_dir,
        epochs=args.epochs,
        gpu_index=args.gpu_index,
        seed=args.seed,
        max_grad_norm=args.max_grad_norm,
        pair_loss_coefficient=args.pair_loss_coefficient,
        pair_loss_beta=args.pair_loss_beta,
        displacement_loss_coefficient=args.displacement_loss_coefficient,
        displacement_loss_beta=args.displacement_loss_beta,
        multiscale_rollout_coefficient=args.multiscale_rollout_coefficient,
        multiscale_rollout_beta=args.multiscale_rollout_beta,
        multiscale_rollout_horizons=args.multiscale_rollout_horizons,
        save_every_epoch=args.save_every_epoch,
        calibration_batches=args.calibration_batches,
        calibration_output=args.calibration_output,
        python_bin=os.environ.get(
            "NEURALMD_PYTHON",
            "/mnt/localDisk3/weizian/conda_envs/protein-quanta-neuralmd/bin/python",
        ),
        data_root=os.environ.get(
            "MISATO_DATA_ROOT", "/mnt/localDisk3/weizian/datasets/misato"
        ),
    )
    if args.dry_run:
        print(shlex.join(command))
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    python_paths = [str(project_root), str(neuralmd_root)]
    if environment.get("PYTHONPATH"):
        python_paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    subprocess.run(
        command,
        cwd=neuralmd_root / "examples",
        env=environment,
        check=True,
    )


if __name__ == "__main__":
    main()
