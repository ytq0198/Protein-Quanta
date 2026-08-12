"""Canonical command construction for the pinned NeuralMD MISATO setup."""

import math
from pathlib import Path


def build_training_command(
    output_dir,
    epochs,
    gpu_index=0,
    seed=42,
    max_grad_norm=0.0,
    pair_loss_coefficient=0.0,
    pair_loss_beta=0.5,
    displacement_loss_coefficient=0.0,
    displacement_loss_beta=0.5,
    save_every_epoch=0,
    calibration_batches=0,
    calibration_output=None,
    python_bin="/mnt/localDisk3/weizian/conda_envs/protein-quanta-neuralmd/bin/python",
    neuralmd_script="main_MISATO_multi_traj_NeuralMD.py",
    data_root="/mnt/localDisk3/weizian/datasets/misato",
):
    """Build the official seed-42 NeuralMD command with explicit booleans."""
    if int(epochs) != epochs or epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    if int(gpu_index) != gpu_index or gpu_index < 0:
        raise ValueError("gpu_index must be a non-negative integer")
    if int(seed) != seed or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if max_grad_norm < 0 or not math.isfinite(float(max_grad_norm)):
        raise ValueError("max_grad_norm must be finite and non-negative")
    if pair_loss_coefficient < 0 or not math.isfinite(float(pair_loss_coefficient)):
        raise ValueError("pair_loss_coefficient must be finite and non-negative")
    if pair_loss_beta <= 0 or not math.isfinite(float(pair_loss_beta)):
        raise ValueError("pair_loss_beta must be finite and positive")
    if displacement_loss_coefficient < 0 or not math.isfinite(
        float(displacement_loss_coefficient)
    ):
        raise ValueError("displacement_loss_coefficient must be finite and non-negative")
    if displacement_loss_beta <= 0 or not math.isfinite(float(displacement_loss_beta)):
        raise ValueError("displacement_loss_beta must be finite and positive")
    if int(save_every_epoch) != save_every_epoch or save_every_epoch < 0:
        raise ValueError("save_every_epoch must be a non-negative integer")
    if int(calibration_batches) != calibration_batches or calibration_batches < 0:
        raise ValueError("calibration_batches must be a non-negative integer")
    if calibration_batches > 0 and calibration_output is None:
        raise ValueError("calibration_output is required in calibration mode")

    command = [
        str(python_bin),
        str(neuralmd_script),
        "--device",
        str(int(gpu_index)),
        "--seed",
        str(int(seed)),
        "--input_data_dir",
        str(data_root),
        "--dataset",
        "MISATO_100",
        "--batch_size",
        "8",
        "--num_workers",
        "0",
        "--epochs",
        str(int(epochs)),
        "--print_every_epoch",
        "5",
        "--lr",
        "1e-4",
        "--no_NeuralMD_Binding_start_with_first_frame",
        "--NeuralMD_Binding_frame_num",
        "20",
        "--NeuralMD_step_size",
        "5",
        "--NeuralMD_scaling",
        "100",
        "--NeuralMD_velocity_refined_value_coefficient",
        "0",
        "--NeuralMD_binding_model",
        "NeuralMD_Binding01",
        "--ODE_method",
        "euler",
        "--FrameNet_num_radial",
        "100",
        "--no_MLP_velocity",
        "--max_grad_norm",
        str(float(max_grad_norm)),
        "--pair_loss_coefficient",
        str(float(pair_loss_coefficient)),
        "--pair_loss_beta",
        str(float(pair_loss_beta)),
        "--displacement_loss_coefficient",
        str(float(displacement_loss_coefficient)),
        "--displacement_loss_beta",
        str(float(displacement_loss_beta)),
        "--save_every_epoch",
        str(int(save_every_epoch)),
        "--calibration_batches",
        str(int(calibration_batches)),
        "--no_eval_test_during_training",
        "--output_model_dir",
        Path(output_dir).as_posix(),
    ]
    if calibration_output is not None:
        command.extend(
            ["--calibration_output", Path(calibration_output).as_posix()]
        )
    return command
