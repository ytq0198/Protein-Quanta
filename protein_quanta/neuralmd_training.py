"""Canonical command construction for the pinned NeuralMD MISATO setup."""

import math
from pathlib import Path


def build_training_command(
    output_dir,
    epochs,
    gpu_index=0,
    max_grad_norm=0.0,
    python_bin="/mnt/localDisk3/weizian/conda_envs/protein-quanta-neuralmd/bin/python",
    neuralmd_script="main_MISATO_multi_traj_NeuralMD.py",
    data_root="/mnt/localDisk3/weizian/datasets/misato",
):
    """Build the official seed-42 NeuralMD command with explicit booleans."""
    if int(epochs) != epochs or epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    if int(gpu_index) != gpu_index or gpu_index < 0:
        raise ValueError("gpu_index must be a non-negative integer")
    if max_grad_norm < 0 or not math.isfinite(float(max_grad_norm)):
        raise ValueError("max_grad_norm must be finite and non-negative")

    return [
        str(python_bin),
        str(neuralmd_script),
        "--device",
        str(int(gpu_index)),
        "--seed",
        "42",
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
        "--output_model_dir",
        str(Path(output_dir)),
    ]
