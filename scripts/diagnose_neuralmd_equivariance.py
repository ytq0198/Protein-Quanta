"""Diagnose E18a failures without training or held-out data access."""

import argparse
import json
from pathlib import Path

import torch
from torchdiffeq import odeint

from NeuralMD.dataloaders.dataloader_MISATO import DataLoaderMISATO
from NeuralMD.datasets.MISATO import DatasetMISATOSemiFlexibleMultiTrajectory
from models.NeuralMD_Binding01_2nd_ODE import NeuralMD_Binding01
from protein_quanta.neuralmd_multiscale import neuralmd_ode_rollout
from protein_quanta.rollout_loss import multiscale_coordinate_smooth_l1
from scripts.audit_neuralmd_multiscale import (
    finite_parameter_gradient,
    model_arguments,
    proper_rotation,
    transformed_batch,
)


def rollout(model, batch, horizon, step_size):
    return neuralmd_ode_rollout(
        model, odeint, batch, start=0, horizon=horizon, scaling=100,
        step_size=step_size, method="euler",
    )[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    args = parser.parse_args()
    device = torch.device(args.device)
    dataset = DatasetMISATOSemiFlexibleMultiTrajectory(str(args.data_root), mode="train")
    batch = next(iter(DataLoaderMISATO(dataset, batch_size=1, num_workers=0, shuffle=False))).to(device)
    model = NeuralMD_Binding01(model_arguments()).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=False)["binding_model"])
    if args.dtype == "float64":
        model = model.double()
        for name in ("ligand_trajectory_pos", "ligand_mass", "protein_pos"):
            setattr(batch, name, getattr(batch, name).double())
    model.eval()

    gradients = {}
    truth = batch.ligand_trajectory_pos[:, 1:6, :].transpose(0, 1)
    for step_size in (0.05, 0.025, 0.01):
        prediction = rollout(model, batch, 5, step_size)
        loss = multiscale_coordinate_smooth_l1(prediction[1:], truth)
        finite, norm = finite_parameter_gradient(model, loss)
        gradients[str(step_size)] = {"finite": finite, "norm": norm}

    rotation = proper_rotation(device, batch.protein_pos.dtype)
    zero = torch.zeros(3, device=device)
    translation = torch.tensor([1.25, -2.5, 0.75], device=device)
    with torch.no_grad():
        reference = rollout(model, batch, 40, 0.05)
        rotated = rollout(model, transformed_batch(batch, rotation, zero), 40, 0.05)
        identity = torch.eye(3, device=device, dtype=batch.protein_pos.dtype)
        translated = rollout(model, transformed_batch(batch, identity, translation), 40, 0.05)
        rigid = rollout(model, transformed_batch(batch, rotation, translation), 40, 0.05)
    rotation_error = float((rotated - reference @ rotation.T).abs().max().cpu())
    translation_error = float((translated - (reference + translation)).abs().max().cpu())
    rigid_error = float((rigid - (reference @ rotation.T + translation)).abs().max().cpu())
    coordinate_scale = float(reference.abs().max().cpu())
    report = {
        "status": "diagnostic only; no training",
        "dtype": args.dtype,
        "horizon_5_parameter_gradients_by_euler_step": gradients,
        "rotation_max_error_angstrom": rotation_error,
        "translation_max_error_angstrom": translation_error,
        "rigid_max_error_angstrom": rigid_error,
        "reference_max_abs_coordinate_angstrom": coordinate_scale,
        "rigid_relative_to_coordinate_scale": rigid_error / max(coordinate_scale, 1e-12),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
