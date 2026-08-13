"""Locate whether NeuralMD rotation error originates in one derivative or rollout."""

import argparse
import json
from pathlib import Path

import torch
from torchdiffeq import odeint

from NeuralMD.dataloaders.dataloader_MISATO import DataLoaderMISATO
from NeuralMD.datasets.MISATO import DatasetMISATOSemiFlexibleMultiTrajectory
from models.NeuralMD_Binding01_2nd_ODE import NeuralMD_Binding01
from protein_quanta.neuralmd_multiscale import neuralmd_condition, neuralmd_ode_rollout
from scripts.audit_neuralmd_multiscale import model_arguments, proper_rotation, transformed_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:2")
    args = parser.parse_args()
    device = torch.device(args.device)
    dataset = DatasetMISATOSemiFlexibleMultiTrajectory(str(args.data_root), mode="train")
    batch = next(iter(DataLoaderMISATO(dataset, batch_size=1, num_workers=0, shuffle=False))).to(device)
    model = NeuralMD_Binding01(model_arguments()).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=False)["binding_model"])
    model.eval()
    rotation = proper_rotation(device, batch.protein_pos.dtype)
    rotated_batch = transformed_batch(batch, rotation, torch.zeros(3, device=device))

    position = batch.ligand_trajectory_pos[:, 0, :]
    velocity = batch.ligand_trajectory_pos[:, 1, :] - position
    rotated_position = rotated_batch.ligand_trajectory_pos[:, 0, :]
    rotated_velocity = rotated_batch.ligand_trajectory_pos[:, 1, :] - rotated_position
    with torch.no_grad():
        acceleration, position_derivative = model(
            torch.zeros((), device=device),
            (velocity, position),
            neuralmd_condition(batch),
        )
        rotated_acceleration, rotated_position_derivative = model(
            torch.zeros((), device=device),
            (rotated_velocity, rotated_position),
            neuralmd_condition(rotated_batch),
        )
    derivative_errors = {
        "acceleration": float((rotated_acceleration - acceleration @ rotation.T).abs().max().cpu()),
        "position_derivative": float((rotated_position_derivative - position_derivative @ rotation.T).abs().max().cpu()),
    }
    horizon_errors = {}
    with torch.no_grad():
        for horizon in (5, 10, 20, 40):
            reference = neuralmd_ode_rollout(
                model, odeint, batch, 0, horizon, 100, 0.05, "euler"
            )[1]
            transformed = neuralmd_ode_rollout(
                model, odeint, rotated_batch, 0, horizon, 100, 0.05, "euler"
            )[1]
            horizon_errors[str(horizon)] = float(
                (transformed - reference @ rotation.T).abs().max().cpu()
            )
    report = {
        "status": "diagnostic only; no training",
        "single_derivative_rotation_errors": derivative_errors,
        "rollout_rotation_max_error_angstrom": horizon_errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
