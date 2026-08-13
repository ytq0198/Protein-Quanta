"""Run E18a correctness gates for the dense radial equivariant pilot."""

import argparse
import json
from pathlib import Path

import torch
from torchdiffeq import odeint

from NeuralMD.dataloaders.dataloader_MISATO import DataLoaderMISATO
from NeuralMD.datasets.MISATO import DatasetMISATOSemiFlexibleMultiTrajectory
from protein_quanta.dense_equivariant_dynamics import DenseEquivariantAcceleration
from protein_quanta.neuralmd_multiscale import neuralmd_ode_rollout
from protein_quanta.rollout_loss import multiscale_coordinate_smooth_l1
from scripts.audit_neuralmd_multiscale import (
    finite_parameter_gradient,
    proper_rotation,
    transformed_batch,
)


def rollout(model, batch, horizon, initial_position=None):
    return neuralmd_ode_rollout(
        model, odeint, batch, start=0, horizon=horizon, scaling=100,
        step_size=0.025, method="euler", initial_position=initial_position,
    )[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:2")
    args = parser.parse_args()
    torch.manual_seed(20260814)
    device = torch.device(args.device)
    dataset = DatasetMISATOSemiFlexibleMultiTrajectory(str(args.data_root), mode="train")
    batch = next(iter(DataLoaderMISATO(dataset, batch_size=1, num_workers=0, shuffle=False))).to(device)
    model = DenseEquivariantAcceleration(hidden_dim=32).to(device)
    model.eval()
    protein_before = batch.protein_pos.detach().clone()

    horizons = {}
    for horizon in (5, 10, 20, 40):
        prediction = rollout(model, batch, horizon)
        truth = batch.ligand_trajectory_pos[:, 1 : horizon + 1, :].transpose(0, 1)
        loss = multiscale_coordinate_smooth_l1(prediction[1:], truth)
        finite, norm = finite_parameter_gradient(model, loss)
        horizons[str(horizon)] = {
            "loss": float(loss.detach().cpu()),
            "parameter_gradient_norm": norm,
            "finite_forward": bool(torch.isfinite(prediction).all()),
            "finite_backward": finite,
        }

    initial = batch.ligand_trajectory_pos[:, 0, :].detach().clone().requires_grad_(True)
    terminal = rollout(model, batch, 40, initial_position=initial)[-1]
    initial_gradient = torch.autograd.grad(terminal.square().mean(), initial)[0]

    rotation = proper_rotation(device, batch.protein_pos.dtype)
    translation = torch.tensor([1.25, -2.5, 0.75], device=device)
    rigid_batch = transformed_batch(batch, rotation, translation)
    with torch.no_grad():
        reference = rollout(model, batch, 40)
        transformed = rollout(model, rigid_batch, 40)
    rigid_error = float(
        (transformed - (reference @ rotation.T + translation)).abs().max().cpu()
    )
    report = {
        "status": "pass",
        "scope": "one real MISATO-100 train complex; random-init correctness pilot",
        "architecture": "dense invariant-radial scalar messages times relative vectors",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "euler_step_size": 0.025,
        "horizons": horizons,
        "protein_bitwise_unchanged": bool(torch.equal(protein_before, batch.protein_pos)),
        "terminal_to_initial_gradient_norm": float(torch.linalg.vector_norm(initial_gradient).detach().cpu()),
        "terminal_to_initial_gradient_finite": bool(torch.isfinite(initial_gradient).all()),
        "rigid_equivariance_max_error_angstrom": rigid_error,
        "rigid_equivariance_threshold_angstrom": 1e-4,
    }
    gates = [
        all(item["finite_forward"] and item["finite_backward"]
            and item["parameter_gradient_norm"] > 0 for item in horizons.values()),
        report["protein_bitwise_unchanged"],
        report["terminal_to_initial_gradient_finite"],
        report["terminal_to_initial_gradient_norm"] > 0,
        rigid_error < 1e-4,
    ]
    report["gates"] = {"passed": sum(gates), "total": len(gates)}
    if not all(gates):
        report["status"] = "fail"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not all(gates):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
