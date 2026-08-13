"""Run velocity-aware E(3) correctness gates on one real MISATO complex."""

import argparse
import json
from pathlib import Path

import torch
from torchdiffeq import odeint

from NeuralMD.dataloaders.dataloader_MISATO import DataLoaderMISATO
from NeuralMD.datasets.MISATO import DatasetMISATOSemiFlexibleMultiTrajectory
from protein_quanta.neuralmd_multiscale import neuralmd_ode_rollout
from protein_quanta.rollout_loss import multiscale_coordinate_smooth_l1
from protein_quanta.velocity_equivariant_dynamics import (
    VelocityEquivariantAcceleration,
)


def proper_rotation(device, dtype):
    generator = torch.Generator(device="cpu").manual_seed(20260814)
    matrix = torch.randn(3, 3, generator=generator, dtype=dtype).to(device)
    rotation, _ = torch.linalg.qr(matrix)
    if torch.linalg.det(rotation) < 0:
        rotation[:, -1] = -rotation[:, -1]
    return rotation


def transformed_batch(batch, matrix, translation):
    transformed = batch.clone()
    transformed.protein_pos = batch.protein_pos @ matrix.T + translation
    transformed.ligand_trajectory_pos = (
        batch.ligand_trajectory_pos @ matrix.T + translation
    )
    return transformed


def finite_parameter_gradient(model, loss):
    gradients = torch.autograd.grad(
        loss, tuple(model.parameters()), allow_unused=True
    )
    finite = True
    squared_norm = loss.new_zeros(())
    for gradient in gradients:
        if gradient is not None:
            finite = finite and bool(torch.isfinite(gradient).all())
            squared_norm = squared_norm + gradient.square().sum()
    return finite, float(torch.sqrt(squared_norm).detach().cpu())


def reflection(device, dtype):
    matrix = proper_rotation(device, dtype).clone()
    matrix[:, -1] = -matrix[:, -1]
    return matrix


def rollout(model, batch, horizon, initial_position=None):
    return neuralmd_ode_rollout(
        model,
        odeint,
        batch,
        start=0,
        horizon=horizon,
        scaling=1,
        step_size=1,
        method="euler",
        initial_velocity_scale=1,
        initial_position=initial_position,
    )[1]


def rigid_error(model, batch, matrix, translation):
    transformed = transformed_batch(batch, matrix, translation)
    with torch.no_grad():
        reference = rollout(model, batch, 40)
        transformed_path = rollout(model, transformed, 40)
    expected = reference @ matrix.T + translation
    return float((transformed_path - expected).abs().max().cpu())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:3")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))

    torch.manual_seed(20260814)
    device = torch.device(args.device)
    dataset = DatasetMISATOSemiFlexibleMultiTrajectory(
        str(args.data_root), mode="train"
    )
    batch = next(iter(DataLoaderMISATO(
        dataset, batch_size=1, num_workers=0, shuffle=False
    ))).to(device)
    model = VelocityEquivariantAcceleration(hidden_dim=32).to(device)
    model.eval()
    protein_before = batch.protein_pos.detach().clone()

    horizons = {}
    for horizon in config["protocol"]["horizons"]:
        prediction = rollout(model, batch, horizon)
        truth = batch.ligand_trajectory_pos[
            :, 1 : horizon + 1, :
        ].transpose(0, 1)
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
    condition = (
        batch.ligand_x,
        batch.batch_ligand,
        batch.ligand_mass,
        batch.protein_pos[batch.mask_n],
        batch.protein_pos[batch.mask_ca],
        batch.protein_pos[batch.mask_c],
        batch.protein_backbone_residue,
        batch.batch_residue,
    )
    position = batch.ligand_trajectory_pos[:, 0, :]
    velocity = batch.ligand_trajectory_pos[:, 1, :] - position
    with torch.no_grad():
        acceleration = model(0, (velocity, position), condition)[0]
        zero_velocity_acceleration = model(
            0, (torch.zeros_like(velocity), position), condition
        )[0]
    velocity_sensitivity = float(torch.linalg.vector_norm(
        acceleration - zero_velocity_acceleration
    ).cpu())

    translation = torch.tensor([1.25, -2.5, 0.75], device=device)
    rotation_error = rigid_error(
        model, batch, proper_rotation(device, batch.protein_pos.dtype), translation
    )
    reflection_error = rigid_error(
        model, batch, reflection(device, batch.protein_pos.dtype), translation
    )
    threshold = config["gates"]["proper_rigid_equivariance_max_error_angstrom"]
    checks = {
        "finite_forward_backward_all_horizons": all(
            row["finite_forward"] and row["finite_backward"]
            for row in horizons.values()
        ),
        "nonzero_parameter_gradient_all_horizons": all(
            row["parameter_gradient_norm"] > 0 for row in horizons.values()
        ),
        "protein_bitwise_unchanged": bool(
            torch.equal(protein_before, batch.protein_pos)
        ),
        "terminal_to_initial_gradient_finite_nonzero": bool(
            torch.isfinite(initial_gradient).all()
            and torch.linalg.vector_norm(initial_gradient) > 0
        ),
        "proper_rigid_equivariance": rotation_error < threshold,
        "reflection_equivariance": reflection_error < config["gates"][
            "reflection_equivariance_max_error_angstrom"
        ],
        "velocity_sensitive": velocity_sensitivity > 0,
    }
    passed = all(checks.values())
    report = {
        "status": "correctness_gate_pass" if passed else "correctness_gate_fail",
        "scope": config["scope"],
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "horizons": horizons,
        "terminal_to_initial_gradient_norm": float(
            torch.linalg.vector_norm(initial_gradient).detach().cpu()
        ),
        "proper_rigid_equivariance_max_error_angstrom": rotation_error,
        "reflection_equivariance_max_error_angstrom": reflection_error,
        "velocity_sensitivity_norm": velocity_sensitivity,
        "checks": checks,
        "decision": config["decision"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
