"""Run E18a correctness gates on one real MISATO complex and NeuralMD model."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torchdiffeq import odeint

from NeuralMD.dataloaders.dataloader_MISATO import DataLoaderMISATO
from NeuralMD.datasets.MISATO import DatasetMISATOSemiFlexibleMultiTrajectory
from models.NeuralMD_Binding01_2nd_ODE import NeuralMD_Binding01
from protein_quanta.neuralmd_multiscale import neuralmd_ode_rollout
from protein_quanta.rollout_loss import multiscale_coordinate_smooth_l1


def model_arguments():
    return SimpleNamespace(
        model_3d_ligand="FrameNet01", model_3d_protein="FrameNetProtein03",
        emb_dim=128, NeuralMD_velocity_refined_value_coefficient=0,
        use_MLP_velocity=False, FrameNet_cutoff=5, FrameNet_num_layers=4,
        FrameNet_num_radial=100, FrameNet_complex_layer=1,
        FrameNet_rbf_type="RBF_repredding_01", FrameNet_gamma=None,
        FrameNet_readout="mean",
    )


def proper_rotation(device, dtype):
    generator = torch.Generator(device="cpu").manual_seed(20260814)
    matrix = torch.randn(3, 3, generator=generator, dtype=dtype).to(device)
    rotation, _ = torch.linalg.qr(matrix)
    if torch.linalg.det(rotation) < 0:
        rotation[:, -1] = -rotation[:, -1]
    return rotation


def transformed_batch(batch, rotation, translation):
    transformed = batch.clone()
    transformed.protein_pos = batch.protein_pos @ rotation.T + translation
    transformed.ligand_trajectory_pos = (
        batch.ligand_trajectory_pos @ rotation.T + translation
    )
    return transformed


def finite_parameter_gradient(model, loss):
    gradients = torch.autograd.grad(loss, tuple(model.parameters()), allow_unused=True)
    finite = True
    squared_norm = loss.new_zeros(())
    for gradient in gradients:
        if gradient is not None:
            finite = finite and bool(torch.isfinite(gradient).all())
            squared_norm = squared_norm + gradient.square().sum()
    return finite, float(torch.sqrt(squared_norm).detach().cpu())


def rollout(model, batch, horizon, initial_position=None):
    return neuralmd_ode_rollout(
        model, odeint, batch, start=0, horizon=horizon, scaling=100,
        step_size=0.05, method="euler", initial_position=initial_position,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:2")
    args = parser.parse_args()

    device = torch.device(args.device)
    dataset = DatasetMISATOSemiFlexibleMultiTrajectory(
        str(args.data_root), mode="train"
    )
    batch = next(iter(DataLoaderMISATO(
        dataset, batch_size=1, num_workers=0, shuffle=False
    ))).to(device)
    model = NeuralMD_Binding01(model_arguments()).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["binding_model"])
    model.eval()

    protein_before = batch.protein_pos.detach().clone()
    horizon_results = {}
    for horizon in (5, 10, 20, 40):
        _, position = rollout(model, batch, horizon)
        truth = batch.ligand_trajectory_pos[:, 1 : horizon + 1, :].transpose(0, 1)
        loss = multiscale_coordinate_smooth_l1(position[1:], truth, beta=0.5)
        finite_gradient, gradient_norm = finite_parameter_gradient(model, loss)
        horizon_results[str(horizon)] = {
            "loss": float(loss.detach().cpu()),
            "parameter_gradient_norm": gradient_norm,
            "finite_forward": bool(torch.isfinite(position).all()),
            "finite_backward": finite_gradient,
        }

    initial = batch.ligand_trajectory_pos[:, 0, :].detach().clone().requires_grad_(True)
    _, terminal_path = rollout(model, batch, 40, initial_position=initial)
    terminal_gradient = torch.autograd.grad(
        terminal_path[-1].square().mean(), initial
    )[0]

    rotation = proper_rotation(device, batch.protein_pos.dtype)
    translation = torch.tensor([1.25, -2.5, 0.75], device=device)
    rigid_batch = transformed_batch(batch, rotation, translation)
    with torch.no_grad():
        _, reference = rollout(model, batch, 40)
        _, transformed = rollout(model, rigid_batch, 40)
    expected = reference @ rotation.T + translation
    equivariance_error = float((transformed - expected).abs().max().cpu())

    report = {
        "status": "pass",
        "scope": "one real MISATO-100 train complex; published NeuralMD weights",
        "checkpoint": str(args.checkpoint),
        "horizons": horizon_results,
        "protein_bitwise_unchanged": bool(torch.equal(protein_before, batch.protein_pos)),
        "terminal_to_initial_gradient_norm": float(
            torch.linalg.vector_norm(terminal_gradient).detach().cpu()
        ),
        "terminal_to_initial_gradient_finite": bool(torch.isfinite(terminal_gradient).all()),
        "rigid_equivariance_max_error_angstrom": equivariance_error,
        "rigid_equivariance_threshold_angstrom": 1e-4,
    }
    gates = [
        all(item["finite_forward"] and item["finite_backward"]
            and item["parameter_gradient_norm"] > 0
            for item in horizon_results.values()),
        report["protein_bitwise_unchanged"],
        report["terminal_to_initial_gradient_finite"],
        report["terminal_to_initial_gradient_norm"] > 0,
        equivariance_error < 1e-4,
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
