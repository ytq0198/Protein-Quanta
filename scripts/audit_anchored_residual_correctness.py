"""Real-complex correctness gate for control-anchored residual dynamics."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as functional

from NeuralMD.dataloaders.dataloader_MISATO import DataLoaderMISATO
from protein_quanta.anchored_velocity_residual import AnchoredGatedVelocityDynamics
from protein_quanta.streaming_misato import StreamingMISATODataset
from protein_quanta.velocity_equivariant_dynamics import VelocityEquivariantAcceleration


def condition_from_batch(batch):
    return (
        batch.ligand_x, batch.batch_ligand, batch.ligand_mass,
        batch.protein_pos[batch.mask_n], batch.protein_pos[batch.mask_ca],
        batch.protein_pos[batch.mask_c], batch.protein_backbone_residue,
        batch.batch_residue,
    )


def orthogonal(device, dtype, reflection=False):
    generator = torch.Generator(device="cpu").manual_seed(20260814)
    matrix = torch.randn(3, 3, generator=generator, dtype=dtype).to(device)
    matrix, _ = torch.linalg.qr(matrix)
    desired = -1 if reflection else 1
    if torch.sign(torch.linalg.det(matrix)).item() != desired:
        matrix[:, -1] = -matrix[:, -1]
    return matrix


def transformed_batch(batch, matrix, translation):
    transformed = batch.clone()
    transformed.protein_pos = batch.protein_pos @ matrix.T + translation
    transformed.ligand_trajectory_pos = (
        batch.ligand_trajectory_pos @ matrix.T + translation
    )
    return transformed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--peptides", type=Path, required=True)
    parser.add_argument("--neuralmd-utils", type=Path, required=True)
    parser.add_argument("--periodic-table", type=Path, required=True)
    parser.add_argument("--anchor-checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    observed_hash = hashlib.sha256(args.anchor_checkpoint.read_bytes()).hexdigest()
    if observed_hash != config["anchor"]["checkpoint_sha256"]:
        raise ValueError("anchor checkpoint SHA256 differs from preregistration")
    device = torch.device(args.device)
    dataset = StreamingMISATODataset(
        args.h5, args.split, args.peptides, args.neuralmd_utils,
        args.periodic_table,
    )
    sample_id = "4K6V"
    index = dataset.sample_ids.index(sample_id)
    batch = next(iter(DataLoaderMISATO(
        torch.utils.data.Subset(dataset, [index]),
        batch_size=1, shuffle=False, num_workers=0,
    ))).to(device)
    dataset.close()

    anchor = VelocityEquivariantAcceleration(hidden_dim=32).to(device)
    checkpoint = torch.load(args.anchor_checkpoint, map_location=device, weights_only=False)
    anchor.load_state_dict(checkpoint["model"])
    residual = VelocityEquivariantAcceleration(
        hidden_dim=32, bounded_damping_max=0.05,
        normalize_velocity_invariants=True,
    ).to(device)
    model = AnchoredGatedVelocityDynamics(
        anchor, residual, hidden_dim=32,
        maximum_gate=config["residual"]["maximum_gate"],
        initial_gate_fraction=config["residual"]["initial_gate_fraction"],
        freeze_anchor=True,
    ).to(device)
    condition = condition_from_batch(batch)
    position = batch.ligand_trajectory_pos[:, 0, :]
    velocity = batch.ligand_trajectory_pos[:, 1, :] - position
    target = (
        batch.ligand_trajectory_pos[:, 2, :]
        - 2 * batch.ligand_trajectory_pos[:, 1, :]
        + position
    )
    anchor_before = {
        name: value.detach().clone() for name, value in model.anchor.state_dict().items()
    }
    anchor_acceleration = model.anchor(0, (velocity, position), condition)[0]
    initial_acceleration = model(0, (velocity, position), condition)[0]
    initial_error = float((initial_acceleration - anchor_acceleration).abs().max().cpu())
    loss = functional.smooth_l1_loss(initial_acceleration, target)
    scale_gradient = torch.autograd.grad(
        loss, model.residual_scale_logit, retain_graph=True
    )[0]
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config["residual"]["single_step_learning_rate"],
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    post_acceleration = model(0, (velocity, position), condition)[0]
    correction = post_acceleration - anchor_acceleration
    correction_ratio = float(
        (torch.linalg.vector_norm(correction)
         / torch.linalg.vector_norm(anchor_acceleration).clamp_min(1e-12)).detach().cpu()
    )
    anchor_unchanged = all(
        torch.equal(value, anchor_before[name])
        for name, value in model.anchor.state_dict().items()
    )

    translation = torch.tensor([1.25, -2.5, 0.75], device=device)
    errors = {}
    for name, reflection in (("proper", False), ("reflection", True)):
        matrix = orthogonal(device, position.dtype, reflection=reflection)
        transformed = transformed_batch(batch, matrix, translation)
        transformed_condition = condition_from_batch(transformed)
        transformed_position = transformed.ligand_trajectory_pos[:, 0, :]
        transformed_velocity = (
            transformed.ligand_trajectory_pos[:, 1, :] - transformed_position
        )
        with torch.no_grad():
            transformed_acceleration = model(
                0, (transformed_velocity, transformed_position), transformed_condition
            )[0]
        expected = post_acceleration.detach() @ matrix.T
        errors[name] = float((transformed_acceleration - expected).abs().max().cpu())

    thresholds = config["gates"]
    checks = {
        "initial_exact_anchor_match": initial_error <= thresholds["initial_anchor_max_abs_error"],
        "initial_scale_gradient_finite_nonzero": bool(
            torch.isfinite(scale_gradient) and scale_gradient.abs() > 0
        ),
        "anchor_frozen_unchanged": (
            all(not parameter.requires_grad for parameter in model.anchor.parameters())
            and anchor_unchanged
        ),
        "post_step_correction_finite_nonzero": bool(
            torch.isfinite(correction).all() and torch.linalg.vector_norm(correction) > 0
        ),
        "post_step_correction_bounded": correction_ratio <= thresholds["post_step_correction_relative_l2_max"],
        "proper_equivariance": errors["proper"] <= thresholds["proper_equivariance_max_abs_error"],
        "reflection_equivariance": errors["reflection"] <= thresholds["reflection_equivariance_max_abs_error"],
    }
    report = {
        "status": "anchored_residual_correctness_pass" if all(checks.values()) else "anchored_residual_correctness_fail",
        "scope": config["scope"],
        "sample_id": sample_id,
        "anchor_checkpoint_sha256": observed_hash,
        "initial_anchor_max_abs_error": initial_error,
        "initial_scale_gradient": float(scale_gradient.detach().cpu()),
        "initial_gate_mean": float(model.gate_coefficient(condition[0], velocity).mean().detach().cpu()),
        "post_step_residual_scale": float(torch.tanh(model.residual_scale_logit).detach().cpu()),
        "post_step_correction_relative_l2": correction_ratio,
        "equivariance_max_abs_error": errors,
        "checks": checks,
        "decision": config["decision"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
