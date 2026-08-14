"""Paired steady-state dense/Top-k benchmark on one full-MISATO train complex."""

import argparse
import copy
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as functional

from NeuralMD.dataloaders.dataloader_MISATO import DataLoaderMISATO
from protein_quanta.streaming_misato import StreamingMISATODataset
from protein_quanta.velocity_equivariant_dynamics import VelocityEquivariantAcceleration


def condition_from_batch(batch):
    return (
        batch.ligand_x, batch.batch_ligand, batch.ligand_mass,
        batch.protein_pos[batch.mask_n], batch.protein_pos[batch.mask_ca],
        batch.protein_pos[batch.mask_c], batch.protein_backbone_residue,
        batch.batch_residue,
    )


def acceleration_loss(model, velocity, position, condition, target):
    acceleration = model(0, (velocity, position), condition)[0]
    return acceleration, functional.smooth_l1_loss(acceleration, target)


def benchmark(model, velocity, position, condition, target, warmup, iterations, device):
    for _ in range(warmup):
        model.zero_grad(set_to_none=True)
        _, loss = acceleration_loss(model, velocity, position, condition, target)
        loss.backward()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    gradient_norms = []
    losses = []
    for _ in range(iterations):
        model.zero_grad(set_to_none=True)
        _, loss = acceleration_loss(model, velocity, position, condition, target)
        loss.backward()
        gradient_norm = torch.sqrt(sum(
            parameter.grad.square().sum()
            for parameter in model.parameters()
            if parameter.grad is not None
        ))
        losses.append(float(loss.detach().cpu()))
        gradient_norms.append(float(gradient_norm.detach().cpu()))
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    with torch.no_grad():
        acceleration = model(0, (velocity, position), condition)[0]
    return {
        "seconds_per_forward_backward": elapsed / iterations,
        "loss": losses[-1],
        "gradient_norm": gradient_norms[-1],
        "all_finite_nonzero": all(
            torch.isfinite(torch.tensor(values)).all() and all(value > 0 for value in values)
            for values in (losses, gradient_norms)
        ),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }, acceleration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--peptides", type=Path, required=True)
    parser.add_argument("--neuralmd-utils", type=Path, required=True)
    parser.add_argument("--periodic-table", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    torch.manual_seed(20260814)
    device = torch.device(args.device)
    device_index = device.index if device.index is not None else 0
    torch.cuda.set_device(device_index)

    dataset = StreamingMISATODataset(
        args.h5, args.split, args.peptides, args.neuralmd_utils,
        args.periodic_table,
    )
    batch = next(iter(DataLoaderMISATO(
        dataset, batch_size=1, shuffle=False, num_workers=0
    ))).to(device)
    position = batch.ligand_trajectory_pos[:, 0, :]
    velocity = batch.ligand_trajectory_pos[:, 1, :] - position
    target = (
        batch.ligand_trajectory_pos[:, 2, :]
        - 2 * batch.ligand_trajectory_pos[:, 1, :]
        + position
    )
    condition = condition_from_batch(batch)
    base = VelocityEquivariantAcceleration(
        hidden_dim=32, bounded_damping_max=0.05,
        normalize_velocity_invariants=True,
    ).to(device)
    models = {"dense": base}
    for top_k in config["model"]["protein_top_k_candidates"]:
        model = VelocityEquivariantAcceleration(
            hidden_dim=32, bounded_damping_max=0.05,
            normalize_velocity_invariants=True, protein_top_k=top_k,
        ).to(device)
        model.load_state_dict(copy.deepcopy(base.state_dict()))
        models[f"top_k_{top_k}"] = model

    rows = {}
    accelerations = {}
    benchmark_config = config["benchmark"]
    for name, model in models.items():
        rows[name], accelerations[name] = benchmark(
            model, velocity, position, condition, target,
            benchmark_config["warmup_updates"],
            benchmark_config["measured_updates"], device_index,
        )
    dense_time = rows["dense"]["seconds_per_forward_backward"]
    dense_acceleration = accelerations["dense"]
    dense_norm = torch.linalg.vector_norm(dense_acceleration).clamp_min(1e-12)
    thresholds = config["gate"]
    for name, row in rows.items():
        difference = accelerations[name] - dense_acceleration
        row["dense_speedup"] = dense_time / row["seconds_per_forward_backward"]
        row["relative_acceleration_l2_error"] = float(
            (torch.linalg.vector_norm(difference) / dense_norm).cpu()
        )
        row["acceleration_max_abs_error"] = float(difference.abs().max().cpu())
        row["passes"] = (
            name != "dense"
            and row["dense_speedup"] >= thresholds["minimum_dense_speedup"]
            and row["relative_acceleration_l2_error"]
            <= thresholds["maximum_relative_acceleration_l2_error"]
            and row["all_finite_nonzero"]
        )
    passing = [
        int(name.rsplit("_", 1)[-1]) for name, row in rows.items() if row["passes"]
    ]
    selected = min(passing) if passing else None
    report = {
        "status": "sparse_throughput_gate_pass" if selected else "sparse_throughput_gate_fail",
        "scope": config["scope"],
        "sample": {
            "sample_id": dataset.sample_ids[0],
            "ligand_heavy_atoms": int(batch.ligand_x.shape[0]),
            "protein_residues": int(batch.protein_backbone_residue.shape[0]),
        },
        "benchmark": benchmark_config,
        "thresholds": thresholds,
        "models": rows,
        "selected_protein_top_k": selected,
        "decision_boundary": config["decision_boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    dataset.close()


if __name__ == "__main__":
    main()
