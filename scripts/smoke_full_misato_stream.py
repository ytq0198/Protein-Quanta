"""Audit one full-MISATO streaming batch and one velocity-aware update."""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as functional
from torch_geometric.loader import DataLoader

from protein_quanta.streaming_misato import StreamingMISATODataset
from protein_quanta.velocity_equivariant_dynamics import VelocityEquivariantAcceleration


def condition_from_batch(batch):
    return (
        batch.ligand_x,
        batch.batch_ligand,
        batch.ligand_mass,
        batch.protein_pos[batch.mask_n],
        batch.protein_pos[batch.mask_ca],
        batch.protein_pos[batch.mask_c],
        batch.protein_backbone_residue,
        batch.batch_residue,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--peptides", type=Path, required=True)
    parser.add_argument("--neuralmd-utils", type=Path, required=True)
    parser.add_argument("--periodic-table", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    started = time.perf_counter()
    dataset = StreamingMISATODataset(
        args.h5, args.split, args.peptides, args.neuralmd_utils,
        args.periodic_table,
    )
    if len(dataset) != args.expected_count:
        raise ValueError(f"filtered count {len(dataset)} != {args.expected_count}")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    loaded = time.perf_counter()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    batch = batch.to(device)
    model = VelocityEquivariantAcceleration(
        hidden_dim=32,
        bounded_damping_max=0.05,
        normalize_velocity_invariants=True,
    ).to(device)
    position = batch.ligand_trajectory_pos[:, 0, :]
    velocity = batch.ligand_trajectory_pos[:, 1, :] - position
    acceleration = model(0, (velocity, position), condition_from_batch(batch))[0]
    target = (
        batch.ligand_trajectory_pos[:, 2, :]
        - 2 * batch.ligand_trajectory_pos[:, 1, :]
        + position
    )
    loss = functional.smooth_l1_loss(acceleration, target)
    loss.backward()
    gradient_norm = torch.sqrt(sum(
        parameter.grad.square().sum()
        for parameter in model.parameters()
        if parameter.grad is not None
    ))
    finished = time.perf_counter()
    checks = {
        "filtered_count_matches": len(dataset) == args.expected_count,
        "one_batch_finite": bool(
            torch.isfinite(batch.ligand_trajectory_pos).all()
            and torch.isfinite(acceleration).all()
            and torch.isfinite(loss)
        ),
        "gradient_finite_nonzero": bool(
            torch.isfinite(gradient_norm) and gradient_norm > 0
        ),
    }
    report = {
        "status": "full_misato_stream_smoke_pass" if all(checks.values()) else "full_misato_stream_smoke_fail",
        "scope": "first filtered training complex only; no validation/test access",
        "dataset": {
            "filtered_count": len(dataset),
            "sample_id": batch.sample_id[0] if isinstance(batch.sample_id, list) else batch.sample_id,
            "ligand_heavy_atoms": int(batch.ligand_x.shape[0]),
            "protein_backbone_atoms": int(batch.protein_pos.shape[0]),
            "frames": int(batch.ligand_trajectory_pos.shape[1]),
        },
        "model": {
            "variant": "bounded normalized velocity-aware",
            "parameter_count": sum(p.numel() for p in model.parameters()),
        },
        "loss": float(loss.detach().cpu()),
        "gradient_norm": float(gradient_norm.detach().cpu()),
        "timing_seconds": {
            "construct_and_first_batch": loaded - started,
            "forward_backward": finished - loaded,
        },
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    dataset.close()
    if not all(checks.values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
