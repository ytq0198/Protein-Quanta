"""Paired 8-complex dense dynamics preflight with explicit-bond Phys audit."""

import argparse
import copy
import hashlib
import json
import random
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as functional
from torchdiffeq import odeint

from NeuralMD.dataloaders.dataloader_MISATO import DataLoaderMISATO
from NeuralMD.datasets.MISATO import DatasetMISATOSemiFlexibleMultiTrajectory
from protein_quanta.collision import bond_length_diagnostics
from protein_quanta.dense_equivariant_dynamics import DenseEquivariantAcceleration
from protein_quanta.neuralmd_multiscale import neuralmd_ode_rollout
from protein_quanta.rollout_loss import multiscale_coordinate_smooth_l1


def read_ids(path):
    return [line.strip().upper() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rollout(model, batch, start, horizon):
    return neuralmd_ode_rollout(
        model, odeint, batch, start=start, horizon=horizon, scaling=100,
        step_size=0.025, method="euler",
    )[1]


def make_schedule(sample_count, epochs, seed):
    order_rng = random.Random(seed)
    local_rng = random.Random(seed + 1)
    multi_rng = random.Random(seed + 2)
    schedule = []
    horizons = (5, 10, 20, 40)
    for epoch in range(epochs):
        order = list(range(sample_count)); order_rng.shuffle(order)
        rows = []
        for sample_index in order:
            local_end = local_rng.randint(1, 99)
            horizon = horizons[multi_rng.randrange(len(horizons))]
            multi_start = multi_rng.randint(0, 99 - horizon)
            rows.append((sample_index, local_end, multi_start, horizon))
        schedule.append(rows)
    return schedule


def prepare_single_complex(sample, device):
    """Add the batch vectors normally created by DataLoaderMISATO."""
    sample = sample.to(device)
    sample.batch_ligand = torch.zeros(
        sample.ligand_x.shape[0], dtype=torch.long, device=device
    )
    sample.batch_residue = torch.zeros(
        sample.protein_backbone_residue.shape[0], dtype=torch.long, device=device
    )
    return sample


def train_one(model, samples, schedule, coefficient, learning_rate, clip):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    rows = []; clipped = nonfinite = 0
    for epoch_index, epoch_schedule in enumerate(schedule, start=1):
        losses = []; gradients = []
        for sample_index, local_end, multi_start, horizon in epoch_schedule:
            batch = samples[sample_index]
            local_start = max(0, local_end - 20)
            local_prediction = rollout(model, batch, local_start, local_end - local_start)
            local_truth = batch.ligand_trajectory_pos[:, local_start + 1 : local_end + 1, :].transpose(0, 1)
            local_loss = functional.mse_loss(local_prediction[1:], local_truth)
            multi_prediction = rollout(model, batch, multi_start, horizon)
            multi_truth = batch.ligand_trajectory_pos[:, multi_start + 1 : multi_start + horizon + 1, :].transpose(0, 1)
            multi_loss = multiscale_coordinate_smooth_l1(multi_prediction[1:], multi_truth)
            loss = local_loss + coefficient * multi_loss
            optimizer.zero_grad(set_to_none=True); loss.backward()
            squared = loss.new_zeros(()); finite = bool(torch.isfinite(loss))
            for parameter in model.parameters():
                if parameter.grad is not None:
                    finite = finite and bool(torch.isfinite(parameter.grad).all())
                    squared = squared + parameter.grad.square().sum()
            norm = float(torch.sqrt(squared).detach().cpu())
            if not finite:
                nonfinite += 1; raise RuntimeError("non-finite paired preflight")
            clipped += int(norm > clip)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip); optimizer.step()
            losses.append(float(loss.detach().cpu())); gradients.append(norm)
        rows.append({
            "epoch": epoch_index, "loss_mean": float(np.mean(losses)),
            "gradient_norm_mean": float(np.mean(gradients)),
            "gradient_norm_max": float(np.max(gradients)),
        })
    return rows, clipped, nonfinite


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--topology-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:2")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    topology_report = json.loads(args.topology_report.read_text(encoding="utf-8"))
    topology = {row["sample_id"]: row for row in topology_report["samples"] if row["status"] == "matched"}
    ids = config["data"]["topology_covered_ids"]
    if set(ids) != set(topology):
        raise ValueError("topology coverage differs from preregistration")
    train_ids = read_ids(args.data_root / "raw" / "train_MD.txt")
    index = {identifier: position for position, identifier in enumerate(train_ids)}
    device = torch.device(args.device)
    dataset = DatasetMISATOSemiFlexibleMultiTrajectory(str(args.data_root), mode="train")
    samples = [
        prepare_single_complex(dataset[index[identifier]], device)
        for identifier in ids
    ]
    seed = config["pairing"]["seed"]
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    initial = DenseEquivariantAcceleration(hidden_dim=32).to(device)
    control = copy.deepcopy(initial); candidate = copy.deepcopy(initial)
    schedule = make_schedule(len(samples), config["pairing"]["epochs"], seed)
    common = (config["pairing"]["learning_rate"], config["pairing"]["gradient_clip_norm"])
    control_epochs, control_clipped, control_nonfinite = train_one(control, samples, schedule, 0.0, *common)
    candidate_epochs, candidate_clipped, candidate_nonfinite = train_one(candidate, samples, schedule, 0.25, *common)

    metrics = {"control": [], "candidate": []}
    for identifier, batch in zip(ids, samples):
        truth = batch.ligand_trajectory_pos[:, 1:41, :].transpose(0, 1).detach().cpu().numpy()
        bonds = np.asarray(topology[identifier]["bonds"], dtype=int)
        for name, model in (("control", control), ("candidate", candidate)):
            with torch.no_grad():
                prediction = rollout(model, batch, 0, 40)[1:].cpu().numpy()
            phys = bond_length_diagnostics(prediction, truth, bonds)
            coordinate_rmse = float(np.sqrt(np.mean((prediction - truth) ** 2)))
            metrics[name].append({"sample_id": identifier, "coordinate_rmse": coordinate_rmse, **phys})
    summary = {
        name: {
            metric: float(np.mean([row[metric] for row in rows]))
            for metric in ("coordinate_rmse", "bond_length_mae_angstrom", "extreme_bond_event_percent")
        } for name, rows in metrics.items()
    }
    extreme_delta = summary["candidate"]["extreme_bond_event_percent"] - summary["control"]["extreme_bond_event_percent"]
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, model in (("control", control), ("candidate", candidate)):
        path = args.checkpoint_dir / f"{name}_final.pth"
        torch.save({"model": model.state_dict(), "config": config}, path)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = {
        "status": "phys_pass_runtime_pass" if extreme_delta <= 0.1 else "phys_fail",
        "scope": "8/10 topology-covered frozen development complexes; no validation/test",
        "coverage_fraction": 0.8,
        "omitted_ids": config["data"]["topology_missing_ids"],
        "schedule": schedule,
        "training": {
            "control": {"epochs": control_epochs, "clipped": control_clipped, "nonfinite": control_nonfinite},
            "candidate": {"epochs": candidate_epochs, "clipped": candidate_clipped, "nonfinite": candidate_nonfinite},
        },
        "summary": summary,
        "candidate_minus_control_extreme_percentage_points": extreme_delta,
        "phys_threshold_percentage_points": 0.1,
        "samples": metrics,
        "checkpoint_sha256": hashes,
        "decision": "necessary preflight only; not an effect gate or permission to access validation/test",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in report if key not in ("schedule", "samples")}, indent=2))
    if report["status"] == "phys_fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
