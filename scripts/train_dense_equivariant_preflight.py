"""Train-only 10-complex runtime preflight for dense equivariant dynamics."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torchdiffeq import odeint

from NeuralMD.dataloaders.dataloader_MISATO import DataLoaderMISATO
from NeuralMD.datasets.MISATO import DatasetMISATOSemiFlexibleMultiTrajectory
from protein_quanta.dense_equivariant_dynamics import DenseEquivariantAcceleration
from protein_quanta.neuralmd_multiscale import neuralmd_ode_rollout
from protein_quanta.rollout_loss import (
    multiscale_coordinate_smooth_l1,
    sample_rollout_segment,
)


def read_ids(path):
    return [line.strip().upper() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_indices(train_ids, selected_ids):
    index = {identifier: position for position, identifier in enumerate(train_ids)}
    missing = [identifier for identifier in selected_ids if identifier not in index]
    if missing:
        raise ValueError(f"selected IDs missing from train split: {missing}")
    return [index[identifier] for identifier in selected_ids]


def rollout(model, batch, start, horizon):
    return neuralmd_ode_rollout(
        model, odeint, batch, start=start, horizon=horizon, scaling=100,
        step_size=0.025, method="euler",
    )[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--development-ids", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:2")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    selected_ids = config["data"]["ids"]
    development_ids = read_ids(args.development_ids)
    if any(identifier not in development_ids for identifier in selected_ids):
        raise ValueError("preflight contains IDs outside frozen development split")
    train_ids = read_ids(args.data_root / "raw" / "train_MD.txt")
    indices = select_indices(train_ids, selected_ids)

    seed = config["training"]["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(args.device)
    dataset = DatasetMISATOSemiFlexibleMultiTrajectory(str(args.data_root), mode="train")
    subset = torch.utils.data.Subset(dataset, indices)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoaderMISATO(
        subset, batch_size=1, num_workers=0, shuffle=True, generator=generator
    )
    model = DenseEquivariantAcceleration(hidden_dim=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    epoch_reports = []
    total_batches = clipped_batches = nonfinite_batches = 0
    protein_changed = False
    for epoch in range(1, config["training"]["epochs"] + 1):
        losses = []; local_losses = []; multiscale_losses = []; grad_norms = []
        for batch in loader:
            batch = batch.to(device)
            protein_before = batch.protein_pos.detach().clone()
            end = random.randint(1, 99)
            start = max(0, end - config["training"]["local_max_horizon"])
            local_prediction = rollout(model, batch, start, end - start)
            local_truth = batch.ligand_trajectory_pos[:, start + 1 : end + 1, :].transpose(0, 1)
            local_loss = functional.mse_loss(local_prediction[1:], local_truth)

            multi_start, multi_end, horizon = sample_rollout_segment(
                100, config["training"]["multiscale_horizons"], rng=random
            )
            multi_prediction = rollout(model, batch, multi_start, horizon)
            multi_truth = batch.ligand_trajectory_pos[:, multi_start + 1 : multi_end + 1, :].transpose(0, 1)
            multi_loss = multiscale_coordinate_smooth_l1(multi_prediction[1:], multi_truth)
            loss = local_loss + config["training"]["multiscale_coefficient"] * multi_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            squared_norm = loss.new_zeros(())
            finite_gradient = True
            for parameter in model.parameters():
                if parameter.grad is not None:
                    finite_gradient = finite_gradient and bool(torch.isfinite(parameter.grad).all())
                    squared_norm = squared_norm + parameter.grad.square().sum()
            grad_norm = float(torch.sqrt(squared_norm).detach().cpu())
            finite = bool(torch.isfinite(loss)) and finite_gradient
            if not finite:
                nonfinite_batches += 1
                raise RuntimeError("non-finite loss or gradient in preflight")
            clip = config["training"]["gradient_clip_norm"]
            if grad_norm > clip:
                clipped_batches += 1
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            protein_changed = protein_changed or not torch.equal(protein_before, batch.protein_pos)
            total_batches += 1
            losses.append(float(loss.detach().cpu())); local_losses.append(float(local_loss.detach().cpu()))
            multiscale_losses.append(float(multi_loss.detach().cpu())); grad_norms.append(grad_norm)
        epoch_reports.append({
            "epoch": epoch,
            "loss_mean": float(np.mean(losses)),
            "local_loss_mean": float(np.mean(local_losses)),
            "multiscale_loss_mean": float(np.mean(multiscale_losses)),
            "gradient_norm_mean": float(np.mean(grad_norms)),
            "gradient_norm_max": float(np.max(grad_norms)),
        })
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": config}, args.checkpoint)
    checkpoint_sha256 = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    clipping_rate = clipped_batches / max(total_batches, 1)
    report = {
        "status": "runtime_pass_phys_blocked",
        "scope": "10 frozen development complexes; no validation/test",
        "selected_ids": selected_ids,
        "processed_indices": indices,
        "epochs": epoch_reports,
        "total_batches": total_batches,
        "nonfinite_batches": nonfinite_batches,
        "clipped_batches": clipped_batches,
        "clipping_rate": clipping_rate,
        "protein_coordinate_changed": protein_changed,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "checkpoint_sha256": checkpoint_sha256,
        "phys_gate": "blocked: explicit covalent bond graphs not present in processed data",
    }
    runtime_pass = nonfinite_batches == 0 and clipping_rate <= 0.5 and not protein_changed
    if not runtime_pass:
        report["status"] = "runtime_fail"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not runtime_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
