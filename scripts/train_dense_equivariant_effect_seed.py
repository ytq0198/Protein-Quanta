"""Train one paired seed without knowing or reading the frozen holdout split."""

import argparse
import copy
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torchdiffeq import odeint

from NeuralMD.datasets.MISATO import DatasetMISATOSemiFlexibleMultiTrajectory
from protein_quanta.dense_equivariant_dynamics import DenseEquivariantAcceleration
from protein_quanta.neuralmd_multiscale import neuralmd_ode_rollout
from protein_quanta.rollout_loss import multiscale_coordinate_smooth_l1
from scripts.train_evaluate_dense_paired_phys import (
    make_schedule,
    prepare_single_complex,
    read_ids,
)


def rollout(model, sample, start, horizon):
    return neuralmd_ode_rollout(
        model, odeint, sample, start=start, horizon=horizon, scaling=100,
        step_size=0.025, method="euler",
    )[1]


def parameter_vector(model):
    return torch.cat([parameter.detach().flatten().cpu() for parameter in model.parameters()])


def train(model, samples, schedule, coefficient, learning_rate, clip):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epoch_rows = []
    clipped = nonfinite = 0
    for epoch_index, rows in enumerate(schedule, start=1):
        losses = []
        local_losses = []
        multiscale_losses = []
        gradients = []
        for sample_index, local_end, multi_start, horizon in rows:
            sample = samples[sample_index]
            local_start = max(0, local_end - 20)
            local_prediction = rollout(model, sample, local_start, local_end - local_start)
            local_truth = sample.ligand_trajectory_pos[
                :, local_start + 1 : local_end + 1, :
            ].transpose(0, 1)
            local_loss = functional.mse_loss(local_prediction[1:], local_truth)
            multi_prediction = rollout(model, sample, multi_start, horizon)
            multi_truth = sample.ligand_trajectory_pos[
                :, multi_start + 1 : multi_start + horizon + 1, :
            ].transpose(0, 1)
            multi_loss = multiscale_coordinate_smooth_l1(
                multi_prediction[1:], multi_truth
            )
            loss = local_loss + coefficient * multi_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            squared = loss.new_zeros(())
            finite = bool(torch.isfinite(loss))
            for parameter in model.parameters():
                if parameter.grad is not None:
                    finite = finite and bool(torch.isfinite(parameter.grad).all())
                    squared = squared + parameter.grad.square().sum()
            norm = float(torch.sqrt(squared).detach().cpu())
            if not finite:
                nonfinite += 1
                raise RuntimeError("non-finite effect-gate update")
            clipped += int(norm > clip)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            local_losses.append(float(local_loss.detach().cpu()))
            multiscale_losses.append(float(multi_loss.detach().cpu()))
            gradients.append(norm)
        epoch_rows.append({
            "epoch": epoch_index,
            "loss_mean": float(np.mean(losses)),
            "local_loss_mean": float(np.mean(local_losses)),
            "multiscale_loss_mean": float(np.mean(multiscale_losses)),
            "gradient_norm_mean": float(np.mean(gradients)),
            "gradient_norm_max": float(np.max(gradients)),
        })
        print(
            f"epoch {epoch_index} coefficient {coefficient} "
            f"loss {np.mean(losses):.8f}",
            flush=True,
        )
    return epoch_rows, clipped, nonfinite


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--development-ids", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:2")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.seed not in config["pairing"]["seeds"]:
        raise ValueError("seed is not preregistered")
    development_ids = read_ids(args.development_ids)
    if len(development_ids) != config["data"]["development_count"]:
        raise ValueError("invalid frozen development split")

    train_ids = read_ids(args.data_root / "raw" / "train_MD.txt")
    index = {identifier: position for position, identifier in enumerate(train_ids)}
    device = torch.device(args.device)
    dataset = DatasetMISATOSemiFlexibleMultiTrajectory(str(args.data_root), mode="train")
    development = [
        prepare_single_complex(dataset[index[identifier]], device)
        for identifier in development_ids
    ]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    initial = DenseEquivariantAcceleration(hidden_dim=32).to(device)
    initial_vector = parameter_vector(initial)
    control = copy.deepcopy(initial)
    candidate = copy.deepcopy(initial)
    schedule = make_schedule(
        len(development), config["pairing"]["epochs"], args.seed
    )
    common = (
        config["pairing"]["learning_rate"],
        config["pairing"]["gradient_clip_norm"],
    )
    control_epochs, control_clipped, control_nonfinite = train(
        control,
        development,
        schedule,
        config["pairing"]["control_multiscale_coefficient"],
        *common,
    )
    candidate_epochs, candidate_clipped, candidate_nonfinite = train(
        candidate,
        development,
        schedule,
        config["pairing"]["candidate_multiscale_coefficient"],
        *common,
    )

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_hashes = {}
    for name, model in (("control", control), ("candidate", candidate)):
        path = args.checkpoint_dir / f"{name}_seed_{args.seed}_final.pth"
        torch.save(
            {"model": model.state_dict(), "seed": args.seed, "config": config},
            path,
        )
        checkpoint_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()

    control_vector = parameter_vector(control)
    candidate_vector = parameter_vector(candidate)
    report = {
        "status": "training_complete_holdout_unread",
        "scope": "64 frozen development complexes only",
        "seed": args.seed,
        "development_ids": development_ids,
        "training": {
            "control": {
                "epochs": control_epochs,
                "clipped": control_clipped,
                "nonfinite": control_nonfinite,
            },
            "candidate": {
                "epochs": candidate_epochs,
                "clipped": candidate_clipped,
                "nonfinite": candidate_nonfinite,
            },
        },
        "parameter_l2": {
            "control_from_initial": float(
                torch.linalg.vector_norm(control_vector - initial_vector)
            ),
            "candidate_from_initial": float(
                torch.linalg.vector_norm(candidate_vector - initial_vector)
            ),
            "candidate_from_control": float(
                torch.linalg.vector_norm(candidate_vector - control_vector)
            ),
        },
        "checkpoint_sha256": checkpoint_hashes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "seed": args.seed,
        "parameter_l2": report["parameter_l2"],
        "checkpoint_sha256": checkpoint_hashes,
    }, indent=2))


if __name__ == "__main__":
    main()
