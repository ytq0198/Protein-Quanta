"""Paired development-only learnability preflight for ODE time parameterization."""

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
from protein_quanta.metrics import coordinate_rmse, dynamics_distribution_metrics
from protein_quanta.neuralmd_multiscale import neuralmd_ode_rollout
from protein_quanta.rollout_loss import multiscale_coordinate_smooth_l1
from protein_quanta.scenarios import competition_scenarios
from scripts.train_evaluate_dense_paired_phys import (
    make_schedule,
    prepare_single_complex,
    read_ids,
)


def rollout(model, sample, start, horizon, protocol):
    return neuralmd_ode_rollout(
        model,
        odeint,
        sample,
        start=start,
        horizon=horizon,
        scaling=protocol["scaling"],
        step_size=protocol["euler_step_size"],
        method="euler",
        initial_velocity_scale=protocol["initial_velocity_scale"],
    )[1]


def train(model, samples, schedule, protocol, settings):
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    epochs = []
    clipped = nonfinite = updates = 0
    for epoch_index, rows in enumerate(schedule, start=1):
        losses = []
        gradient_norms = []
        for sample_index, local_end, multi_start, horizon in rows:
            sample = samples[sample_index]
            local_start = max(0, local_end - settings["local_max_horizon"])
            local_prediction = rollout(
                model, sample, local_start, local_end - local_start, protocol
            )
            local_truth = sample.ligand_trajectory_pos[
                :, local_start + 1 : local_end + 1, :
            ].transpose(0, 1)
            local_loss = functional.mse_loss(local_prediction[1:], local_truth)
            multiscale_prediction = rollout(
                model, sample, multi_start, horizon, protocol
            )
            multiscale_truth = sample.ligand_trajectory_pos[
                :, multi_start + 1 : multi_start + horizon + 1, :
            ].transpose(0, 1)
            multiscale_loss = multiscale_coordinate_smooth_l1(
                multiscale_prediction[1:], multiscale_truth
            )
            loss = local_loss + settings["multiscale_coefficient"] * multiscale_loss
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
                raise RuntimeError("non-finite frame-time learnability update")
            clipped += int(norm > settings["gradient_clip_norm"])
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), settings["gradient_clip_norm"]
            )
            optimizer.step()
            updates += 1
            losses.append(float(loss.detach().cpu()))
            gradient_norms.append(norm)
        epoch = {
            "epoch": epoch_index,
            "loss_mean": float(np.mean(losses)),
            "gradient_norm_mean": float(np.mean(gradient_norms)),
            "gradient_norm_max": float(np.max(gradient_norms)),
        }
        epochs.append(epoch)
        print(json.dumps({"protocol": protocol, **epoch}), flush=True)
    return {"epochs": epochs, "updates": updates, "clipped": clipped, "nonfinite": nonfinite}


def evaluate(model, samples, protocol):
    rows = {scenario.name: [] for scenario in competition_scenarios()}
    with torch.no_grad():
        for sample in samples:
            for scenario in competition_scenarios():
                first, _ = scenario.initializer_indices
                path = rollout(
                    model, sample, first, scenario.target_end - first, protocol
                )
                prediction = path[2:].cpu().numpy()
                truth = sample.ligand_trajectory_pos[
                    :, scenario.target_start : scenario.target_end + 1, :
                ].transpose(0, 1).cpu().numpy()
                dynamics = dynamics_distribution_metrics(
                    prediction, truth, maximum_lag=10
                )
                rows[scenario.name].append({
                    "coordinate_rmse_angstrom": coordinate_rmse(prediction, truth),
                    "step_amplitude_ratio": dynamics["step_amplitude_ratio"],
                    "nonfinite_frame_fraction": float(
                        1.0 - np.mean(np.isfinite(prediction).all(axis=(1, 2)))
                    ),
                })
    return {
        scenario: {
            metric: float(np.mean([row[metric] for row in values]))
            for metric in (
                "coordinate_rmse_angstrom",
                "step_amplitude_ratio",
                "nonfinite_frame_fraction",
            )
        }
        for scenario, values in rows.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:3")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    development = json.loads(args.development_manifest.read_text(encoding="utf-8"))
    allowed = set(development["development_ids"])
    train_ids = config["data"]["train_ids"]
    diagnostic_ids = config["data"]["diagnostic_ids"]
    if len(train_ids) != 48 or len(diagnostic_ids) != 16:
        raise ValueError("preflight requires a frozen 48/16 development split")
    if set(train_ids).isdisjoint(diagnostic_ids) is False:
        raise ValueError("training and diagnostic sets overlap")
    if set(train_ids) | set(diagnostic_ids) != allowed:
        raise ValueError("preflight split differs from frozen 64-complex development set")

    official_train_ids = read_ids(args.data_root / "raw" / "train_MD.txt")
    index = {identifier: position for position, identifier in enumerate(official_train_ids)}
    device = torch.device(args.device)
    dataset = DatasetMISATOSemiFlexibleMultiTrajectory(str(args.data_root), mode="train")
    training_samples = [
        prepare_single_complex(dataset[index[identifier]], device)
        for identifier in train_ids
    ]
    diagnostic_samples = [
        prepare_single_complex(dataset[index[identifier]], device)
        for identifier in diagnostic_ids
    ]

    settings = config["training"]
    seed = settings["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    initial = DenseEquivariantAcceleration(hidden_dim=32).to(device)
    models = {
        name: copy.deepcopy(initial) for name in config["protocols"]
    }
    schedule = make_schedule(len(training_samples), settings["epochs"], seed)
    initial_metrics = {
        name: evaluate(initial, diagnostic_samples, protocol)
        for name, protocol in config["protocols"].items()
    }
    training = {
        name: train(models[name], training_samples, schedule, protocol, settings)
        for name, protocol in config["protocols"].items()
    }
    final_metrics = {
        name: evaluate(models[name], diagnostic_samples, protocol)
        for name, protocol in config["protocols"].items()
    }

    frame_initial = initial_metrics["frame_time"]["T3"]
    frame_final = final_metrics["frame_time"]["T3"]
    thresholds = config["gate"]
    checks = {
        "all_updates_finite": all(
            result["nonfinite"] == 0 for result in training.values()
        ),
        "frame_time_T3_rmse_improves": (
            (frame_initial["coordinate_rmse_angstrom"]
             - frame_final["coordinate_rmse_angstrom"])
            / frame_initial["coordinate_rmse_angstrom"]
            >= thresholds["frame_time_T3_rmse_improvement_from_initial_min_fraction"]
        ),
        "frame_time_T3_amplitude_in_range": (
            thresholds["frame_time_T3_step_amplitude_ratio_min"]
            <= frame_final["step_amplitude_ratio"]
            <= thresholds["frame_time_T3_step_amplitude_ratio_max"]
        ),
        "all_diagnostic_rollouts_finite": all(
            metrics[scenario]["nonfinite_frame_fraction"] == 0
            for metrics in final_metrics.values()
            for scenario in ("T1", "T2", "T3")
        ),
    }
    passed = all(checks.values())
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_sha256 = {}
    for name, model in models.items():
        path = args.checkpoint_dir / f"{name}_final.pth"
        torch.save({"model": model.state_dict(), "config": config}, path)
        checkpoint_sha256[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = {
        "status": "learnability_gate_pass" if passed else "learnability_gate_fail",
        "scope": "48 train / 16 diagnostic inside frozen development; old holdout and official validation/test untouched",
        "protocols": config["protocols"],
        "initial_metrics": initial_metrics,
        "training": training,
        "final_metrics": final_metrics,
        "checks": checks,
        "checkpoint_sha256": checkpoint_sha256,
        "interpretation": thresholds["interpretation"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "initial_metrics", "final_metrics", "checks")}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
