"""Train and evaluate a minimal ProAR-inspired anti-drift mechanism."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from protein_quanta.anti_drift import (
    BridgeInterpolator,
    EndpointForecaster,
    block_rollout,
    parameter_count,
)
from protein_quanta.scenarios import competition_scenarios
from protein_quanta.temporal_features import FEATURE_NAMES
from scripts.train_temporal_architecture_screen import (
    _load_features,
    _normalization,
    _read_split,
)


def _seed_everything(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _sample_windows(values, count, maximum_horizon, generator):
    batch, frames, _ = values.shape
    trajectory = torch.randint(batch, (count,), generator=generator)
    horizon = torch.randint(1, maximum_horizon + 1, (count,), generator=generator)
    latest_start = frames - horizon
    unit = torch.rand(count, generator=generator)
    start_index = torch.floor(unit * latest_start.float()).long()
    return trajectory, start_index, horizon


def _gather(values, trajectory, frame):
    return values[trajectory, frame]


def _train_interpolator(model, train, settings, device, generator):
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    criterion = nn.MSELoss()
    train = torch.as_tensor(train, dtype=torch.float32)
    losses = []
    samples_per_epoch = train.shape[0] * settings["windows_per_trajectory_per_epoch"]
    for _ in range(settings["interpolator_epochs"]):
        trajectory, start_index, horizon = _sample_windows(
            train, samples_per_epoch, settings["maximum_horizon"], generator
        )
        eligible = horizon > 1
        trajectory, start_index, horizon = (
            trajectory[eligible], start_index[eligible], horizon[eligible]
        )
        step = 1 + torch.floor(
            torch.rand(step_count := horizon.numel(), generator=generator)
            * (horizon - 1).float()
        ).long()
        permutation = torch.randperm(step_count, generator=generator)
        epoch_losses = []
        for offset in range(0, step_count, settings["batch_size"]):
            index = permutation[offset : offset + settings["batch_size"]]
            tr = trajectory[index]
            start_i = start_index[index]
            h = horizon[index]
            j = step[index]
            start = _gather(train, tr, start_i).to(device)
            endpoint = _gather(train, tr, start_i + h).to(device)
            target = _gather(train, tr, start_i + j).to(device)
            prediction = model(start, endpoint, j.to(device), h.to(device))
            loss = criterion(prediction, target)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip_norm"])
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)))
    return losses


def _train_forecaster(interpolator, model, train, settings, device, generator):
    for parameter in interpolator.parameters():
        parameter.requires_grad_(False)
    interpolator.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    criterion = nn.MSELoss()
    train = torch.as_tensor(train, dtype=torch.float32)
    losses = []
    samples_per_epoch = train.shape[0] * settings["windows_per_trajectory_per_epoch"]
    for _ in range(settings["forecaster_epochs"]):
        trajectory, start_index, horizon = _sample_windows(
            train, samples_per_epoch, settings["maximum_horizon"], generator
        )
        step = torch.floor(
            torch.rand(samples_per_epoch, generator=generator) * horizon.float()
        ).long()
        permutation = torch.randperm(samples_per_epoch, generator=generator)
        epoch_losses = []
        for offset in range(0, samples_per_epoch, settings["batch_size"]):
            index = permutation[offset : offset + settings["batch_size"]]
            tr = trajectory[index]
            start_i = start_index[index]
            h = horizon[index]
            j = step[index]
            start = _gather(train, tr, start_i).to(device)
            endpoint = _gather(train, tr, start_i + h).to(device)
            with torch.no_grad():
                intermediate = start.clone()
                interior = j > 0
                if torch.any(interior):
                    intermediate[interior] = interpolator(
                        start[interior], endpoint[interior],
                        j[interior].to(device), h[interior].to(device),
                    )
            prediction = model(intermediate, start, j.to(device), h.to(device))
            loss = criterion(prediction, endpoint)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip_norm"])
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)))
    return losses


def _evaluate(interpolator, forecaster, validation, device, maximum_horizon):
    interpolator.eval()
    forecaster.eval()
    values = torch.as_tensor(validation, dtype=torch.float32, device=device)
    report = {"one_pass": {}, "alternating": {}}
    with torch.no_grad():
        for scenario in competition_scenarios():
            observed = values[:, scenario.observed_start : scenario.observed_end + 1]
            truth = values[:, scenario.target_start : scenario.target_end + 1]
            for name, alternating in (("one_pass", False), ("alternating", True)):
                prediction = block_rollout(
                    interpolator, forecaster, observed, truth.shape[1],
                    maximum_horizon=maximum_horizon, alternating=alternating,
                )
                error = prediction - truth
                report[name][scenario.name] = {
                    "standardized_rmse": float(torch.sqrt(error.square().mean()).cpu()),
                    "structure_rmse": float(
                        torch.sqrt(error[..., :8].square().mean()).cpu()
                    ),
                    "step_rmse": float(
                        torch.sqrt(error[..., 8:].square().mean()).cpu()
                    ),
                    "finite": bool(torch.isfinite(prediction).all().cpu()),
                }
    for method in report.values():
        method["macro_scenario_rmse"] = float(
            np.mean([method[name]["standardized_rmse"] for name in ("T1", "T2", "T3")])
        )
    return report


def aggregate_and_decide(seed_results):
    aggregate = {}
    for method in ("one_pass", "alternating"):
        aggregate[method] = {}
        for metric in ("macro_scenario_rmse",):
            values = [row["validation"][method][metric] for row in seed_results]
            aggregate[method][metric] = {
                "mean": float(np.mean(values)),
                "sample_std": float(np.std(values, ddof=1)),
                "values": values,
            }
        aggregate[method]["scenarios"] = {}
        for scenario in ("T1", "T2", "T3"):
            values = [
                row["validation"][method][scenario]["standardized_rmse"]
                for row in seed_results
            ]
            aggregate[method]["scenarios"][scenario] = {
                "mean": float(np.mean(values)),
                "sample_std": float(np.std(values, ddof=1)),
                "values": values,
            }
    control = aggregate["one_pass"]
    candidate = aggregate["alternating"]
    paired = [
        row["validation"]["alternating"]["macro_scenario_rmse"]
        - row["validation"]["one_pass"]["macro_scenario_rmse"]
        for row in seed_results
    ]
    t1_regression = (
        candidate["scenarios"]["T1"]["mean"]
        / control["scenarios"]["T1"]["mean"] - 1.0
    )
    t3_improvement = 1.0 - (
        candidate["scenarios"]["T3"]["mean"]
        / control["scenarios"]["T3"]["mean"]
    )
    decision = {
        "paired_macro_difference": {
            "mean": float(np.mean(paired)),
            "sample_std": float(np.std(paired, ddof=1)),
            "values": paired,
        },
        "candidate_seed_wins": int(sum(value < 0 for value in paired)),
        "T1_relative_regression": float(t1_regression),
        "T3_relative_improvement": float(t3_improvement),
        "all_finite": all(
            row["validation"][method][scenario]["finite"]
            and row["validation"][method][scenario]["standardized_rmse"] < 10.0
            for row in seed_results
            for method in ("one_pass", "alternating")
            for scenario in ("T1", "T2", "T3")
        ),
    }
    decision["passed"] = bool(
        candidate["macro_scenario_rmse"]["mean"]
        < control["macro_scenario_rmse"]["mean"]
        and decision["candidate_seed_wins"] >= 2
        and t3_improvement >= 0.05
        and t1_regression <= 0.02
        and decision["all_finite"]
    )
    return aggregate, decision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--train-split", type=Path, required=True)
    parser.add_argument("--validation-split", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    settings = {
        "maximum_horizon": config["model"]["block_horizon"],
        "interpolator_epochs": config["training"]["interpolator_epochs"],
        "forecaster_epochs": config["training"]["forecaster_epochs"],
        "windows_per_trajectory_per_epoch": config["training"]["windows_per_trajectory_per_epoch"],
        "batch_size": config["training"]["batch_size"],
        "learning_rate": config["training"]["learning_rate"],
        "gradient_clip_norm": config["training"]["gradient_clip_norm"],
    }
    train_ids = _read_split(args.train_split)
    validation_ids = _read_split(args.validation_split)
    train_raw = _load_features(args.h5, train_ids)
    validation_raw = _load_features(args.h5, validation_ids)
    mean, scale = _normalization(train_raw)
    train = (train_raw - mean) / scale
    validation = (validation_raw - mean) / scale
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    results = []
    for seed in config["training"]["seeds"]:
        _seed_everything(seed, device)
        generator = torch.Generator().manual_seed(seed + 90125)
        interpolator = BridgeInterpolator(
            train.shape[-1], config["model"]["hidden_dim"],
            config["model"]["layers"], settings["maximum_horizon"],
        ).to(device)
        forecaster = EndpointForecaster(
            train.shape[-1], config["model"]["hidden_dim"],
            config["model"]["layers"], settings["maximum_horizon"],
        ).to(device)
        if max(parameter_count(interpolator), parameter_count(forecaster)) >= 10000:
            raise ValueError("network exceeds pre-registered parameter cap")
        interpolation_loss = _train_interpolator(
            interpolator, train, settings, device, generator
        )
        forecast_loss = _train_forecaster(
            interpolator, forecaster, train, settings, device, generator
        )
        validation_report = _evaluate(
            interpolator, forecaster, validation, device, settings["maximum_horizon"]
        )
        row = {
            "seed": seed,
            "interpolator_parameter_count": parameter_count(interpolator),
            "forecaster_parameter_count": parameter_count(forecaster),
            "interpolator_final_train_mse": interpolation_loss[-1],
            "forecaster_final_train_mse": forecast_loss[-1],
            "validation": validation_report,
        }
        results.append(row)
        print(json.dumps(row), flush=True)
    aggregate, decision = aggregate_and_decide(results)
    report = {
        "status": "ProAR-inspired deterministic mechanism proxy; not full ProAR and not official competition scores",
        "protocol": {
            "preregistration_commit": "4556a31",
            "train_ids": train_ids,
            "validation_ids": validation_ids,
            "test_accessed": False,
            "feature_names": FEATURE_NAMES,
            "normalization_mean": mean.tolist(),
            "normalization_scale": scale.tolist(),
            "device": str(device),
            "config": config,
        },
        "results": results,
        "aggregate": aggregate,
        "promotion": decision,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"aggregate": aggregate, "promotion": decision}, indent=2))


if __name__ == "__main__":
    main()
