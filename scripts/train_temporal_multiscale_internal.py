"""Train paired one-step and multi-scale closed-loop models on a train-only split."""

import argparse
import json
from pathlib import Path
import random
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch import nn

from protein_quanta.temporal_features import FEATURE_NAMES
from protein_quanta.temporal_models import TemporalFeatureForecaster, parameter_count
from scripts.train_temporal_architecture_screen import (
    _load_features,
    _normalization,
    _read_split,
    _scenario_metrics,
    _scenario_macro_rmse,
    train_architecture,
)
from scripts.train_temporal_closed_loop import aggregate_and_decide, differentiable_rollout


def train_multiscale_seed(
    train,
    holdout,
    architecture_config,
    settings,
    seed,
    device,
    static_step_values,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = TemporalFeatureForecaster(
        feature_dim=train.shape[-1],
        architecture="transformer",
        hidden_dim=architecture_config["hidden_dim"],
        layers=architecture_config["layers"],
        maximum_length=128,
        transformer_heads=architecture_config["heads"],
    ).to(device)
    count = parameter_count(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    criterion = nn.MSELoss()
    train_tensor = torch.as_tensor(train, dtype=torch.float32)
    permutation_generator = torch.Generator().manual_seed(seed)
    sampling_generator = torch.Generator().manual_seed(seed + 3_000_003)
    horizons = tuple(settings["closed_loop_horizons"])
    history = []
    for epoch in range(1, settings["epochs"] + 1):
        permutation = torch.randperm(train_tensor.shape[0], generator=permutation_generator)
        model.train()
        one_step_losses = []
        closed_loop_losses = []
        total_losses = []
        sampled_horizons = []
        for start in range(0, permutation.numel(), settings["batch_size"]):
            batch = train_tensor[permutation[start : start + settings["batch_size"]]].to(device)
            one_step_prediction, _ = model(batch[:, :-1])
            one_step_loss = criterion(one_step_prediction, batch[:, 1:])
            horizon_index = int(
                torch.randint(0, len(horizons), (1,), generator=sampling_generator).item()
            )
            horizon = horizons[horizon_index]
            prefix_length = int(
                torch.randint(
                    10,
                    batch.shape[1] - horizon + 1,
                    (1,),
                    generator=sampling_generator,
                ).item()
            )
            rollout = differentiable_rollout(model, batch[:, :prefix_length], horizon)
            closed_loop_loss = criterion(
                rollout, batch[:, prefix_length : prefix_length + horizon]
            )
            loss = (
                settings["one_step_loss_weight"] * one_step_loss
                + settings["closed_loop_loss_weight"] * closed_loop_loss
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip_norm"])
            optimizer.step()
            one_step_losses.append(float(one_step_loss.detach().cpu()))
            closed_loop_losses.append(float(closed_loop_loss.detach().cpu()))
            total_losses.append(float(loss.detach().cpu()))
            sampled_horizons.append(horizon)
        if epoch % 20 == 0 or epoch == settings["epochs"]:
            scenarios = _scenario_metrics(model, holdout, device, static_step_values)
            record = {
                "epoch": epoch,
                "one_step_train_mse": float(np.mean(one_step_losses)),
                "closed_loop_train_mse": float(np.mean(closed_loop_losses)),
                "total_train_loss": float(np.mean(total_losses)),
                "sampled_horizon_mean": float(np.mean(sampled_horizons)),
                "macro_scenario_rmse": _scenario_macro_rmse(scenarios),
                "scenarios": scenarios,
            }
            history.append(record)
            print("multiscale", seed, json.dumps(record), flush=True)
    return {
        "seed": seed,
        "parameter_count": count,
        "checkpoint_policy": "final_epoch",
        "history": history,
        "final": history[-1],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--development-split", type=Path, required=True)
    parser.add_argument("--holdout-split", type=Path, required=True)
    parser.add_argument("--architecture-config", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    architecture = json.loads(args.architecture_config.read_text(encoding="utf-8"))
    preregistration = json.loads(args.preregistration.read_text(encoding="utf-8"))
    development_ids = _read_split(args.development_split)
    holdout_ids = _read_split(args.holdout_split)
    if set(development_ids) & set(holdout_ids):
        raise ValueError("development and holdout IDs overlap")
    if len(development_ids) != preregistration["split"]["development_count"]:
        raise ValueError("development count does not match preregistration")
    if len(holdout_ids) != preregistration["split"]["holdout_count"]:
        raise ValueError("holdout count does not match preregistration")
    development_raw = _load_features(args.h5, development_ids)
    holdout_raw = _load_features(args.h5, holdout_ids)
    mean, scale = _normalization(development_raw)
    development = (development_raw - mean) / scale
    holdout = (holdout_raw - mean) / scale
    static_step_values = -mean[8:] / scale[8:]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    settings = preregistration["training"]
    baseline_results = [
        train_architecture(
            "transformer",
            development,
            holdout,
            architecture,
            epochs=settings["epochs"],
            batch_size=settings["batch_size"],
            learning_rate=settings["learning_rate"],
            evaluation_interval=20,
            seed=seed,
            device=device,
            static_step_values=static_step_values,
            checkpoint_policy="final_epoch",
        )
        for seed in settings["seeds"]
    ]
    candidate_results = [
        train_multiscale_seed(
            development,
            holdout,
            architecture["architectures"]["transformer"],
            settings,
            seed,
            device,
            static_step_values,
        )
        for seed in settings["seeds"]
    ]
    aggregate, decision = aggregate_and_decide(
        candidate_results, {"results": baseline_results}
    )
    report = {
        "status": "train-only multi-scale closed-loop feasibility; not official competition scores",
        "protocol": {
            "development_ids": development_ids,
            "holdout_ids": holdout_ids,
            "official_validation_accessed": False,
            "official_test_accessed": False,
            "feature_names": FEATURE_NAMES,
            "normalization_mean": mean.tolist(),
            "normalization_scale": scale.tolist(),
            "device": str(device),
            "preregistration": preregistration,
        },
        "baseline_results": baseline_results,
        "candidate_results": candidate_results,
        "aggregate": aggregate,
        "promotion": decision,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"aggregate": aggregate, "promotion": decision}, indent=2))


if __name__ == "__main__":
    main()
