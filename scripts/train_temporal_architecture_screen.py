"""Train parameter-capped temporal cores on invariant MISATO trajectory features."""

import argparse
import json
import random
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import h5py
import numpy as np
import torch
from torch import nn

from protein_quanta.misato import load_ligand_trajectory
from protein_quanta.scenarios import competition_scenarios
from protein_quanta.temporal_features import FEATURE_NAMES, invariant_trajectory_features
from protein_quanta.temporal_models import TemporalFeatureForecaster, parameter_count


def _read_split(path):
    values = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError("split must contain unique sample IDs")
    return values


def _load_features(h5_path, sample_ids):
    features = []
    with h5py.File(h5_path, "r") as handle:
        for sample_id in sample_ids:
            trajectory = load_ligand_trajectory(handle[sample_id]).coordinates
            features.append(invariant_trajectory_features(trajectory))
    values = np.stack(features)
    if values.shape[1] != 100:
        raise ValueError("architecture screen requires 100-frame trajectories")
    return values


def _normalization(train):
    mean = train.reshape(-1, train.shape[-1]).mean(axis=0)
    scale = train.reshape(-1, train.shape[-1]).std(axis=0)
    scale = np.maximum(scale, 1e-6)
    return mean, scale


def _scenario_metrics(model, normalized, device, static_step_values):
    model.eval()
    summary = {}
    with torch.no_grad():
        values = torch.as_tensor(normalized, dtype=torch.float32, device=device)
        for scenario in competition_scenarios():
            observed = values[:, scenario.observed_start : scenario.observed_end + 1]
            truth = values[:, scenario.target_start : scenario.target_end + 1]
            prediction = model.rollout(observed, truth.shape[1])
            error = prediction - truth
            static = observed[:, -1:, :].expand_as(truth).clone()
            static[..., 8:] = torch.as_tensor(
                static_step_values, dtype=static.dtype, device=device
            )
            static_error = static - truth
            summary[scenario.name] = {
                "standardized_rmse": float(torch.sqrt(torch.mean(error.square())).cpu()),
                "structure_rmse": float(
                    torch.sqrt(torch.mean(error[..., :8].square())).cpu()
                ),
                "step_rmse": float(
                    torch.sqrt(torch.mean(error[..., 8:].square())).cpu()
                ),
                "static_standardized_rmse": float(
                    torch.sqrt(torch.mean(static_error.square())).cpu()
                ),
                "finite": bool(torch.isfinite(prediction).all().cpu()),
            }
    return summary


def _scenario_macro_rmse(summary):
    """Equal-weight proxy used until the official T1/T2/T3 weights are released."""
    names = ("T1", "T2", "T3")
    return float(np.mean([summary[name]["standardized_rmse"] for name in names]))


def _weighted_rmse(summary):
    """Backward-compatible alias; no competition weights are currently public."""
    return _scenario_macro_rmse(summary)


def train_architecture(
    architecture,
    train,
    validation,
    config,
    epochs,
    batch_size,
    learning_rate,
    evaluation_interval,
    seed,
    device,
    static_step_values,
    checkpoint_policy="best_validation",
    input_noise_std=0.0,
):
    if checkpoint_policy not in ("best_validation", "final_epoch"):
        raise ValueError("unknown checkpoint policy")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model_config = config["architectures"][architecture]
    model = TemporalFeatureForecaster(
        feature_dim=train.shape[-1],
        architecture=architecture,
        hidden_dim=model_config["hidden_dim"],
        layers=model_config["layers"],
        maximum_length=128,
        transformer_heads=model_config.get("heads", 4),
    ).to(device)
    count = parameter_count(model)
    if count >= 10000:
        raise ValueError(f"{architecture} exceeds the pre-registered parameter cap")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    generator = torch.Generator().manual_seed(seed)
    noise_generator = torch.Generator().manual_seed(seed + 1_000_003)
    train_tensor = torch.as_tensor(train, dtype=torch.float32)
    history = []
    best = None
    best_state = None
    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(train_tensor.shape[0], generator=generator)
        model.train()
        losses = []
        for start in range(0, permutation.numel(), batch_size):
            batch = train_tensor[permutation[start : start + batch_size]].to(device)
            model_input = batch[:, :-1]
            if input_noise_std > 0:
                noise = torch.randn(
                    model_input.shape,
                    generator=noise_generator,
                    dtype=model_input.dtype,
                ).to(device)
                model_input = model_input + input_noise_std * noise
            prediction, _ = model(model_input)
            loss = criterion(prediction, batch[:, 1:])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if epoch % evaluation_interval == 0 or epoch == epochs:
            summary = _scenario_metrics(
                model, validation, device, static_step_values
            )
            macro = _scenario_macro_rmse(summary)
            record = {
                "epoch": epoch,
                "train_mse": float(np.mean(losses)),
                "macro_scenario_rmse": macro,
                "scenarios": summary,
            }
            history.append(record)
            if best is None or macro < best["macro_scenario_rmse"]:
                best = record
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
            print(architecture, json.dumps(record), flush=True)
    exploratory_best = best
    if checkpoint_policy == "best_validation":
        model.load_state_dict(best_state)
        selected = exploratory_best
    else:
        selected = history[-1]
    return {
        "architecture": architecture,
        "seed": seed,
        "parameter_count": count,
        "model_config": model_config,
        "checkpoint_policy": checkpoint_policy,
        "input_noise_std": float(input_noise_std),
        "best": selected,
        "exploratory_best": exploratory_best,
        "history": history,
    }


def _promotion(results):
    by_name = {result["architecture"]: result for result in results}
    mlp = by_name["mlp"]["best"]
    decisions = {}
    for name, result in by_name.items():
        best = result["best"]
        beats_mlp = best["macro_scenario_rmse"] < mlp["macro_scenario_rmse"]
        beats_static_count = sum(
            row["standardized_rmse"] < row["static_standardized_rmse"]
            for row in best["scenarios"].values()
        )
        long_improvement = max(
            1.0
            - best["scenarios"][scenario]["standardized_rmse"]
            / mlp["scenarios"][scenario]["standardized_rmse"]
            for scenario in ("T2", "T3")
        )
        robust = all(
            row["finite"] and row["standardized_rmse"] < 10.0
            for row in best["scenarios"].values()
        )
        decisions[name] = {
            "beats_mlp_mean_macro": beats_mlp,
            "beats_static_scenario_count": beats_static_count,
            "best_T2_T3_improvement_over_mlp": float(long_improvement),
            "robust": robust,
            "passed": bool(
                name != "mlp"
                and beats_mlp
                and beats_static_count >= 2
                and long_improvement >= 0.05
                and robust
            ),
        }
    return decisions


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
    train_ids = _read_split(args.train_split)
    validation_ids = _read_split(args.validation_split)
    train = _load_features(args.h5, train_ids)
    validation = _load_features(args.h5, validation_ids)
    mean, scale = _normalization(train)
    train = (train - mean) / scale
    validation = (validation - mean) / scale
    static_step_values = -mean[8:] / scale[8:]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    settings = config["training"]
    results = [
        train_architecture(
            architecture,
            train,
            validation,
            config,
            epochs=settings["epochs"],
            batch_size=settings["batch_size"],
            learning_rate=settings["learning_rate"],
            evaluation_interval=config["selection"]["evaluation_interval_epochs"],
            seed=settings["seed"],
            device=device,
            static_step_values=static_step_values,
        )
        for architecture in config["architectures"]
    ]
    report = {
        "status": "invariant temporal architecture screen; not competition Geo/Phys/Dyn/Stab scores",
        "protocol": {
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
        "promotion": _promotion(results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"promotion": report["promotion"]}, indent=2))


if __name__ == "__main__":
    main()
