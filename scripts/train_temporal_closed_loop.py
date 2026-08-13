"""Run the pre-registered differentiable closed-loop temporal pilot."""

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
)


def differentiable_rollout(model, observed, horizon):
    """Autoregress without detaching so future errors train through earlier predictions."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    generated = observed
    outputs = []
    for _ in range(horizon):
        prediction, _ = model(generated)
        current = prediction[:, -1:]
        outputs.append(current)
        generated = torch.cat((generated, current), dim=1)
    return torch.cat(outputs, dim=1)


def _mean_std(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "sample_std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "values": values.tolist(),
    }


def _macro(record):
    return _scenario_macro_rmse(record["scenarios"])


def aggregate_and_decide(results, reference_report):
    references = {
        row["seed"]: row["best"]
        for row in reference_report["results"]
        if row["architecture"] == "transformer"
    }
    rows = sorted(results, key=lambda row: row["seed"])
    if {row["seed"] for row in rows} != set(references):
        raise ValueError("closed-loop and reference seeds do not match")
    candidate_macro = [_macro(row["final"]) for row in rows]
    reference_macro = [_macro(references[row["seed"]]) for row in rows]
    paired = [new - old for new, old in zip(candidate_macro, reference_macro)]
    scenarios = {}
    for scenario in ("T1", "T2", "T3"):
        new = [row["final"]["scenarios"][scenario]["standardized_rmse"] for row in rows]
        old = [references[row["seed"]]["scenarios"][scenario]["standardized_rmse"] for row in rows]
        scenarios[scenario] = {
            "closed_loop": _mean_std(new),
            "one_step": _mean_std(old),
            "paired_difference": _mean_std([a - b for a, b in zip(new, old)]),
            "relative_mean_change": float(np.mean(new) / np.mean(old) - 1.0),
        }
    finite = bool(np.isfinite(candidate_macro).all())
    decision = {
        "beats_reference_mean": bool(np.mean(candidate_macro) < np.mean(reference_macro)),
        "beats_reference_seed_count": int(sum(value < 0 for value in paired)),
        "T1_relative_mean_change": scenarios["T1"]["relative_mean_change"],
        "T1_guard_passed": bool(scenarios["T1"]["relative_mean_change"] <= 0.02),
        "T3_relative_improvement": float(-scenarios["T3"]["relative_mean_change"]),
        "T3_mechanism_passed": bool(scenarios["T3"]["relative_mean_change"] <= -0.05),
        "finite": finite,
    }
    decision["passed"] = bool(
        decision["beats_reference_mean"]
        and decision["beats_reference_seed_count"] >= 2
        and decision["T1_guard_passed"]
        and decision["T3_mechanism_passed"]
        and finite
    )
    return {
        "closed_loop_macro_scenario_rmse": _mean_std(candidate_macro),
        "one_step_macro_scenario_rmse": _mean_std(reference_macro),
        "paired_macro_difference": _mean_std(paired),
        "scenarios": scenarios,
    }, decision


def train_seed(train, validation, architecture_config, settings, seed, device, static_step_values):
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
    prefix_generator = torch.Generator().manual_seed(seed + 2_000_003)
    history = []
    horizon = settings["closed_loop_horizon"]
    for epoch in range(1, settings["epochs"] + 1):
        permutation = torch.randperm(train_tensor.shape[0], generator=permutation_generator)
        model.train()
        one_step_losses = []
        closed_loop_losses = []
        total_losses = []
        for start in range(0, permutation.numel(), settings["batch_size"]):
            batch = train_tensor[permutation[start : start + settings["batch_size"]]].to(device)
            one_step_prediction, _ = model(batch[:, :-1])
            one_step_loss = criterion(one_step_prediction, batch[:, 1:])
            prefix_length = int(
                torch.randint(
                    10,
                    batch.shape[1] - horizon + 1,
                    (1,),
                    generator=prefix_generator,
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
        if epoch % 20 == 0 or epoch == settings["epochs"]:
            scenarios = _scenario_metrics(model, validation, device, static_step_values)
            record = {
                "epoch": epoch,
                "one_step_train_mse": float(np.mean(one_step_losses)),
                "closed_loop_train_mse": float(np.mean(closed_loop_losses)),
                "total_train_loss": float(np.mean(total_losses)),
                "macro_scenario_rmse": _scenario_macro_rmse(scenarios),
                "scenarios": scenarios,
            }
            history.append(record)
            print("closed_loop", seed, json.dumps(record), flush=True)
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
    parser.add_argument("--train-split", type=Path, required=True)
    parser.add_argument("--validation-split", type=Path, required=True)
    parser.add_argument("--architecture-config", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    architecture = json.loads(args.architecture_config.read_text(encoding="utf-8"))
    preregistration = json.loads(args.preregistration.read_text(encoding="utf-8"))
    reference = json.loads(args.reference_report.read_text(encoding="utf-8"))
    train_ids = _read_split(args.train_split)
    validation_ids = _read_split(args.validation_split)
    if train_ids != reference["protocol"]["train_ids"] or validation_ids != reference["protocol"]["validation_ids"]:
        raise ValueError("reference report uses different sample IDs")
    train_raw = _load_features(args.h5, train_ids)
    validation_raw = _load_features(args.h5, validation_ids)
    mean, scale = _normalization(train_raw)
    train = (train_raw - mean) / scale
    validation = (validation_raw - mean) / scale
    static_step_values = -mean[8:] / scale[8:]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    settings = preregistration["training"]
    results = [
        train_seed(
            train,
            validation,
            architecture["architectures"]["transformer"],
            settings,
            seed,
            device,
            static_step_values,
        )
        for seed in settings["seeds"]
    ]
    aggregate, decision = aggregate_and_decide(results, reference)
    report = {
        "status": "pre-registered differentiable closed-loop feasibility; not official competition scores",
        "protocol": {
            "train_ids": train_ids,
            "validation_ids": validation_ids,
            "test_accessed": False,
            "feature_names": FEATURE_NAMES,
            "normalization_mean": mean.tolist(),
            "normalization_scale": scale.tolist(),
            "device": str(device),
            "preregistration": preregistration,
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
