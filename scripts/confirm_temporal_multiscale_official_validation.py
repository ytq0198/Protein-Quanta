"""One-time frozen official-validation confirmation for the train-only winner."""

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

from protein_quanta.temporal_models import TemporalFeatureForecaster
from scripts.train_temporal_architecture_screen import (
    _load_features,
    _normalization,
    _read_split,
    _scenario_metrics,
    _scenario_macro_rmse,
)
from scripts.train_temporal_closed_loop import differentiable_rollout


def train_frozen_model(train, model_config, settings, seed, device, multiscale):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = TemporalFeatureForecaster(
        feature_dim=train.shape[-1],
        architecture="transformer",
        hidden_dim=model_config["hidden_dim"],
        layers=model_config["layers"],
        maximum_length=128,
        transformer_heads=model_config["heads"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    criterion = nn.MSELoss()
    train_tensor = torch.as_tensor(train, dtype=torch.float32)
    permutation_generator = torch.Generator().manual_seed(seed)
    sampling_generator = torch.Generator().manual_seed(seed + 3_000_003)
    horizons = tuple(settings["closed_loop_horizons"])
    last_loss = None
    for _ in range(settings["epochs"]):
        permutation = torch.randperm(train_tensor.shape[0], generator=permutation_generator)
        model.train()
        losses = []
        for start in range(0, permutation.numel(), settings["batch_size"]):
            batch = train_tensor[permutation[start : start + settings["batch_size"]]].to(device)
            prediction, _ = model(batch[:, :-1])
            loss = criterion(prediction, batch[:, 1:])
            if multiscale:
                horizon = horizons[
                    int(torch.randint(0, len(horizons), (1,), generator=sampling_generator).item())
                ]
                prefix = int(
                    torch.randint(
                        10,
                        batch.shape[1] - horizon + 1,
                        (1,),
                        generator=sampling_generator,
                    ).item()
                )
                rollout = differentiable_rollout(model, batch[:, :prefix], horizon)
                loss = loss + settings["closed_loop_loss_weight"] * criterion(
                    rollout, batch[:, prefix : prefix + horizon]
                )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip_norm"])
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        last_loss = float(np.mean(losses))
    return model, last_loss


def summarize(candidate_rows, baseline_rows):
    scenarios = {}
    for name in ("T1", "T2", "T3"):
        candidate = np.array([row["scenarios"][name]["standardized_rmse"] for row in candidate_rows])
        baseline = np.array([row["scenarios"][name]["standardized_rmse"] for row in baseline_rows])
        scenarios[name] = {
            "candidate_mean": float(candidate.mean()),
            "baseline_mean": float(baseline.mean()),
            "relative_mean_change": float(candidate.mean() / baseline.mean() - 1.0),
            "paired_differences": (candidate - baseline).tolist(),
        }
    candidate_macro = np.array([row["macro_scenario_rmse"] for row in candidate_rows])
    baseline_macro = np.array([row["macro_scenario_rmse"] for row in baseline_rows])
    paired = candidate_macro - baseline_macro
    decision = {
        "beats_reference_mean": bool(candidate_macro.mean() < baseline_macro.mean()),
        "beats_reference_seed_count": int((paired < 0).sum()),
        "T1_guard_passed": bool(scenarios["T1"]["relative_mean_change"] <= 0.02),
        "T3_direction_passed": bool(scenarios["T3"]["relative_mean_change"] < 0),
        "T3_strong_marker_passed": bool(scenarios["T3"]["relative_mean_change"] <= -0.05),
        "finite": bool(np.isfinite(candidate_macro).all()),
    }
    decision["passed"] = bool(
        decision["beats_reference_mean"]
        and decision["beats_reference_seed_count"] >= 2
        and decision["T1_guard_passed"]
        and decision["T3_direction_passed"]
        and decision["finite"]
    )
    return {
        "candidate_macro_mean": float(candidate_macro.mean()),
        "baseline_macro_mean": float(baseline_macro.mean()),
        "candidate_macro_sample_std": float(candidate_macro.std(ddof=1)),
        "baseline_macro_sample_std": float(baseline_macro.std(ddof=1)),
        "paired_macro_differences": paired.tolist(),
        "scenarios": scenarios,
    }, decision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--development-split", type=Path, required=True)
    parser.add_argument("--official-validation-split", type=Path, required=True)
    parser.add_argument("--architecture-config", type=Path, required=True)
    parser.add_argument("--training-preregistration", type=Path, required=True)
    parser.add_argument("--confirmation-preregistration", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    architecture = json.loads(args.architecture_config.read_text(encoding="utf-8"))
    training = json.loads(args.training_preregistration.read_text(encoding="utf-8"))
    confirmation = json.loads(args.confirmation_preregistration.read_text(encoding="utf-8"))
    development_ids = _read_split(args.development_split)
    validation_ids = _read_split(args.official_validation_split)
    development_raw = _load_features(args.h5, development_ids)
    validation_raw = _load_features(args.h5, validation_ids)
    mean, scale = _normalization(development_raw)
    development = (development_raw - mean) / scale
    validation = (validation_raw - mean) / scale
    static_step_values = -mean[8:] / scale[8:]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    settings = training["training"]
    config = architecture["architectures"]["transformer"]
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    trained = []
    for seed in settings["seeds"]:
        for label, multiscale in (("baseline", False), ("multiscale", True)):
            model, final_train_loss = train_frozen_model(
                development, config, settings, seed, device, multiscale
            )
            checkpoint = args.checkpoint_dir / f"{label}_seed_{seed}.pth"
            torch.save(model.state_dict(), checkpoint)
            trained.append((label, seed, model, checkpoint, final_train_loss))

    # This is the single official-validation access point, after every final state exists.
    rows = {"baseline": [], "multiscale": []}
    for label, seed, model, checkpoint, final_train_loss in trained:
        scenario_metrics = _scenario_metrics(model, validation, device, static_step_values)
        rows[label].append(
            {
                "seed": seed,
                "checkpoint": checkpoint.as_posix(),
                "final_train_loss": final_train_loss,
                "macro_scenario_rmse": _scenario_macro_rmse(scenario_metrics),
                "scenarios": scenario_metrics,
            }
        )
    aggregate, decision = summarize(rows["multiscale"], rows["baseline"])
    report = {
        "status": "one-time frozen official-validation mechanism confirmation; not official competition scores",
        "protocol": {
            "development_ids": development_ids,
            "official_validation_ids": validation_ids,
            "official_validation_access_count": 1,
            "official_test_accessed": False,
            "normalization_source": "64-complex development split only",
            "training_preregistration": training,
            "confirmation_preregistration": confirmation,
            "device": str(device),
        },
        "baseline_results": rows["baseline"],
        "candidate_results": rows["multiscale"],
        "aggregate": aggregate,
        "confirmation": decision,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"aggregate": aggregate, "confirmation": decision}, indent=2))


if __name__ == "__main__":
    main()
