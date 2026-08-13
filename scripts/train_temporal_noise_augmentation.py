"""Run the pre-registered train-state noise augmentation pilot."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from protein_quanta.temporal_features import FEATURE_NAMES
from scripts.train_temporal_architecture_screen import (
    _load_features,
    _normalization,
    _read_split,
    train_architecture,
)


def _mean_std(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "sample_std": float(values.std(ddof=1)),
        "values": values.tolist(),
    }


def _macro(row):
    return float(
        np.mean(
            [row["scenarios"][name]["standardized_rmse"] for name in ("T1", "T2", "T3")]
        )
    )


def aggregate_noise_pilot(noise_results, reference_report):
    references = {
        row["seed"]: row["best"]
        for row in reference_report["results"]
        if row["architecture"] == "transformer"
    }
    rows = sorted(noise_results, key=lambda row: row["seed"])
    if {row["seed"] for row in rows} != set(references):
        raise ValueError("noise and reference seeds do not match")
    augmented = [_macro(row["best"]) for row in rows]
    baseline = [_macro(references[row["seed"]]) for row in rows]
    paired = [new - old for new, old in zip(augmented, baseline)]
    scenarios = {}
    for scenario in ("T1", "T2", "T3"):
        new_values = [row["best"]["scenarios"][scenario]["standardized_rmse"] for row in rows]
        old_values = [references[row["seed"]]["scenarios"][scenario]["standardized_rmse"] for row in rows]
        scenarios[scenario] = {
            "noise": _mean_std(new_values),
            "no_noise": _mean_std(old_values),
            "paired_difference": _mean_std(
                [new - old for new, old in zip(new_values, old_values)]
            ),
            "relative_mean_change": float(np.mean(new_values) / np.mean(old_values) - 1.0),
        }
    finite = bool(np.isfinite(augmented).all())
    decision = {
        "beats_reference_mean": bool(np.mean(augmented) < np.mean(baseline)),
        "beats_reference_seed_count": int(sum(value < 0 for value in paired)),
        "T1_relative_mean_change": scenarios["T1"]["relative_mean_change"],
        "T1_guard_passed": bool(scenarios["T1"]["relative_mean_change"] <= 0.02),
        "finite": finite,
    }
    decision["passed"] = bool(
        decision["beats_reference_mean"]
        and decision["beats_reference_seed_count"] >= 2
        and decision["T1_guard_passed"]
        and finite
    )
    return {
        "noise_macro_scenario_rmse": _mean_std(augmented),
        "no_noise_macro_scenario_rmse": _mean_std(baseline),
        "paired_macro_difference": _mean_std(paired),
        "scenarios": scenarios,
    }, decision


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

    config = json.loads(args.architecture_config.read_text(encoding="utf-8"))
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
        train_architecture(
            "transformer",
            train,
            validation,
            config,
            epochs=settings["epochs"],
            batch_size=settings["batch_size"],
            learning_rate=settings["learning_rate"],
            evaluation_interval=config["evaluation_interval_epochs"],
            seed=seed,
            device=device,
            static_step_values=static_step_values,
            checkpoint_policy="final_epoch",
            input_noise_std=settings["input_noise_std"],
        )
        for seed in settings["seeds"]
    ]
    aggregate, decision = aggregate_noise_pilot(results, reference)
    report = {
        "status": "pre-registered train-state noise feasibility; not official competition scores",
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
