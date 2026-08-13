"""Run a fixed-epoch multi-seed temporal comparison including RoPE."""

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
        "sample_std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "values": values.tolist(),
    }


def _macro(row):
    return float(
        np.mean(
            [row["scenarios"][name]["standardized_rmse"] for name in ("T1", "T2", "T3")]
        )
    )


def aggregate_results(results):
    grouped = {}
    for result in results:
        grouped.setdefault(result["architecture"], []).append(result)
    mlp_by_seed = {row["seed"]: row["best"] for row in grouped["mlp"]}
    aggregate = {}
    for architecture, rows in grouped.items():
        rows = sorted(rows, key=lambda row: row["seed"])
        macros = [_macro(row["best"]) for row in rows]
        paired = [
            _macro(row["best"])
            - _macro(mlp_by_seed[row["seed"]])
            for row in rows
        ]
        scenarios = {}
        for scenario in ("T1", "T2", "T3"):
            scenarios[scenario] = {
                "standardized_rmse": _mean_std(
                    [row["best"]["scenarios"][scenario]["standardized_rmse"] for row in rows]
                ),
                "structure_rmse": _mean_std(
                    [row["best"]["scenarios"][scenario]["structure_rmse"] for row in rows]
                ),
                "step_rmse": _mean_std(
                    [row["best"]["scenarios"][scenario]["step_rmse"] for row in rows]
                ),
                "static_standardized_rmse": _mean_std(
                    [row["best"]["scenarios"][scenario]["static_standardized_rmse"] for row in rows]
                ),
            }
        aggregate[architecture] = {
            "parameter_count": rows[0]["parameter_count"],
            "macro_scenario_rmse": _mean_std(macros),
            "paired_macro_difference_vs_mlp": _mean_std(paired),
            "beats_mlp_seed_count": int(sum(value < 0 for value in paired)),
            "scenarios": scenarios,
        }

    mlp = aggregate["mlp"]
    decisions = {}
    for architecture, row in aggregate.items():
        beats_static_count = sum(
            row["scenarios"][scenario]["standardized_rmse"]["mean"]
            < row["scenarios"][scenario]["static_standardized_rmse"]["mean"]
            for scenario in ("T1", "T2", "T3")
        )
        long_improvement = max(
            1.0
            - row["scenarios"][scenario]["standardized_rmse"]["mean"]
            / mlp["scenarios"][scenario]["standardized_rmse"]["mean"]
            for scenario in ("T2", "T3")
        )
        finite = all(
            np.isfinite(value)
            for value in row["macro_scenario_rmse"]["values"]
        )
        decisions[architecture] = {
            "beats_mlp_mean_macro": row["macro_scenario_rmse"]["mean"]
            < mlp["macro_scenario_rmse"]["mean"],
            "beats_mlp_seed_count": row["beats_mlp_seed_count"],
            "beats_static_scenario_count": int(beats_static_count),
            "best_T2_T3_mean_improvement_over_mlp": float(long_improvement),
            "finite": bool(finite),
            "passed": bool(
                architecture != "mlp"
                and row["macro_scenario_rmse"]["mean"]
                < mlp["macro_scenario_rmse"]["mean"]
                and row["beats_mlp_seed_count"] >= 2
                and beats_static_count >= 2
                and long_improvement >= 0.05
                and finite
            ),
        }
    return aggregate, decisions


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
    train_raw = _load_features(args.h5, train_ids)
    validation_raw = _load_features(args.h5, validation_ids)
    mean, scale = _normalization(train_raw)
    train = (train_raw - mean) / scale
    validation = (validation_raw - mean) / scale
    static_step_values = -mean[8:] / scale[8:]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    settings = config["training"]
    results = []
    for seed in settings["seeds"]:
        for architecture in config["architectures"]:
            results.append(
                train_architecture(
                    architecture,
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
                )
            )
    aggregate, decisions = aggregate_results(results)
    report = {
        "status": "fixed-epoch multi-seed invariant temporal screen; not official competition scores",
        "protocol": {
            "train_ids": train_ids,
            "validation_ids": validation_ids,
            "test_accessed": False,
            "feature_names": FEATURE_NAMES,
            "normalization_mean": mean.tolist(),
            "normalization_scale": scale.tolist(),
            "physical_static_step_standardized": static_step_values.tolist(),
            "device": str(device),
            "config": config,
        },
        "results": results,
        "aggregate": aggregate,
        "promotion": decisions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"aggregate": aggregate, "promotion": decisions}, indent=2))


if __name__ == "__main__":
    main()
