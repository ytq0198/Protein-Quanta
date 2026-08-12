"""Evaluate project-defined distributional Dyn diagnostics on saved rollouts."""

import argparse
import json
from pathlib import Path

import numpy as np

from protein_quanta.baselines import static_rollout
from protein_quanta.metrics import dynamics_distribution_metrics


def _parse_models(values):
    models = {}
    for token in values:
        if "=" not in token:
            raise ValueError("model must use NAME=TRAJECTORY_DIR syntax")
        name, directory = token.split("=", 1)
        if not name or not directory or name in models:
            raise ValueError(f"invalid or duplicate model specification: {token}")
        models[name] = Path(directory)
    if not models:
        raise ValueError("at least one model is required")
    return models


def _load_target(path):
    with np.load(path) as payload:
        prediction = np.asarray(payload["prediction"], dtype=float)
        truth = np.asarray(payload["truth"], dtype=float)
    if prediction.shape != truth.shape or prediction.ndim != 3:
        raise ValueError(f"{path}: incompatible trajectory arrays")
    if prediction.shape[0] < 5 or prediction.shape[-1] != 3:
        raise ValueError(f"{path}: expected at least five xyz frames")
    if not np.isfinite(prediction).all() or not np.isfinite(truth).all():
        raise ValueError(f"{path}: trajectories must be finite")
    return prediction[2:], truth[2:], truth[:2]


def _aggregate(rows, scenario_names, model_names):
    metric_names = tuple(
        name
        for name in rows[0]["scenarios"][scenario_names[0]][model_names[0]]
        if name != "velocity_autocorrelation_lags"
    )
    return {
        scenario: {
            model: {
                metric: float(
                    np.mean(
                        [row["scenarios"][scenario][model][metric] for row in rows]
                    )
                )
                for metric in metric_names
            }
            for model in model_names
        }
        for scenario in scenario_names
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-lag", type=int, default=10)
    args = parser.parse_args()

    reference = json.loads(args.reference_report.read_text(encoding="utf-8"))
    sample_ids = reference["protocol"]["sample_ids"]
    scenario_names = [row["name"] for row in reference["protocol"]["scenarios"]]
    model_dirs = _parse_models(args.model)
    model_names = list(model_dirs) + ["static", "truth"]
    rows = []
    for sample_id in sample_ids:
        scenarios = {}
        for scenario_name in scenario_names:
            trajectories = {}
            shared_truth = None
            observed = None
            for model_name, directory in model_dirs.items():
                prediction, truth, model_observed = _load_target(
                    directory / f"{sample_id}_{scenario_name}.npz"
                )
                if shared_truth is not None and not np.allclose(truth, shared_truth):
                    raise ValueError(f"{sample_id}/{scenario_name}: model truths disagree")
                shared_truth, observed = truth, model_observed
                trajectories[model_name] = prediction
            trajectories["static"] = static_rollout(observed, shared_truth.shape[0])
            trajectories["truth"] = shared_truth
            scenarios[scenario_name] = {
                model_name: dynamics_distribution_metrics(
                    trajectory,
                    shared_truth,
                    maximum_lag=args.maximum_lag,
                )
                for model_name, trajectory in trajectories.items()
            }
        rows.append({"sample_id": sample_id, "scenarios": scenarios})

    report = {
        "status": "project distributional Dyn diagnostics; not official Dyn score",
        "protocol": {
            "reference_report": str(args.reference_report.resolve()),
            "sample_ids": sample_ids,
            "sample_count": len(sample_ids),
            "scenarios": scenario_names,
            "models": {name: str(path.resolve()) for name, path in model_dirs.items()},
            "static_and_truth_controls": True,
            "maximum_velocity_autocorrelation_lag": args.maximum_lag,
            "aggregation": "unweighted mean of per-complex diagnostics",
        },
        "summary": _aggregate(rows, scenario_names, model_names),
        "samples": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
