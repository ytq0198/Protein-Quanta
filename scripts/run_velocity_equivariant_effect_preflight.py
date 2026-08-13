"""Compare position-only and velocity-aware frame-time dynamics on a new split."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

from NeuralMD.datasets.MISATO import DatasetMISATOSemiFlexibleMultiTrajectory
from protein_quanta.dense_equivariant_dynamics import DenseEquivariantAcceleration
from protein_quanta.velocity_equivariant_dynamics import VelocityEquivariantAcceleration
from scripts.run_frame_time_learnability_preflight import evaluate, train
from scripts.train_evaluate_dense_paired_phys import (
    make_schedule,
    prepare_single_complex,
    read_ids,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--parent-config", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:3")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    parent = json.loads(args.parent_config.read_text(encoding="utf-8"))
    train_ids = config["data"]["train_ids"]
    diagnostic_ids = config["data"]["diagnostic_ids"]
    if len(train_ids) != 36 or len(diagnostic_ids) != 12:
        raise ValueError("effect preflight requires frozen 36/12 split")
    if set(train_ids).isdisjoint(diagnostic_ids) is False:
        raise ValueError("training and diagnostic IDs overlap")
    if set(train_ids) | set(diagnostic_ids) != set(parent["data"]["train_ids"]):
        raise ValueError("new split is not exactly the prior 48-complex training set")

    official_ids = read_ids(args.data_root / "raw" / "train_MD.txt")
    index = {identifier: position for position, identifier in enumerate(official_ids)}
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
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    control = DenseEquivariantAcceleration(hidden_dim=32).to(device)
    torch.manual_seed(seed)
    candidate = VelocityEquivariantAcceleration(hidden_dim=32).to(device)
    models = {"control": control, "candidate": candidate}
    protocol = config["protocol"]
    schedule = make_schedule(len(training_samples), settings["epochs"], seed)
    initial_metrics = {
        name: evaluate(model, diagnostic_samples, protocol)
        for name, model in models.items()
    }
    training = {
        name: train(model, training_samples, schedule, protocol, settings)
        for name, model in models.items()
    }
    final_metrics = {
        name: evaluate(model, diagnostic_samples, protocol)
        for name, model in models.items()
    }
    thresholds = config["gate"]
    control_t3 = final_metrics["control"]["T3"]
    candidate_t3 = final_metrics["candidate"]["T3"]
    candidate_t1 = final_metrics["candidate"]["T1"]
    control_t1 = final_metrics["control"]["T1"]
    candidate_clipping_rate = (
        training["candidate"]["clipped"] / training["candidate"]["updates"]
    )
    checks = {
        "all_updates_and_rollouts_finite": (
            all(row["nonfinite"] == 0 for row in training.values())
            and all(
                metrics[scenario]["nonfinite_frame_fraction"] == 0
                for metrics in final_metrics.values()
                for scenario in ("T1", "T2", "T3")
            )
        ),
        "candidate_T3_rmse_improves": (
            (control_t3["coordinate_rmse_angstrom"]
             - candidate_t3["coordinate_rmse_angstrom"])
            / control_t3["coordinate_rmse_angstrom"]
            >= thresholds["candidate_T3_rmse_min_improvement_over_control_fraction"]
        ),
        "candidate_T1_worsening_bounded": (
            (candidate_t1["coordinate_rmse_angstrom"]
             - control_t1["coordinate_rmse_angstrom"])
            / control_t1["coordinate_rmse_angstrom"]
            <= thresholds["candidate_T1_rmse_max_worsening_fraction"]
        ),
        "candidate_T3_amplitude_in_range": (
            thresholds["candidate_T3_step_amplitude_ratio_min"]
            <= candidate_t3["step_amplitude_ratio"]
            <= thresholds["candidate_T3_step_amplitude_ratio_max"]
        ),
        "candidate_clipping_rate_bounded": (
            candidate_clipping_rate <= thresholds["candidate_clipping_rate_max"]
        ),
    }
    passed = all(checks.values())
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, model in models.items():
        path = args.checkpoint_dir / f"{name}_final.pth"
        torch.save({"model": model.state_dict(), "config": config}, path)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = {
        "status": "velocity_effect_gate_pass" if passed else "velocity_effect_gate_fail",
        "scope": "36 train / 12 unseen diagnostic inside prior training set; all previous diagnostics/holdouts and official validation/test untouched",
        "parameter_count": {
            name: sum(parameter.numel() for parameter in model.parameters())
            for name, model in models.items()
        },
        "initial_metrics": initial_metrics,
        "training": training,
        "final_metrics": final_metrics,
        "candidate_clipping_rate": candidate_clipping_rate,
        "checks": checks,
        "checkpoint_sha256": hashes,
        "decision": thresholds["decision"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        key: report[key] for key in (
            "status", "parameter_count", "final_metrics",
            "candidate_clipping_rate", "checks",
        )
    }, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
