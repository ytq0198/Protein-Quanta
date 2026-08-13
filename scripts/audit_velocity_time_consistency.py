"""Audit NeuralMD frame/ODE-time velocity units on the frozen development set."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torchdiffeq import odeint

from NeuralMD.datasets.MISATO import DatasetMISATOSemiFlexibleMultiTrajectory
from protein_quanta.metrics import coordinate_rmse, dynamics_distribution_metrics
from protein_quanta.neuralmd_multiscale import neuralmd_ode_rollout
from protein_quanta.scenarios import competition_scenarios
from scripts.train_evaluate_dense_paired_phys import prepare_single_complex, read_ids


class ZeroAcceleration(torch.nn.Module):
    """Analytic constant-velocity control with no learned parameters."""

    def forward(self, time, state, condition):
        velocity, position = state
        return torch.zeros_like(velocity), velocity


def evaluate_protocol(model, sample, velocity_scale):
    rows = {}
    for scenario in competition_scenarios():
        first, second = scenario.initializer_indices
        path = neuralmd_ode_rollout(
            model,
            odeint,
            sample,
            start=first,
            horizon=scenario.target_end - first,
            scaling=100,
            step_size=0.025,
            method="euler",
            initial_velocity_scale=velocity_scale,
        )[1]
        initializer_truth = sample.ligand_trajectory_pos[:, second, :]
        prediction = path[2:].detach().cpu().numpy()
        truth = sample.ligand_trajectory_pos[
            :, scenario.target_start : scenario.target_end + 1, :
        ].transpose(0, 1).cpu().numpy()
        rows[scenario.name] = {
            "initializer_frame_rmse_angstrom": float(
                torch.sqrt(torch.mean((path[1] - initializer_truth) ** 2)).cpu()
            ),
            "coordinate_rmse_angstrom": coordinate_rmse(prediction, truth),
            "step_amplitude_ratio": dynamics_distribution_metrics(
                prediction, truth, maximum_lag=10
            )["step_amplitude_ratio"],
        }
    return rows


def aggregate(samples):
    return {
        protocol: {
            scenario: {
                metric: float(np.mean([
                    row["protocols"][protocol][scenario][metric]
                    for row in samples
                ]))
                for metric in (
                    "initializer_frame_rmse_angstrom",
                    "coordinate_rmse_angstrom",
                    "step_amplitude_ratio",
                )
            }
            for scenario in ("T1", "T2", "T3")
        }
        for protocol in ("upstream_convention", "unit_consistent")
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--development-ids", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:3")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    ids = read_ids(args.development_ids)
    if len(ids) != 64:
        raise ValueError("unit audit requires the frozen 64-complex development split")
    train_ids = read_ids(args.data_root / "raw" / "train_MD.txt")
    index = {identifier: position for position, identifier in enumerate(train_ids)}
    if any(identifier not in index for identifier in ids):
        raise ValueError("development ID missing from official train split")

    device = torch.device(args.device)
    dataset = DatasetMISATOSemiFlexibleMultiTrajectory(str(args.data_root), mode="train")
    model = ZeroAcceleration().to(device)
    samples = []
    with torch.no_grad():
        for identifier in ids:
            sample = prepare_single_complex(dataset[index[identifier]], device)
            samples.append({
                "sample_id": identifier,
                "protocols": {
                    name: evaluate_protocol(
                        model, sample, protocol["initial_velocity_scale"]
                    )
                    for name, protocol in config["protocols"].items()
                },
            })

    summary = aggregate(samples)
    thresholds = config["pass_gate"]
    checks = {}
    for scenario in ("T1", "T2", "T3"):
        upstream = summary["upstream_convention"][scenario]
        consistent = summary["unit_consistent"][scenario]
        checks[f"{scenario}_initializer_exact"] = (
            consistent["initializer_frame_rmse_angstrom"]
            <= thresholds["unit_consistent_initializer_rmse_max_angstrom"]
        )
        checks[f"{scenario}_amplitude_in_range"] = (
            thresholds["unit_consistent_step_amplitude_ratio_min"]
            <= consistent["step_amplitude_ratio"]
            <= thresholds["unit_consistent_step_amplitude_ratio_max"]
        )
        checks[f"{scenario}_strictly_better_initializer"] = (
            consistent["initializer_frame_rmse_angstrom"]
            < upstream["initializer_frame_rmse_angstrom"]
        )
    passed = all(checks.values())
    report = {
        "status": "unit_consistency_gate_pass" if passed else "unit_consistency_gate_fail",
        "scope": config["scope"],
        "protocols": config["protocols"],
        "summary": summary,
        "checks": checks,
        "samples": samples,
        "decision_rule": config["decision_rule"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "summary", "checks")}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
