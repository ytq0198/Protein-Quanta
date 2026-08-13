"""Evaluate the frozen 64/16 gate only after every paired checkpoint exists."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torchdiffeq import odeint

from NeuralMD.datasets.MISATO import DatasetMISATOSemiFlexibleMultiTrajectory
from protein_quanta.collision import bond_length_diagnostics
from protein_quanta.dense_equivariant_dynamics import DenseEquivariantAcceleration
from protein_quanta.effect_gate_barrier import verify_training_artifacts
from protein_quanta.metrics import (
    contact_map_agreement,
    coordinate_rmse,
    dynamics_distribution_metrics,
    error_growth_summary,
    radius_of_gyration_error,
    rmsf_error,
)
from protein_quanta.neuralmd_multiscale import neuralmd_ode_rollout
from protein_quanta.scenarios import competition_scenarios
from scripts.train_evaluate_dense_paired_phys import prepare_single_complex, read_ids


def rollout(model, sample, start, horizon):
    return neuralmd_ode_rollout(
        model, odeint, sample, start=start, horizon=horizon, scaling=100,
        step_size=0.025, method="euler",
    )[1]


def evaluate_model(model, ids, samples, topology):
    scenario_rows = {scenario.name: [] for scenario in competition_scenarios()}
    per_sample = []
    for identifier, sample in zip(ids, samples):
        sample_rows = {}
        bonds = np.asarray(topology[identifier]["bonds"], dtype=int)
        for scenario in competition_scenarios():
            first, _ = scenario.initializer_indices
            prediction_path = rollout(
                model, sample, first, scenario.target_end - first
            )
            prediction = prediction_path[2:].detach().cpu().numpy()
            truth = sample.ligand_trajectory_pos[
                :, scenario.target_start : scenario.target_end + 1, :
            ].transpose(0, 1).cpu().numpy()
            growth = error_growth_summary(prediction, truth)
            metrics = {
                "coordinate_rmse_angstrom": coordinate_rmse(prediction, truth),
                "rmse_slope_angstrom_per_frame": growth[
                    "coordinate_rmse_slope_angstrom_per_frame"
                ],
                "rmsf_mae_angstrom": rmsf_error(prediction, truth),
                "radius_of_gyration_mae_angstrom": float(
                    np.mean(radius_of_gyration_error(prediction, truth))
                ),
                "contact_map_agreement_mean": float(
                    np.mean(contact_map_agreement(prediction, truth, cutoff=4.5))
                ),
                "nonfinite_frame_fraction": float(
                    1.0 - np.mean(np.isfinite(prediction).all(axis=(1, 2)))
                ),
                "dynamics_distribution": dynamics_distribution_metrics(
                    prediction, truth, maximum_lag=10
                ),
                **bond_length_diagnostics(prediction, truth, bonds),
            }
            sample_rows[scenario.name] = metrics
            scenario_rows[scenario.name].append(metrics)
        per_sample.append({"sample_id": identifier, "scenarios": sample_rows})

    summary = {}
    scalar_metrics = (
        "coordinate_rmse_angstrom",
        "rmse_slope_angstrom_per_frame",
        "rmsf_mae_angstrom",
        "radius_of_gyration_mae_angstrom",
        "contact_map_agreement_mean",
        "nonfinite_frame_fraction",
        "bond_length_mae_angstrom",
        "bond_length_relative_mae",
        "bond_length_violation_percent",
        "extreme_bond_event_percent",
    )
    for scenario_name, rows in scenario_rows.items():
        summary[scenario_name] = {
            metric: float(np.mean([row[metric] for row in rows]))
            for metric in scalar_metrics
        }
        dynamic_keys = [
            key for key in rows[0]["dynamics_distribution"]
            if key != "velocity_autocorrelation_lags"
        ]
        summary[scenario_name]["dynamics_distribution"] = {
            key: float(np.mean([
                row["dynamics_distribution"][key] for row in rows
            ]))
            for key in dynamic_keys
        }
    return summary, per_sample


def mean_summaries(seed_results, arm):
    result = {}
    for scenario in ("T1", "T2", "T3"):
        rows = [row["summary"][arm][scenario] for row in seed_results]
        result[scenario] = {}
        for key, value in rows[0].items():
            if key == "dynamics_distribution":
                result[scenario][key] = {
                    nested: float(np.mean([row[key][nested] for row in rows]))
                    for nested in value
                }
            else:
                result[scenario][key] = float(np.mean([row[key] for row in rows]))
    return result


def improvement_fraction(control, candidate):
    denominator = max(abs(control), np.finfo(float).eps)
    return float((control - candidate) / denominator)


def evaluate_gate(config, seed_results):
    control = mean_summaries(seed_results, "control")
    candidate = mean_summaries(seed_results, "candidate")
    scenarios = ("T1", "T2", "T3")
    macro_control = float(np.mean([
        control[name]["coordinate_rmse_angstrom"] for name in scenarios
    ]))
    macro_candidate = float(np.mean([
        candidate[name]["coordinate_rmse_angstrom"] for name in scenarios
    ]))
    seed_macro = []
    for row in seed_results:
        left = float(np.mean([
            row["summary"]["control"][name]["coordinate_rmse_angstrom"]
            for name in scenarios
        ]))
        right = float(np.mean([
            row["summary"]["candidate"][name]["coordinate_rmse_angstrom"]
            for name in scenarios
        ]))
        seed_macro.append({
            "seed": row["seed"],
            "control": left,
            "candidate": right,
            "candidate_minus_control": right - left,
            "candidate_improves": right < left,
        })

    dynamic_checks = {
        "rmsf_mae_angstrom": (
            candidate["T3"]["rmsf_mae_angstrom"]
            < control["T3"]["rmsf_mae_angstrom"]
        ),
        "radius_of_gyration_mae_angstrom": (
            candidate["T3"]["radius_of_gyration_mae_angstrom"]
            < control["T3"]["radius_of_gyration_mae_angstrom"]
        ),
        "contact_map_agreement_mean": (
            candidate["T3"]["contact_map_agreement_mean"]
            > control["T3"]["contact_map_agreement_mean"]
        ),
    }
    all_phys_control = np.mean([
        control[name]["bond_length_mae_angstrom"] for name in scenarios
    ])
    all_phys_candidate = np.mean([
        candidate[name]["bond_length_mae_angstrom"] for name in scenarios
    ])
    all_extreme_control = np.mean([
        control[name]["extreme_bond_event_percent"] for name in scenarios
    ])
    all_extreme_candidate = np.mean([
        candidate[name]["extreme_bond_event_percent"] for name in scenarios
    ])
    thresholds = config["pass_gate"]
    checks = {
        "macro_coordinate_rmse_improves": macro_candidate < macro_control,
        "paired_seed_wins_at_least": (
            sum(row["candidate_improves"] for row in seed_macro)
            >= thresholds["paired_seed_wins_at_least"]
        ),
        "T1_coordinate_rmse_max_worsening_fraction": (
            (candidate["T1"]["coordinate_rmse_angstrom"]
             - control["T1"]["coordinate_rmse_angstrom"])
            / max(control["T1"]["coordinate_rmse_angstrom"], np.finfo(float).eps)
            <= thresholds["T1_coordinate_rmse_max_worsening_fraction"]
        ),
        "T3_coordinate_rmse_min_improvement_fraction": (
            improvement_fraction(
                control["T3"]["coordinate_rmse_angstrom"],
                candidate["T3"]["coordinate_rmse_angstrom"],
            ) >= thresholds["T3_coordinate_rmse_min_improvement_fraction"]
        ),
        "T3_rmse_slope_min_improvement_fraction": (
            improvement_fraction(
                control["T3"]["rmse_slope_angstrom_per_frame"],
                candidate["T3"]["rmse_slope_angstrom_per_frame"],
            ) >= thresholds["T3_rmse_slope_min_improvement_fraction"]
        ),
        "T3_dynamic_metrics_improved_at_least": (
            sum(dynamic_checks.values())
            >= thresholds["T3_dynamic_metrics_improved_at_least"]
        ),
        "bond_length_mae_max_worsening_fraction": (
            (all_phys_candidate - all_phys_control)
            / max(all_phys_control, np.finfo(float).eps)
            <= thresholds["bond_length_mae_max_worsening_fraction"]
        ),
        "extreme_bond_event_max_increase_percentage_points": (
            all_extreme_candidate - all_extreme_control
            <= thresholds["extreme_bond_event_max_increase_percentage_points"]
        ),
        "all_rollouts_finite": all(
            candidate[name]["nonfinite_frame_fraction"] == 0.0
            and control[name]["nonfinite_frame_fraction"] == 0.0
            for name in scenarios
        ),
    }
    deltas = [row["candidate_minus_control"] for row in seed_macro]
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "dynamic_checks": dynamic_checks,
        "mean_summary": {"control": control, "candidate": candidate},
        "macro_coordinate_rmse": {
            "control": macro_control,
            "candidate": macro_candidate,
            "improvement_fraction": improvement_fraction(
                macro_control, macro_candidate
            ),
        },
        "paired_seed_macro": seed_macro,
        "paired_candidate_minus_control_mean": float(np.mean(deltas)),
        "paired_candidate_minus_control_sd": float(np.std(deltas, ddof=1)),
        "bond_length_mae_macro": {
            "control": float(all_phys_control),
            "candidate": float(all_phys_candidate),
        },
        "extreme_bond_event_macro": {
            "control": float(all_extreme_control),
            "candidate": float(all_extreme_candidate),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--holdout-ids", type=Path, required=True)
    parser.add_argument("--topology-report", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:2")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    # Deliberate barrier: do not read the holdout IDs, topology, or dataset until
    # every final artifact exists and its manifest hash has been verified.
    verified = verify_training_artifacts(
        config, args.training_dir, args.checkpoint_dir
    )

    holdout_ids = read_ids(args.holdout_ids)
    if len(holdout_ids) != config["data"]["holdout_count"]:
        raise ValueError("invalid frozen holdout split")
    topology_report = json.loads(args.topology_report.read_text(encoding="utf-8"))
    topology = {
        row["sample_id"]: row for row in topology_report["samples"]
        if row["status"] == "matched"
    }
    if set(topology) != set(holdout_ids):
        raise ValueError("explicit topology does not cover the frozen holdout")

    train_ids = read_ids(args.data_root / "raw" / "train_MD.txt")
    index = {identifier: position for position, identifier in enumerate(train_ids)}
    device = torch.device(args.device)
    dataset = DatasetMISATOSemiFlexibleMultiTrajectory(str(args.data_root), mode="train")
    holdout = [
        prepare_single_complex(dataset[index[identifier]], device)
        for identifier in holdout_ids
    ]

    seed_results = []
    for seed in config["pairing"]["seeds"]:
        summaries = {}
        samples = {}
        for arm in ("control", "candidate"):
            checkpoint = torch.load(
                verified[seed]["checkpoints"][arm]["path"],
                map_location=device,
            )
            if checkpoint.get("seed") != seed or checkpoint.get("config") != config:
                raise ValueError(f"checkpoint metadata mismatch for {arm} seed {seed}")
            model = DenseEquivariantAcceleration(hidden_dim=32).to(device)
            model.load_state_dict(checkpoint["model"])
            model.eval()
            with torch.no_grad():
                summaries[arm], samples[arm] = evaluate_model(
                    model, holdout_ids, holdout, topology
                )
        seed_results.append({
            "seed": seed,
            "summary": summaries,
            "samples": samples,
        })

    gate = evaluate_gate(config, seed_results)
    report = {
        "status": "effect_gate_pass" if gate["passed"] else "effect_gate_fail",
        "scope": "frozen 16-complex train holdout; official validation/test untouched",
        "holdout_disclosure": (
            "target trajectories were opened only after all six checkpoint hashes "
            "passed; identifiers and frame-0 geometry had previously been used only "
            "to establish explicit PDB CONECT topology coverage"
        ),
        "verified_training_artifacts": verified,
        "gate": gate,
        "seeds": seed_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "gate": gate}, indent=2))


if __name__ == "__main__":
    main()
