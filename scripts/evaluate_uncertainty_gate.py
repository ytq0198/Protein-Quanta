"""Run validation-only grouped LOOCV for the uncertainty anchor gate."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from protein_quanta.anchoring import anchored_residual_rollout
from protein_quanta.uncertainty_gate import (
    FEATURE_NAMES,
    extract_gate_features,
    grouped_leave_one_out,
    strong_anchor_label,
)
from scripts.evaluate_anchor_scenarios import _load_trajectory
from scripts.evaluate_naive_baselines import _evaluate
from scripts.evaluate_neuralmd_scenarios import _aggregate_scenarios
from scripts.smoke_neuralmd import _scenario_rollout_comparison


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _balanced_accuracy(labels, predictions):
    labels = np.asarray(labels, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    if labels.shape != predictions.shape or labels.ndim != 1:
        raise ValueError("labels and predictions must be matching vectors")
    recalls = []
    for label in (0, 1):
        selected = labels == label
        if not selected.any():
            raise ValueError("both classes are required for balanced accuracy")
        recalls.append(float(np.mean(predictions[selected] == label)))
    return float(np.mean(recalls))


def _metric_changes(baseline, candidate):
    changes = {}
    for scenario in ("T1", "T2", "T3"):
        before = baseline[scenario]["anchored"]
        after = candidate[scenario]["anchored"]
        changes[scenario] = {
            "coordinate_rmse_percent": 100.0
            * (
                after["coordinate_rmse_angstrom"]
                / before["coordinate_rmse_angstrom"]
                - 1.0
            ),
            "matching_percent": 100.0
            * (
                after["matching_mean_angstrom"]
                / before["matching_mean_angstrom"]
                - 1.0
            ),
            "stability_points": after["stability_mean_percent"]
            - before["stability_mean_percent"],
            "rmsf_percent": 100.0
            * (
                after["rmsf_mae_angstrom"]
                / before["rmsf_mae_angstrom"]
                - 1.0
            ),
        }
    return changes


def _promotion_gates(
    changes, selected_strong_count, record_count, balanced_accuracy
):
    selection_rate = selected_strong_count / record_count
    return {
        "T1_matching_and_stability_improve": changes["T1"]["matching_percent"]
        < 0
        and changes["T1"]["stability_points"] > 0,
        "T2_matching_and_stability_improve": changes["T2"]["matching_percent"]
        < 0
        and changes["T2"]["stability_points"] > 0,
        "all_coordinate_rmse_changes_at_most_2_percent": all(
            row["coordinate_rmse_percent"] <= 2.0 for row in changes.values()
        ),
        "all_rmsf_changes_at_most_5_percent": all(
            row["rmsf_percent"] <= 5.0 for row in changes.values()
        ),
        "strong_selection_rate_between_0_20_and_0_80": 0.20
        <= selection_rate
        <= 0.80,
        "balanced_accuracy_at_least_0_60": balanced_accuracy >= 0.60,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decay-scale-frames", type=float, default=98.0)
    parser.add_argument("--contact-cutoff", type=float, default=4.5)
    parser.add_argument("--l2", type=float, default=10.0)
    args = parser.parse_args()

    reference = json.loads(args.reference_report.read_text(encoding="utf-8"))
    baseline_report = json.loads(args.baseline_report.read_text(encoding="utf-8"))
    sample_ids = reference["protocol"]["sample_ids"]
    scenarios = [row["name"] for row in reference["protocol"]["scenarios"]]
    if scenarios != ["T1", "T2", "T3"]:
        raise ValueError("reference report must contain T1, T2, and T3 in order")

    records = []
    trajectories = {}
    for sample_id in sample_ids:
        for scenario in scenarios:
            path = args.trajectory_dir / f"{sample_id}_{scenario}.npz"
            prediction, truth = _load_trajectory(path)
            history = truth[:2]
            metrics = {}
            anchored_trajectories = {}
            for beta in (1.0, 8.0):
                anchored = anchored_residual_rollout(
                    prediction,
                    history,
                    beta=beta,
                    decay_scale_frames=args.decay_scale_frames,
                )
                anchored_trajectories[beta] = anchored
                metrics[beta] = _evaluate(
                    anchored[2:], truth[2:], args.contact_cutoff
                )
            features = extract_gate_features(prediction, history, scenario)
            records.append(
                {
                    "sample_id": sample_id,
                    "scenario": scenario,
                    "features": np.asarray(
                        [features[name] for name in FEATURE_NAMES], dtype=float
                    ),
                    "feature_values": features,
                    "label": int(strong_anchor_label(metrics[1.0], metrics[8.0])),
                    "beta1_metrics": metrics[1.0],
                    "beta8_metrics": metrics[8.0],
                }
            )
            trajectories[(sample_id, scenario)] = (
                prediction,
                truth,
                anchored_trajectories,
            )

    oof = grouped_leave_one_out(records, l2=args.l2)
    prediction_by_key = {
        (row["sample_id"], row["scenario"]): row for row in oof["predictions"]
    }
    samples = []
    detailed_records = []
    for sample_id in sample_ids:
        scenario_results = {}
        for scenario in scenarios:
            key = (sample_id, scenario)
            record = next(
                row
                for row in records
                if (row["sample_id"], row["scenario"]) == key
            )
            decision = prediction_by_key[key]
            prediction, truth, anchored_trajectories = trajectories[key]
            selected = anchored_trajectories[decision["selected_beta"]]
            comparison = _scenario_rollout_comparison(
                prediction,
                truth,
                observed_local_frames=2,
                contact_cutoff=args.contact_cutoff,
            )
            comparison["anchored"] = _evaluate(
                selected[2:], truth[2:], args.contact_cutoff
            )
            scenario_results[scenario] = {"comparison": comparison}
            detailed_records.append(
                {
                    **decision,
                    "feature_values": record["feature_values"],
                    "beta1_metrics": record["beta1_metrics"],
                    "beta8_metrics": record["beta8_metrics"],
                    "selected_metrics": comparison["anchored"],
                }
            )
        samples.append({"sample_id": sample_id, "scenarios": scenario_results})

    summary = _aggregate_scenarios(samples)
    changes = _metric_changes(baseline_report["summary"], summary)
    labels = [row["label"] for row in oof["predictions"]]
    predicted_labels = [row["selected_beta"] == 8.0 for row in oof["predictions"]]
    balanced_accuracy = _balanced_accuracy(labels, predicted_labels)
    selected_strong_count = int(sum(predicted_labels))
    gates = _promotion_gates(
        changes,
        selected_strong_count=selected_strong_count,
        record_count=len(records),
        balanced_accuracy=balanced_accuracy,
    )
    report = {
        "protocol": {
            "status": "validation-only grouped OOF proxy; not official score",
            "selection_allowed": True,
            "internal_test_accessed": False,
            "feature_names": list(FEATURE_NAMES),
            "model": "balanced L2 logistic regression",
            "l2": args.l2,
            "decision_threshold": 0.5,
            "candidate_betas": [1.0, 8.0],
            "decay_scale_frames": args.decay_scale_frames,
            "contact_cutoff_angstrom": args.contact_cutoff,
            "sample_ids": sample_ids,
            "sample_count": len(sample_ids),
            "record_count": len(records),
            "trajectory_dir": str(args.trajectory_dir.resolve()),
            "reference_report": str(args.reference_report.resolve()),
            "reference_report_sha256": _sha256(args.reference_report),
            "baseline_report": str(args.baseline_report.resolve()),
            "baseline_report_sha256": _sha256(args.baseline_report),
        },
        "oof_classification": {
            "positive_count": int(sum(labels)),
            "selected_strong_count": selected_strong_count,
            "selected_strong_rate": selected_strong_count / len(records),
            "balanced_accuracy": balanced_accuracy,
        },
        "relative_to_global_beta1": changes,
        "promotion_gates": gates,
        "validation_feasible": bool(all(gates.values())),
        "folds": oof["folds"],
        "summary": summary,
        "records": detailed_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "oof_classification": report["oof_classification"],
                "relative_to_global_beta1": changes,
                "promotion_gates": gates,
                "validation_feasible": report["validation_feasible"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

