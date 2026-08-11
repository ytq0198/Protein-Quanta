"""Evaluate Static and Linear rollouts on MISATO ligand trajectories."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np

from protein_quanta.baselines import linear_rollout, static_rollout
from protein_quanta.metrics import (
    aligned_rmsd,
    contact_map_agreement,
    coordinate_mae,
    coordinate_rmse,
    distance_matching,
    distance_stability,
    error_growth_summary,
    radius_of_gyration_error,
    rmsf_error,
)
from protein_quanta.misato import load_ligand_trajectory


def _evaluate(prediction, truth, contact_cutoff):
    matching = distance_matching(prediction, truth)
    stability = distance_stability(prediction, truth)
    aligned = aligned_rmsd(prediction, truth)
    gyration_error = radius_of_gyration_error(prediction, truth)
    contact_agreement = contact_map_agreement(
        prediction, truth, cutoff=contact_cutoff
    )
    result = {
        "diagnostic_status": "proxy; not official score",
        "coordinate_mae_angstrom": coordinate_mae(prediction, truth),
        "coordinate_rmse_angstrom": coordinate_rmse(prediction, truth),
        "matching_mean_angstrom": float(np.mean(matching)),
        "stability_mean_percent": float(np.mean(stability)),
        "aligned_rmsd_mean_angstrom": float(np.mean(aligned)),
        "radius_of_gyration_mae_angstrom": float(np.mean(gyration_error)),
        "rmsf_mae_angstrom": rmsf_error(prediction, truth),
        "contact_map_agreement_mean": float(np.mean(contact_agreement)),
        "matching_by_frame_angstrom": matching.tolist(),
        "stability_by_frame_percent": stability.tolist(),
        "aligned_rmsd_by_frame_angstrom": aligned.tolist(),
        "radius_of_gyration_error_by_frame_angstrom": gyration_error.tolist(),
        "contact_map_agreement_by_frame": contact_agreement.tolist(),
    }
    result.update(error_growth_summary(prediction, truth))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--observed-frames", type=int, default=2)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--contact-cutoff", type=float, default=4.5)
    args = parser.parse_args()
    if args.observed_frames < 2:
        raise ValueError("observed-frames must be at least 2")

    started = perf_counter()
    rows = []
    with h5py.File(args.input, "r") as handle:
        sample_ids = sorted(handle.keys())
        if args.max_samples is not None:
            sample_ids = sample_ids[: args.max_samples]
        for sample_id in sample_ids:
            trajectory = load_ligand_trajectory(handle[sample_id])
            coordinates = trajectory.coordinates
            if coordinates.shape[0] <= args.observed_frames:
                raise ValueError(f"{sample_id} has no frames left to predict")
            history = coordinates[: args.observed_frames]
            truth = coordinates[args.observed_frames :]
            horizon = truth.shape[0]
            for baseline_name, rollout in (
                ("static", static_rollout),
                ("linear", linear_rollout),
            ):
                prediction = rollout(history, horizon)
                row = {
                    "sample_id": sample_id,
                    "baseline": baseline_name,
                    "observed_frames": args.observed_frames,
                    "predicted_frames": horizon,
                    "heavy_ligand_atoms": coordinates.shape[1],
                }
                row.update(
                    _evaluate(
                        prediction,
                        truth,
                        contact_cutoff=args.contact_cutoff,
                    )
                )
                rows.append(row)

    summary = {}
    for baseline_name in ("static", "linear"):
        selected = [row for row in rows if row["baseline"] == baseline_name]
        summary[baseline_name] = {
            metric: float(np.mean([row[metric] for row in selected]))
            for metric in (
                "coordinate_mae_angstrom",
                "coordinate_rmse_angstrom",
                "matching_mean_angstrom",
                "stability_mean_percent",
                "aligned_rmsd_mean_angstrom",
                "radius_of_gyration_mae_angstrom",
                "rmsf_mae_angstrom",
                "contact_map_agreement_mean",
            )
        }

    report = {
        "protocol": {
            "input": str(args.input.resolve()),
            "sample_count": len(sample_ids),
            "observed_frames": args.observed_frames,
            "forecast_start_frame_zero_based": args.observed_frames,
            "heavy_ligand_atoms_only": True,
            "centering": "NeuralMD global mean over all atoms and all frames",
            "aggregation": "unweighted mean of per-complex metrics",
            "contact_cutoff_angstrom": args.contact_cutoff,
            "diagnostic_metrics": (
                "aligned RMSD, radius-of-gyration error, RMSF error, and "
                "intramolecular contact-map agreement are project-defined proxies"
            ),
            "status": "T1 proxy; not an official competition score",
        },
        "summary": summary,
        "samples": rows,
        "runtime_seconds": perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
