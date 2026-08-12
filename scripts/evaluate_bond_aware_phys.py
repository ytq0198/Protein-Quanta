"""Evaluate bond-aware project Phys diagnostics on saved trajectories."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from protein_quanta.anchoring import anchored_residual_rollout
from protein_quanta.baselines import static_rollout
from protein_quanta.collision import (
    bond_length_diagnostics,
    nonbonded_intramolecular_collision_rate,
)


COVALENT_RADII = {
    5: 0.84,
    6: 0.76,
    7: 0.71,
    8: 0.66,
    9: 0.57,
    14: 1.11,
    15: 1.07,
    16: 1.05,
    17: 1.02,
    34: 1.20,
    35: 1.20,
    53: 1.39,
}


def _metrics(trajectory, truth, bonds, radii):
    result = bond_length_diagnostics(trajectory, truth, bonds)
    result["nonbonded_collision_mean_percent"] = float(
        np.mean(
            nonbonded_intramolecular_collision_rate(
                trajectory,
                radii,
                bonds,
                exclude_graph_distance=2,
            )
        )
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology-report", type=Path, required=True)
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--decay-scale-frames", type=float, default=98.0)
    args = parser.parse_args()

    topology_report = json.loads(args.topology_report.read_text(encoding="utf-8"))
    topology = {
        sample["sample_id"]: sample
        for sample in topology_report["samples"]
        if sample["status"] == "matched"
    }
    reference = json.loads(args.reference_report.read_text(encoding="utf-8"))
    scenario_names = [row["name"] for row in reference["protocol"]["scenarios"]]
    rows = []
    with h5py.File(args.h5, "r") as handle:
        for sample_id in reference["protocol"]["sample_ids"]:
            if sample_id not in topology:
                continue
            group = handle[sample_id]
            ligand_start = int(np.asarray(group["molecules_begin_atom_index"])[-1])
            atomic_numbers = np.asarray(group["atoms_number"])[ligand_start:]
            atomic_numbers = atomic_numbers[atomic_numbers != 1]
            bonds = np.asarray(topology[sample_id]["bonds"], dtype=int)
            scenario_rows = {}
            for scenario_name in scenario_names:
                with np.load(args.trajectory_dir / f"{sample_id}_{scenario_name}.npz") as payload:
                    prediction = np.asarray(payload["prediction"])
                    truth = np.asarray(payload["truth"])
                if prediction.shape[1] != atomic_numbers.size:
                    raise ValueError(f"{sample_id} trajectory/topology atom counts disagree")
                try:
                    radii = np.array(
                        [COVALENT_RADII[int(number)] for number in atomic_numbers]
                    )
                except KeyError as error:
                    raise ValueError(
                        f"{sample_id} uses unsupported atomic number {error.args[0]}"
                    ) from error
                target = truth[2:]
                anchored = anchored_residual_rollout(
                    prediction,
                    truth[:2],
                    beta=args.beta,
                    decay_scale_frames=args.decay_scale_frames,
                )[2:]
                static = static_rollout(truth[:2], target.shape[0])
                scenario_rows[scenario_name] = {
                    name: _metrics(trajectory, target, bonds, radii)
                    for name, trajectory in (
                        ("anchored", anchored),
                        ("neuralmd", prediction[2:]),
                        ("static", static),
                        ("truth", target),
                    )
                }
            rows.append({"sample_id": sample_id, "scenarios": scenario_rows})

    metric_names = (
        "bond_length_mae_angstrom",
        "bond_length_relative_mae",
        "bond_length_violation_percent",
        "extreme_bond_event_percent",
        "nonbonded_collision_mean_percent",
    )
    summary = {
        scenario: {
            model: {
                metric: float(
                    np.mean(
                        [row["scenarios"][scenario][model][metric] for row in rows]
                    )
                )
                for metric in metric_names
            }
            for model in ("anchored", "neuralmd", "static", "truth")
        }
        for scenario in scenario_names
    }
    report = {
        "status": "bond-aware project Phys diagnostics on topology-covered subset; not official Phys score",
        "protocol": {
            "topology_source": topology_report["source"],
            "topology_coverage": topology_report["coverage_fraction"],
            "evaluated_sample_count": len(rows),
            "omitted_sample_count": topology_report["sample_count"] - len(rows),
            "bond_length_reference": "same target frame",
            "bond_violation_relative_threshold": 0.20,
            "nonbonded_exclusions": "covalent graph distance <= 2",
            "collision_threshold": "sum of covalent radii",
            "beta": args.beta,
            "decay_scale_frames": args.decay_scale_frames,
        },
        "summary": summary,
        "samples": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"protocol": report["protocol"], "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
