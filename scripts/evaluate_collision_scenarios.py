"""Evaluate project-defined collision proxies on saved scenario trajectories."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from protein_quanta.anchoring import anchored_residual_rollout
from protein_quanta.baselines import static_rollout
from protein_quanta.collision import binding_collision_rate, intramolecular_collision_rate
from scripts.smoke_neuralmd import _load_official_sample


def _collision_metrics(trajectory, ligand_radii, protein_positions, protein_radii):
    return {
        "ligand_collision_mean_percent": float(
            np.mean(intramolecular_collision_rate(trajectory, ligand_radii))
        ),
        "binding_collision_mean_percent": float(
            np.mean(
                binding_collision_rate(
                    trajectory, ligand_radii, protein_positions, protein_radii
                )
            )
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--decay-scale-frames", type=float, default=98.0)
    args = parser.parse_args()

    sys.path.insert(0, str(args.upstream))
    from NeuralMD.evaluation import covalent_radii_dict

    reference = json.loads(args.reference_report.read_text(encoding="utf-8"))
    sample_ids = reference["protocol"]["sample_ids"]
    scenario_names = [item["name"] for item in reference["protocol"]["scenarios"]]
    samples = []
    for sample_id in sample_ids:
        data, _ = _load_official_sample(args.upstream, args.h5, sample_id)
        ligand_radii = np.array(
            [covalent_radii_dict[int(value) + 1] for value in data.ligand_x]
        )
        protein_types = np.ones(data.protein_pos.shape[0], dtype=int)
        protein_types[data.mask_n.numpy()] = 6
        protein_types[data.mask_ca.numpy()] = 5
        protein_types[data.mask_c.numpy()] = 5
        protein_radii = np.array(
            [covalent_radii_dict[int(value) + 1] for value in protein_types]
        )
        protein_positions = data.protein_pos.numpy()
        scenario_rows = {}
        for scenario_name in scenario_names:
            with np.load(args.trajectory_dir / f"{sample_id}_{scenario_name}.npz") as payload:
                prediction = np.asarray(payload["prediction"])
                truth = np.asarray(payload["truth"])
            anchored = anchored_residual_rollout(
                prediction,
                truth[:2],
                beta=args.beta,
                decay_scale_frames=args.decay_scale_frames,
            )[2:]
            target = truth[2:]
            static = static_rollout(truth[:2], target.shape[0])
            scenario_rows[scenario_name] = {
                model: _collision_metrics(
                    trajectory,
                    ligand_radii,
                    protein_positions,
                    protein_radii,
                )
                for model, trajectory in (
                    ("anchored", anchored),
                    ("neuralmd", prediction[2:]),
                    ("static", static),
                    ("truth", target),
                )
            }
        samples.append({"sample_id": sample_id, "scenarios": scenario_rows})

    summary = {}
    for scenario_name in scenario_names:
        summary[scenario_name] = {}
        for model in ("anchored", "neuralmd", "static", "truth"):
            summary[scenario_name][model] = {
                metric: float(
                    np.mean(
                        [sample["scenarios"][scenario_name][model][metric] for sample in samples]
                    )
                )
                for metric in (
                    "ligand_collision_mean_percent",
                    "binding_collision_mean_percent",
                )
            }
    report = {
        "protocol": {
            "status": "project collision proxy; not official Phys score",
            "pair_rule": "distance below sum of covalent radii",
            "intramolecular_self_and_duplicate_pairs_excluded": True,
            "covalent_bond_exclusions_available": False,
            "beta": args.beta,
            "decay_scale_frames": args.decay_scale_frames,
            "sample_ids": sample_ids,
            "scenarios": scenario_names,
        },
        "summary": summary,
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
