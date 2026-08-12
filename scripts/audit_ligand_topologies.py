"""Audit traceable ligand bond recovery for a named MISATO split."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from protein_quanta.topology import recover_ligand_topology


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--pdb-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-pairwise-distance-mae", type=float, default=2.0)
    args = parser.parse_args()

    sample_ids = [
        line.strip()
        for line in args.split.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    samples = []
    with h5py.File(args.h5, "r") as handle:
        for sample_id in sample_ids:
            group = handle[sample_id]
            ligand_start = int(np.asarray(group["molecules_begin_atom_index"])[-1])
            atomic_numbers = np.asarray(group["atoms_number"])[ligand_start:]
            coordinates = np.asarray(group["trajectory_coordinates"])[0, ligand_start:]
            heavy_mask = atomic_numbers != 1
            pdb_path = args.pdb_dir / f"{sample_id}.pdb"
            if not pdb_path.exists():
                result = {
                    "status": "pdb_unavailable",
                    "heavy_atom_count": int(np.sum(heavy_mask)),
                }
            else:
                result = recover_ligand_topology(
                    pdb_path,
                    atomic_numbers,
                    coordinates[heavy_mask],
                    maximum_pairwise_distance_mae=args.maximum_pairwise_distance_mae,
                )
            samples.append({"sample_id": sample_id, **result})

    matched = sum(sample["status"] == "matched" for sample in samples)
    report = {
        "status": "project topology audit; PDB coordinates validate order but never infer bonds",
        "source": "RCSB PDB atom records and explicit CONECT bonds",
        "split": str(args.split),
        "sample_count": len(samples),
        "matched_sample_count": matched,
        "coverage_fraction": matched / len(samples) if samples else 0.0,
        "maximum_pairwise_distance_mae_angstrom": args.maximum_pairwise_distance_mae,
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in report if key != "samples"}, indent=2))
    if matched != len(samples):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
