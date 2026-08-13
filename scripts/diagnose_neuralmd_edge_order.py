"""Compare exact ordered radius edge tensors under one rigid rotation."""

import argparse
import json
from pathlib import Path

import torch
from torch_geometric.nn import radius, radius_graph

from NeuralMD.dataloaders.dataloader_MISATO import DataLoaderMISATO
from NeuralMD.datasets.MISATO import DatasetMISATOSemiFlexibleMultiTrajectory
from scripts.audit_neuralmd_multiscale import proper_rotation, transformed_batch


def comparison(original, rotated):
    return {
        "shape": list(original.shape),
        "exactly_equal": bool(torch.equal(original, rotated)),
        "entries_different": int((original != rotated).sum().cpu()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:2")
    args = parser.parse_args()
    device = torch.device(args.device)
    dataset = DatasetMISATOSemiFlexibleMultiTrajectory(str(args.data_root), mode="train")
    batch = next(iter(DataLoaderMISATO(dataset, batch_size=1, num_workers=0, shuffle=False))).to(device)
    rotation = proper_rotation(device, batch.protein_pos.dtype)
    rotated = transformed_batch(batch, rotation, torch.zeros(3, device=device))

    ligand = batch.ligand_trajectory_pos[:, 0, :]
    rotated_ligand = rotated.ligand_trajectory_pos[:, 0, :]
    ligand_original = radius_graph(ligand, r=5.0, batch=batch.batch_ligand)
    ligand_rotated = radius_graph(rotated_ligand, r=5.0, batch=rotated.batch_ligand)

    protein = torch.stack((batch.protein_pos[batch.mask_n], batch.protein_pos[batch.mask_ca], batch.protein_pos[batch.mask_c]), dim=1).reshape(-1, 3)
    rotated_protein = torch.stack((rotated.protein_pos[rotated.mask_n], rotated.protein_pos[rotated.mask_ca], rotated.protein_pos[rotated.mask_c]), dim=1).reshape(-1, 3)
    protein_batch = batch.batch_residue.unsqueeze(0).expand(3, -1).contiguous().view(-1)
    protein_original = radius_graph(protein, r=5.0, batch=protein_batch)
    protein_rotated = radius_graph(rotated_protein, r=5.0, batch=protein_batch)

    complex_original = radius(ligand, batch.protein_pos[batch.mask_ca], 5.0, batch.batch_ligand, batch.batch_residue)
    complex_rotated = radius(rotated_ligand, rotated.protein_pos[rotated.mask_ca], 5.0, rotated.batch_ligand, rotated.batch_residue)
    report = {
        "status": "diagnostic only; no training",
        "ligand": comparison(ligand_original, ligand_rotated),
        "protein": comparison(protein_original, protein_rotated),
        "complex": comparison(complex_original, complex_rotated),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
