"""Audit rotation stability of the radius graphs used by NeuralMD."""

import argparse
import json
from pathlib import Path

import torch
from torch_geometric.nn import radius, radius_graph

from NeuralMD.dataloaders.dataloader_MISATO import DataLoaderMISATO
from NeuralMD.datasets.MISATO import DatasetMISATOSemiFlexibleMultiTrajectory
from scripts.audit_neuralmd_multiscale import proper_rotation, transformed_batch


def edge_set(edge_index):
    return set(map(tuple, edge_index.detach().cpu().T.tolist()))


def graph_report(position, rotated_position, batch, cutoff=5.0):
    original = radius_graph(position, r=cutoff, batch=batch)
    rotated = radius_graph(rotated_position, r=cutoff, batch=batch)
    left, right = edge_set(original), edge_set(rotated)
    counts = torch.bincount(original[1], minlength=position.shape[0])
    return {
        "original_edges": len(left),
        "rotated_edges": len(right),
        "symmetric_difference": len(left.symmetric_difference(right)),
        "edge_tensor_exactly_equal": bool(torch.equal(original, rotated)),
        "ordered_edge_entries_different": int((original != rotated).sum().cpu()),
        "max_incoming_neighbors": int(counts.max().cpu()),
        "nodes_at_default_32_neighbor_cap": int((counts >= 32).sum().cpu()),
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
    protein_position = torch.stack((
        batch.protein_pos[batch.mask_n],
        batch.protein_pos[batch.mask_ca],
        batch.protein_pos[batch.mask_c],
    ), dim=1).reshape(-1, 3)
    rotated_protein_position = torch.stack((
        rotated.protein_pos[rotated.mask_n],
        rotated.protein_pos[rotated.mask_ca],
        rotated.protein_pos[rotated.mask_c],
    ), dim=1).reshape(-1, 3)
    expanded_batch = batch.batch_residue.unsqueeze(0).expand(3, -1).contiguous().view(-1)
    ligand_position = batch.ligand_trajectory_pos[:, 0, :]
    rotated_ligand_position = rotated.ligand_trajectory_pos[:, 0, :]
    ligand_graph = graph_report(
        ligand_position, rotated_ligand_position, batch.batch_ligand
    )
    protein_graph = graph_report(
        protein_position, rotated_protein_position, expanded_batch
    )
    complex_original = radius(
        ligand_position, batch.protein_pos[batch.mask_ca], 5.0,
        batch.batch_ligand, batch.batch_residue,
    )
    complex_rotated = radius(
        rotated_ligand_position, rotated.protein_pos[rotated.mask_ca], 5.0,
        rotated.batch_ligand, rotated.batch_residue,
    )
    original_set, rotated_set = edge_set(complex_original), edge_set(complex_rotated)
    report = {
        "status": "diagnostic only; no training",
        "ligand_radius_graph": ligand_graph,
        "protein_radius_graph": protein_graph,
        "complex_radius_edges": {
            "original_edges": len(original_set),
            "rotated_edges": len(rotated_set),
            "symmetric_difference": len(original_set.symmetric_difference(rotated_set)),
            "edge_tensor_exactly_equal": bool(torch.equal(complex_original, complex_rotated)),
            "ordered_edge_entries_different": int((complex_original != complex_rotated).sum().cpu()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
