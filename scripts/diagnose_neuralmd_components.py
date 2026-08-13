"""Localize NeuralMD rotation error across ligand, protein and complex blocks."""

import argparse
import json
from pathlib import Path

import torch

from NeuralMD.dataloaders.dataloader_MISATO import DataLoaderMISATO
from NeuralMD.datasets.MISATO import DatasetMISATOSemiFlexibleMultiTrajectory
from models.NeuralMD_Binding01_2nd_ODE import NeuralMD_Binding01
from scripts.audit_neuralmd_multiscale import model_arguments, proper_rotation, transformed_batch


def block_outputs(model, batch):
    _, ligand_vector, ligand_repr = model.ligand_model(
        z=batch.ligand_x,
        pos=batch.ligand_trajectory_pos[:, 0, :],
        batch=batch.batch_ligand,
        return_repr=True,
    )
    residue_repr = model.protein_model(
        pos_N=batch.protein_pos[batch.mask_n],
        pos_Ca=batch.protein_pos[batch.mask_ca],
        pos_C=batch.protein_pos[batch.mask_c],
        residue_type=batch.protein_backbone_residue.long(),
        batch=batch.batch_residue.long(),
    )
    complex_vector = model.complex_model(
        ligand_repr=ligand_repr,
        ligand_vec_input=ligand_vector,
        pos_ligand=batch.ligand_trajectory_pos[:, 0, :],
        batch_ligand=batch.batch_ligand,
        residue_repr=residue_repr,
        pos_residue=batch.protein_pos[batch.mask_ca],
        batch_residue=batch.batch_residue.long(),
    )
    return ligand_vector, ligand_repr, residue_repr, complex_vector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:2")
    args = parser.parse_args()
    device = torch.device(args.device)
    dataset = DatasetMISATOSemiFlexibleMultiTrajectory(str(args.data_root), mode="train")
    batch = next(iter(DataLoaderMISATO(dataset, batch_size=1, num_workers=0, shuffle=False))).to(device)
    model = NeuralMD_Binding01(model_arguments()).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=False)["binding_model"])
    model.eval()
    rotation = proper_rotation(device, batch.protein_pos.dtype)
    rotated_batch = transformed_batch(batch, rotation, torch.zeros(3, device=device))
    with torch.no_grad():
        ligand_vector, ligand_repr, residue_repr, complex_vector = block_outputs(model, batch)
        rotated_ligand_vector, rotated_ligand_repr, rotated_residue_repr, rotated_complex_vector = block_outputs(model, rotated_batch)
    report = {
        "status": "diagnostic only; no training",
        "ligand_vector_rotation_max_error": float(
            (rotated_ligand_vector - ligand_vector @ rotation.T).abs().max().cpu()
        ),
        "ligand_scalar_repr_invariance_max_error": float((rotated_ligand_repr - ligand_repr).abs().max().cpu()),
        "protein_residue_repr_invariance_max_error": float((rotated_residue_repr - residue_repr).abs().max().cpu()),
        "complex_vector_rotation_max_error": float(
            (rotated_complex_vector - complex_vector @ rotation.T).abs().max().cpu()
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
