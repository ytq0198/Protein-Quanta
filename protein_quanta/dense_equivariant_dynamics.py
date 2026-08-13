"""Small dense E(3)-equivariant acceleration field for correctness pilots."""

import torch
from torch import nn


class DenseEquivariantAcceleration(nn.Module):
    """Predict ligand acceleration from invariant radial scalar messages.

    Every vector output is a learned scalar function of invariant inputs times a
    relative displacement. Protein coordinates are conditions and never updated.
    Dense smooth interactions avoid radius-graph ordering and cutoff discontinuities.
    """

    def __init__(self, hidden_dim=32, ligand_classes=119, residue_classes=26, scale=0.1):
        super().__init__()
        self.scale = float(scale)
        self.ligand_embedding = nn.Embedding(ligand_classes, hidden_dim)
        self.residue_embedding = nn.Embedding(residue_classes, hidden_dim)
        self.ligand_pair = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1), nn.Tanh(),
        )
        self.protein_pair = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1), nn.Tanh(),
        )

    @staticmethod
    def _radial_envelope(squared_distance):
        return torch.exp(-squared_distance / 25.0)

    def forward(self, time, state, condition):
        velocity, ligand_position = state
        ligand_type, ligand_batch, ligand_mass = condition[:3]
        protein_position = condition[4]
        residue_type, residue_batch = condition[6:8]
        ligand_feature = self.ligand_embedding(ligand_type.long())
        protein_feature = self.residue_embedding(residue_type.long())

        ligand_displacement = (
            ligand_position.unsqueeze(0) - ligand_position.unsqueeze(1)
        )
        ligand_squared_distance = ligand_displacement.square().sum(dim=-1, keepdim=True)
        ligand_i = ligand_feature.unsqueeze(1).expand(-1, ligand_feature.shape[0], -1)
        ligand_j = ligand_feature.unsqueeze(0).expand(ligand_feature.shape[0], -1, -1)
        ligand_weight = self.ligand_pair(
            torch.cat((ligand_i, ligand_j, ligand_squared_distance), dim=-1)
        )
        ligand_mask = (
            ligand_batch.unsqueeze(1) == ligand_batch.unsqueeze(0)
        ) & ~torch.eye(
            ligand_position.shape[0], dtype=torch.bool, device=ligand_position.device
        )
        ligand_weight = (
            ligand_weight * self._radial_envelope(ligand_squared_distance)
            * ligand_mask.unsqueeze(-1)
        )
        ligand_force = (ligand_displacement * ligand_weight).sum(dim=1)
        ligand_count = ligand_mask.sum(dim=1, keepdim=True).clamp_min(1)
        ligand_force = ligand_force / ligand_count

        protein_displacement = (
            protein_position.unsqueeze(0) - ligand_position.unsqueeze(1)
        )
        protein_squared_distance = protein_displacement.square().sum(dim=-1, keepdim=True)
        ligand_context = ligand_feature.unsqueeze(1).expand(-1, protein_feature.shape[0], -1)
        protein_context = protein_feature.unsqueeze(0).expand(ligand_feature.shape[0], -1, -1)
        protein_weight = self.protein_pair(
            torch.cat((ligand_context, protein_context, protein_squared_distance), dim=-1)
        )
        protein_mask = ligand_batch.unsqueeze(1) == residue_batch.unsqueeze(0)
        protein_weight = (
            protein_weight * self._radial_envelope(protein_squared_distance)
            * protein_mask.unsqueeze(-1)
        )
        protein_force = (protein_displacement * protein_weight).sum(dim=1)
        protein_count = protein_mask.sum(dim=1, keepdim=True).clamp_min(1)
        protein_force = protein_force / protein_count

        acceleration = self.scale * (ligand_force + protein_force)
        acceleration = acceleration / ligand_mass.unsqueeze(-1).clamp_min(1e-6)
        return acceleration, velocity
