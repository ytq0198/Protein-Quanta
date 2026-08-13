"""Velocity-aware dense E(3)-equivariant ligand acceleration field."""

import torch
from torch import nn
from torch.nn import functional


class VelocityEquivariantAcceleration(nn.Module):
    """Predict acceleration from invariant position/velocity contractions.

    Scalar messages use squared distances, squared speeds, and velocity-relative
    displacement dot products.  Vector outputs are restricted to relative
    displacements and velocity itself, so rotations and reflections commute with
    the field while translations cancel from every learned input.
    """

    def __init__(self, hidden_dim=32, ligand_classes=119, residue_classes=26, scale=0.1):
        super().__init__()
        self.scale = float(scale)
        self.ligand_embedding = nn.Embedding(ligand_classes, hidden_dim)
        self.residue_embedding = nn.Embedding(residue_classes, hidden_dim)
        self.ligand_pair = nn.Sequential(
            nn.Linear(2 * hidden_dim + 5, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )
        self.protein_pair = nn.Sequential(
            nn.Linear(2 * hidden_dim + 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )
        self.damping = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
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
        speed_squared = velocity.square().sum(dim=-1, keepdim=True)

        ligand_displacement = (
            ligand_position.unsqueeze(0) - ligand_position.unsqueeze(1)
        )
        ligand_squared_distance = ligand_displacement.square().sum(
            dim=-1, keepdim=True
        )
        ligand_i = ligand_feature.unsqueeze(1).expand(
            -1, ligand_feature.shape[0], -1
        )
        ligand_j = ligand_feature.unsqueeze(0).expand(
            ligand_feature.shape[0], -1, -1
        )
        speed_i = speed_squared.unsqueeze(1).expand(-1, velocity.shape[0], -1)
        speed_j = speed_squared.unsqueeze(0).expand(velocity.shape[0], -1, -1)
        velocity_i = velocity.unsqueeze(1)
        velocity_j = velocity.unsqueeze(0)
        projection_i = (velocity_i * ligand_displacement).sum(
            dim=-1, keepdim=True
        )
        projection_j = (velocity_j * ligand_displacement).sum(
            dim=-1, keepdim=True
        )
        ligand_weight = self.ligand_pair(torch.cat(
            (
                ligand_i,
                ligand_j,
                ligand_squared_distance,
                speed_i,
                speed_j,
                projection_i,
                projection_j,
            ),
            dim=-1,
        ))
        ligand_mask = (
            ligand_batch.unsqueeze(1) == ligand_batch.unsqueeze(0)
        ) & ~torch.eye(
            ligand_position.shape[0], dtype=torch.bool,
            device=ligand_position.device,
        )
        ligand_weight = (
            ligand_weight
            * self._radial_envelope(ligand_squared_distance)
            * ligand_mask.unsqueeze(-1)
        )
        ligand_force = (ligand_displacement * ligand_weight).sum(dim=1)
        ligand_count = ligand_mask.sum(dim=1, keepdim=True).clamp_min(1)
        ligand_force = ligand_force / ligand_count

        protein_displacement = (
            protein_position.unsqueeze(0) - ligand_position.unsqueeze(1)
        )
        protein_squared_distance = protein_displacement.square().sum(
            dim=-1, keepdim=True
        )
        ligand_context = ligand_feature.unsqueeze(1).expand(
            -1, protein_feature.shape[0], -1
        )
        protein_context = protein_feature.unsqueeze(0).expand(
            ligand_feature.shape[0], -1, -1
        )
        protein_speed = speed_squared.unsqueeze(1).expand(
            -1, protein_feature.shape[0], -1
        )
        protein_projection = (
            velocity.unsqueeze(1) * protein_displacement
        ).sum(dim=-1, keepdim=True)
        protein_weight = self.protein_pair(torch.cat(
            (
                ligand_context,
                protein_context,
                protein_squared_distance,
                protein_speed,
                protein_projection,
            ),
            dim=-1,
        ))
        protein_mask = ligand_batch.unsqueeze(1) == residue_batch.unsqueeze(0)
        protein_weight = (
            protein_weight
            * self._radial_envelope(protein_squared_distance)
            * protein_mask.unsqueeze(-1)
        )
        protein_force = (protein_displacement * protein_weight).sum(dim=1)
        protein_count = protein_mask.sum(dim=1, keepdim=True).clamp_min(1)
        protein_force = protein_force / protein_count

        damping_coefficient = functional.softplus(
            self.damping(torch.cat((ligand_feature, speed_squared), dim=-1))
        )
        force_acceleration = (
            ligand_force + protein_force
        ) / ligand_mass.unsqueeze(-1).clamp_min(1e-6)
        acceleration = self.scale * (
            force_acceleration - damping_coefficient * velocity
        )
        return acceleration, velocity
