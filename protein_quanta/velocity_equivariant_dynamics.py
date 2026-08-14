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

    def __init__(
        self,
        hidden_dim=32,
        ligand_classes=119,
        residue_classes=26,
        scale=0.1,
        bounded_damping_max=None,
        normalize_velocity_invariants=False,
        speed_squared_scale=1.0,
        projection_scale=5.0,
        initial_damping_fraction=0.05,
        protein_top_k=None,
    ):
        super().__init__()
        self.scale = float(scale)
        self.bounded_damping_max = (
            None if bounded_damping_max is None else float(bounded_damping_max)
        )
        self.normalize_velocity_invariants = bool(normalize_velocity_invariants)
        self.speed_squared_scale = float(speed_squared_scale)
        self.projection_scale = float(projection_scale)
        self.protein_top_k = None if protein_top_k is None else int(protein_top_k)
        if self.bounded_damping_max is not None and self.bounded_damping_max <= 0:
            raise ValueError("bounded_damping_max must be positive")
        if self.speed_squared_scale <= 0 or self.projection_scale <= 0:
            raise ValueError("velocity invariant scales must be positive")
        if not 0 < initial_damping_fraction < 1:
            raise ValueError("initial_damping_fraction must lie strictly between 0 and 1")
        if self.protein_top_k is not None and self.protein_top_k <= 0:
            raise ValueError("protein_top_k must be positive")
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
        if self.bounded_damping_max is not None:
            output = self.damping[-1]
            nn.init.zeros_(output.weight)
            initial_logit = torch.logit(torch.tensor(float(initial_damping_fraction)))
            nn.init.constant_(output.bias, float(initial_logit))

    @staticmethod
    def _radial_envelope(squared_distance):
        return torch.exp(-squared_distance / 25.0)

    def _velocity_invariants(self, speed_squared, projection):
        if not self.normalize_velocity_invariants:
            return speed_squared, projection
        normalized_speed = torch.log1p(
            speed_squared / self.speed_squared_scale
        )
        normalized_projection = torch.tanh(
            projection / self.projection_scale
        )
        return normalized_speed, normalized_projection

    def damping_coefficient(self, ligand_feature, speed_squared):
        damping_input_speed, _ = self._velocity_invariants(
            speed_squared, torch.zeros_like(speed_squared)
        )
        raw = self.damping(torch.cat((ligand_feature, damping_input_speed), dim=-1))
        if self.bounded_damping_max is None:
            return functional.softplus(raw)
        return self.bounded_damping_max * torch.sigmoid(raw)

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
        speed_invariant, _ = self._velocity_invariants(
            speed_squared, torch.zeros_like(speed_squared)
        )
        speed_i = speed_invariant.unsqueeze(1).expand(-1, velocity.shape[0], -1)
        speed_j = speed_invariant.unsqueeze(0).expand(velocity.shape[0], -1, -1)
        velocity_i = velocity.unsqueeze(1)
        velocity_j = velocity.unsqueeze(0)
        projection_i = (velocity_i * ligand_displacement).sum(
            dim=-1, keepdim=True
        )
        projection_j = (velocity_j * ligand_displacement).sum(
            dim=-1, keepdim=True
        )
        _, projection_i = self._velocity_invariants(speed_i, projection_i)
        _, projection_j = self._velocity_invariants(speed_j, projection_j)
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

        if self.protein_top_k is None:
            protein_displacement = (
                protein_position.unsqueeze(0) - ligand_position.unsqueeze(1)
            )
            protein_context = protein_feature.unsqueeze(0).expand(
                ligand_feature.shape[0], -1, -1
            )
            protein_mask = ligand_batch.unsqueeze(1) == residue_batch.unsqueeze(0)
        else:
            pair_distance = torch.cdist(ligand_position, protein_position)
            same_complex = ligand_batch.unsqueeze(1) == residue_batch.unsqueeze(0)
            masked_distance = pair_distance.masked_fill(~same_complex, torch.inf)
            neighbour_count = min(self.protein_top_k, protein_position.shape[0])
            nearest_distance, nearest_index = torch.topk(
                masked_distance, neighbour_count, dim=1, largest=False
            )
            protein_displacement = (
                protein_position[nearest_index] - ligand_position.unsqueeze(1)
            )
            protein_context = protein_feature[nearest_index]
            protein_mask = torch.isfinite(nearest_distance)
        protein_squared_distance = protein_displacement.square().sum(
            dim=-1, keepdim=True
        )
        ligand_context = ligand_feature.unsqueeze(1).expand(
            -1, protein_context.shape[1], -1
        )
        protein_speed = speed_invariant.unsqueeze(1).expand(
            -1, protein_context.shape[1], -1
        )
        protein_projection = (
            velocity.unsqueeze(1) * protein_displacement
        ).sum(dim=-1, keepdim=True)
        _, protein_projection = self._velocity_invariants(
            protein_speed, protein_projection
        )
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
        protein_weight = (
            protein_weight
            * self._radial_envelope(protein_squared_distance)
            * protein_mask.unsqueeze(-1)
        )
        protein_force = (protein_displacement * protein_weight).sum(dim=1)
        protein_count = protein_mask.sum(dim=1, keepdim=True).clamp_min(1)
        protein_force = protein_force / protein_count

        damping_coefficient = self.damping_coefficient(
            ligand_feature, speed_squared
        )
        force_acceleration = (
            ligand_force + protein_force
        ) / ligand_mass.unsqueeze(-1).clamp_min(1e-6)
        acceleration = self.scale * (
            force_acceleration - damping_coefficient * velocity
        )
        return acceleration, velocity
