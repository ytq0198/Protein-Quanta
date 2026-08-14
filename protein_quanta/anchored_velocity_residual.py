"""Control-anchored, invariant-gated E(3) residual dynamics."""

import torch
from torch import nn


class AnchoredGatedVelocityDynamics(nn.Module):
    """Add a bounded residual while exactly matching an anchor at initialization.

    The residual scale is initialized at zero.  Its tanh parameterization makes
    the initial acceleration exactly equal to the anchor while retaining a
    nonzero first derivative with respect to the scalar scale.  The per-ligand
    gate depends only on ligand identity and squared speed, so it is invariant to
    translations, rotations, and reflections.
    """

    def __init__(
        self,
        anchor,
        residual,
        hidden_dim=32,
        ligand_classes=119,
        maximum_gate=0.25,
        initial_gate_fraction=0.05,
        freeze_anchor=True,
    ):
        super().__init__()
        if maximum_gate <= 0:
            raise ValueError("maximum_gate must be positive")
        if not 0 < initial_gate_fraction < 1:
            raise ValueError("initial_gate_fraction must lie strictly between 0 and 1")
        self.anchor = anchor
        self.residual = residual
        self.maximum_gate = float(maximum_gate)
        self.gate_embedding = nn.Embedding(ligand_classes, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        output = self.gate[-1]
        nn.init.zeros_(output.weight)
        initial_logit = torch.logit(torch.tensor(float(initial_gate_fraction)))
        nn.init.constant_(output.bias, float(initial_logit))
        self.residual_scale_logit = nn.Parameter(torch.zeros(()))
        if freeze_anchor:
            for parameter in self.anchor.parameters():
                parameter.requires_grad_(False)

    def gate_coefficient(self, ligand_type, velocity):
        feature = self.gate_embedding(ligand_type.long())
        log_speed = torch.log1p(velocity.square().sum(dim=-1, keepdim=True))
        return self.maximum_gate * torch.sigmoid(
            self.gate(torch.cat((feature, log_speed), dim=-1))
        )

    def forward(self, time, state, condition):
        velocity, _ = state
        ligand_type = condition[0]
        anchor_acceleration = self.anchor(time, state, condition)[0]
        residual_acceleration = self.residual(time, state, condition)[0]
        gate = self.gate_coefficient(ligand_type, velocity)
        residual_scale = torch.tanh(self.residual_scale_logit)
        acceleration = (
            anchor_acceleration
            + gate * residual_scale * residual_acceleration
        )
        return acceleration, velocity
