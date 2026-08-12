"""Parameter-controlled temporal cores for invariant dynamics screening."""

import math

import torch
from torch import nn


ARCHITECTURES = ("mlp", "rnn", "lstm", "gru", "transformer")


class TemporalFeatureForecaster(nn.Module):
    def __init__(
        self,
        feature_dim,
        architecture,
        hidden_dim=64,
        layers=1,
        maximum_length=128,
        transformer_heads=4,
    ):
        super().__init__()
        if architecture not in ARCHITECTURES:
            raise ValueError(f"unknown architecture: {architecture}")
        if feature_dim <= 0 or hidden_dim <= 0 or layers <= 0:
            raise ValueError("model dimensions must be positive")
        self.architecture = architecture
        self.maximum_length = maximum_length
        self.input_projection = nn.Linear(feature_dim, hidden_dim)
        if architecture == "mlp":
            self.core = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.GELU()
            )
        elif architecture in ("rnn", "lstm", "gru"):
            core_type = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[
                architecture
            ]
            self.core = core_type(
                hidden_dim,
                hidden_dim,
                num_layers=layers,
                batch_first=True,
            )
        else:
            if hidden_dim % transformer_heads:
                raise ValueError("hidden_dim must be divisible by transformer_heads")
            block = nn.TransformerEncoderLayer(
                hidden_dim,
                transformer_heads,
                dim_feedforward=2 * hidden_dim,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.core = nn.TransformerEncoder(block, num_layers=layers)
            position = torch.arange(maximum_length, dtype=torch.float32)[:, None]
            frequency = torch.exp(
                torch.arange(0, hidden_dim, 2, dtype=torch.float32)
                * (-math.log(10000.0) / hidden_dim)
            )
            encoding = torch.zeros(maximum_length, hidden_dim)
            encoding[:, 0::2] = torch.sin(position * frequency)
            encoding[:, 1::2] = torch.cos(position * frequency[: encoding[:, 1::2].shape[1]])
            self.register_buffer("position_encoding", encoding, persistent=False)
        self.output_projection = nn.Linear(hidden_dim, feature_dim)

    def forward(self, sequence, state=None):
        if sequence.ndim != 3:
            raise ValueError("sequence must have shape (batch, time, features)")
        if sequence.shape[1] > self.maximum_length:
            raise ValueError("sequence exceeds maximum_length")
        encoded = self.input_projection(sequence)
        if self.architecture == "mlp":
            hidden = self.core(encoded)
            next_state = None
        elif self.architecture in ("rnn", "lstm", "gru"):
            hidden, next_state = self.core(encoded, state)
        else:
            encoded = encoded + self.position_encoding[: encoded.shape[1]]
            mask = nn.Transformer.generate_square_subsequent_mask(
                encoded.shape[1], device=encoded.device
            )
            hidden = self.core(encoded, mask=mask, is_causal=True)
            next_state = None
        return self.output_projection(hidden), next_state

    @torch.no_grad()
    def rollout(self, observed, horizon):
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        generated = observed
        if self.architecture in ("rnn", "lstm", "gru"):
            prediction, state = self(observed)
            current = prediction[:, -1:]
            outputs = [current]
            for _ in range(1, horizon):
                prediction, state = self(current, state)
                current = prediction[:, -1:]
                outputs.append(current)
            return torch.cat(outputs, dim=1)
        outputs = []
        for _ in range(horizon):
            prediction, _ = self(generated)
            current = prediction[:, -1:]
            outputs.append(current)
            generated = torch.cat([generated, current], dim=1)
        return torch.cat(outputs, dim=1)


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())
