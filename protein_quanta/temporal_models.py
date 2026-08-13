"""Parameter-controlled temporal cores for invariant dynamics screening."""

import math

import torch
from torch import nn
from torch.nn import functional as F


ARCHITECTURES = (
    "mlp",
    "rnn",
    "lstm",
    "gru",
    "transformer",
    "transformer_rope",
)


def _rotate_half(values):
    even = values[..., 0::2]
    odd = values[..., 1::2]
    return torch.stack((-odd, even), dim=-1).flatten(-2)


def apply_rotary_position(values, positions=None):
    """Apply norm-preserving RoPE to tensors shaped (batch, heads, time, dim)."""
    if values.ndim != 4 or values.shape[-1] % 2:
        raise ValueError("RoPE requires shape (batch, heads, time, even_dim)")
    length = values.shape[-2]
    if positions is None:
        positions = torch.arange(length, device=values.device, dtype=torch.float32)
    positions = torch.as_tensor(positions, device=values.device, dtype=torch.float32)
    if positions.ndim != 1 or positions.numel() != length:
        raise ValueError("positions must contain one value per time step")
    inverse_frequency = 1.0 / (
        10000.0
        ** (
            torch.arange(0, values.shape[-1], 2, device=values.device).float()
            / values.shape[-1]
        )
    )
    angles = torch.outer(positions, inverse_frequency).repeat_interleave(2, dim=-1)
    cosine = angles.cos().to(dtype=values.dtype)[None, None]
    sine = angles.sin().to(dtype=values.dtype)[None, None]
    return values * cosine + _rotate_half(values) * sine


class _RotaryCausalAttention(nn.Module):
    def __init__(self, hidden_dim, heads):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by transformer_heads")
        if (hidden_dim // heads) % 2:
            raise ValueError("RoPE requires an even attention head dimension")
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, values):
        batch, length, hidden_dim = values.shape
        qkv = self.qkv(values).reshape(
            batch, length, 3, self.heads, self.head_dim
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        query = apply_rotary_position(query)
        key = apply_rotary_position(key)
        attended = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=True
        )
        attended = attended.transpose(1, 2).reshape(batch, length, hidden_dim)
        return self.output(attended)


class _RotaryTransformerBlock(nn.Module):
    def __init__(self, hidden_dim, heads):
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.attention = _RotaryCausalAttention(hidden_dim, heads)
        self.feedforward_norm = nn.LayerNorm(hidden_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )

    def forward(self, values):
        values = values + self.attention(self.attention_norm(values))
        return values + self.feedforward(self.feedforward_norm(values))


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
        elif architecture == "transformer":
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
            self.core = nn.TransformerEncoder(
                block, num_layers=layers, enable_nested_tensor=False
            )
            position = torch.arange(maximum_length, dtype=torch.float32)[:, None]
            frequency = torch.exp(
                torch.arange(0, hidden_dim, 2, dtype=torch.float32)
                * (-math.log(10000.0) / hidden_dim)
            )
            encoding = torch.zeros(maximum_length, hidden_dim)
            encoding[:, 0::2] = torch.sin(position * frequency)
            encoding[:, 1::2] = torch.cos(position * frequency[: encoding[:, 1::2].shape[1]])
            self.register_buffer("position_encoding", encoding, persistent=False)
        else:
            self.core = nn.Sequential(
                *[
                    _RotaryTransformerBlock(hidden_dim, transformer_heads)
                    for _ in range(layers)
                ]
            )
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
        elif self.architecture == "transformer":
            encoded = encoded + self.position_encoding[: encoded.shape[1]]
            mask = nn.Transformer.generate_square_subsequent_mask(
                encoded.shape[1], device=encoded.device
            )
            hidden = self.core(encoded, mask=mask, is_causal=True)
            next_state = None
        else:
            hidden = self.core(encoded)
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
