"""Minimal ProAR-inspired interpolation/forecasting mechanism."""

from __future__ import annotations

import torch
from torch import nn


def _condition(step, horizon, maximum_horizon, reference):
    step = torch.as_tensor(step, device=reference.device, dtype=reference.dtype)
    horizon = torch.as_tensor(
        horizon, device=reference.device, dtype=reference.dtype
    )
    if step.ndim == 0:
        step = step.expand(reference.shape[0])
    if horizon.ndim == 0:
        horizon = horizon.expand(reference.shape[0])
    if step.shape != (reference.shape[0],) or horizon.shape != (reference.shape[0],):
        raise ValueError("step and horizon must be scalar or one value per batch item")
    if torch.any(horizon <= 0) or torch.any(horizon > maximum_horizon):
        raise ValueError("horizon must be in [1, maximum_horizon]")
    if torch.any(step < 0) or torch.any(step >= horizon):
        raise ValueError("step must be in [0, horizon)")
    return (step / horizon).unsqueeze(-1), (horizon / maximum_horizon).unsqueeze(-1)


class _ConditionedNetwork(nn.Module):
    def __init__(self, feature_dim, hidden_dim, layers):
        super().__init__()
        if feature_dim <= 0 or hidden_dim <= 0 or layers <= 0:
            raise ValueError("network dimensions must be positive")
        modules = [nn.Linear(2 * feature_dim + 2, hidden_dim), nn.GELU()]
        for _ in range(layers - 1):
            modules.extend((nn.Linear(hidden_dim, hidden_dim), nn.GELU()))
        modules.append(nn.Linear(hidden_dim, feature_dim))
        self.network = nn.Sequential(*modules)

    def forward(self, left, right, relative_step, relative_horizon):
        return self.network(
            torch.cat((left, right, relative_step, relative_horizon), dim=-1)
        )


class BridgeInterpolator(nn.Module):
    """Predict an interior state as a linear bridge plus learned residual."""

    def __init__(self, feature_dim, hidden_dim=48, layers=2, maximum_horizon=4):
        super().__init__()
        self.maximum_horizon = maximum_horizon
        self.residual = _ConditionedNetwork(feature_dim, hidden_dim, layers)

    def forward(self, start, endpoint, step, horizon):
        if start.ndim != 2 or start.shape != endpoint.shape:
            raise ValueError("start and endpoint must share shape (batch, features)")
        relative_step, relative_horizon = _condition(
            step, horizon, self.maximum_horizon, start
        )
        linear = start + relative_step * (endpoint - start)
        correction = self.residual(
            start, endpoint, relative_step, relative_horizon
        )
        return linear + relative_step * (1.0 - relative_step) * correction


class EndpointForecaster(nn.Module):
    """Forecast a block endpoint from an intermediate state and block start."""

    def __init__(self, feature_dim, hidden_dim=48, layers=2, maximum_horizon=4):
        super().__init__()
        self.maximum_horizon = maximum_horizon
        self.delta = _ConditionedNetwork(feature_dim, hidden_dim, layers)

    def forward(self, intermediate, start, step, horizon):
        if intermediate.ndim != 2 or intermediate.shape != start.shape:
            raise ValueError("intermediate and start must share shape (batch, features)")
        relative_step, relative_horizon = _condition(
            step, horizon, self.maximum_horizon, intermediate
        )
        return intermediate + self.delta(
            intermediate, start, relative_step, relative_horizon
        )


def _batch_step(batch_size, value, reference):
    return torch.full(
        (batch_size,), value, dtype=reference.dtype, device=reference.device
    )


@torch.no_grad()
def block_rollout(
    interpolator,
    forecaster,
    observed,
    horizon,
    maximum_horizon=4,
    alternating=True,
):
    """Roll out blockwise with or without alternating endpoint refinement."""

    if observed.ndim != 3 or observed.shape[1] < 1:
        raise ValueError("observed must have shape (batch, time, features)")
    if horizon <= 0 or maximum_horizon <= 0:
        raise ValueError("horizons must be positive")
    start = observed[:, -1]
    outputs = []
    remaining = horizon
    while remaining:
        block = min(maximum_horizon, remaining)
        batch = start.shape[0]
        block_values = _batch_step(batch, block, start)
        zero = _batch_step(batch, 0, start)
        endpoint = forecaster(start, start, zero, block_values)
        if alternating:
            for step_value in range(1, block):
                step = _batch_step(batch, step_value, start)
                intermediate = interpolator(
                    start, endpoint, step, block_values
                )
                endpoint = forecaster(
                    intermediate, start, step, block_values
                )
        for step_value in range(1, block):
            step = _batch_step(batch, step_value, start)
            outputs.append(interpolator(start, endpoint, step, block_values))
        outputs.append(endpoint)
        start = endpoint
        remaining -= block
    return torch.stack(outputs, dim=1)


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())
