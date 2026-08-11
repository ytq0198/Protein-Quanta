"""Reference trajectory baselines."""

import numpy as np


def _validated_history(history, horizon):
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    coordinates = np.asarray(history)
    if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
        raise ValueError("history must have shape (frames, atoms, 3)")
    if coordinates.shape[0] == 0:
        raise ValueError("history must contain at least one observed frame")
    return coordinates


def static_rollout(history, horizon):
    """Repeat the last observed coordinates for every future frame."""
    coordinates = _validated_history(history, horizon)
    return np.repeat(coordinates[-1][None, ...], repeats=horizon, axis=0)


def linear_rollout(history, horizon):
    """Extrapolate using the displacement between the last two frames."""
    coordinates = _validated_history(history, horizon)
    if coordinates.shape[0] < 2:
        raise ValueError("linear rollout requires at least two observed frames")

    velocity = coordinates[-1] - coordinates[-2]
    steps = np.arange(1, horizon + 1).reshape(-1, 1, 1)
    return coordinates[-1][None, ...] + steps * velocity[None, ...]
