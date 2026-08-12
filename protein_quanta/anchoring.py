"""Static-anchor residual transforms for long trajectory rollouts."""

import numpy as np


def validate_scenario_betas(beta_by_scenario, expected_scenarios):
    """Validate and normalize a complete scenario-to-beta mapping."""
    expected = list(expected_scenarios)
    if len(expected) != len(set(expected)):
        raise ValueError("expected_scenarios must contain unique names")

    provided = set(beta_by_scenario)
    expected_set = set(expected)
    missing = sorted(expected_set - provided)
    unknown = sorted(provided - expected_set)
    if missing:
        raise ValueError(f"scenario beta policy is missing: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"scenario beta policy contains unknown: {', '.join(unknown)}")

    normalized = {}
    for scenario in expected:
        beta = float(beta_by_scenario[scenario])
        if not np.isfinite(beta) or beta < 0:
            raise ValueError("scenario beta values must be finite and non-negative")
        normalized[scenario] = beta
    return normalized


def anchored_residual_rollout(
    prediction,
    history,
    beta,
    decay_scale_frames=98,
):
    """Decay a learned forecast residual toward the last observed frame.

    The first two frames are replaced with the exact observed history. Beta
    zero leaves the learned forecast unchanged; positive beta progressively
    reduces its displacement from the Static baseline.
    """
    prediction = np.asarray(prediction)
    history = np.asarray(history)
    if prediction.ndim != 3 or prediction.shape[-1] != 3:
        raise ValueError("prediction must have shape (frames, atoms, 3)")
    if history.ndim != 3 or history.shape[0] != 2 or history.shape[-1] != 3:
        raise ValueError("history must have shape (2, atoms, 3)")
    if prediction.shape[0] < 3:
        raise ValueError("prediction must contain at least three frames")
    if prediction.shape[1:] != history.shape[1:]:
        raise ValueError("prediction and history atom dimensions must match")
    if not np.isfinite(prediction).all() or not np.isfinite(history).all():
        raise ValueError("prediction and history must contain finite values")
    if not np.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and non-negative")
    if not np.isfinite(decay_scale_frames) or decay_scale_frames <= 0:
        raise ValueError("decay_scale_frames must be finite and positive")

    result = prediction.copy()
    result[:2] = history
    steps = np.arange(1, prediction.shape[0] - 1, dtype=float)
    weights = np.exp(-float(beta) * steps / float(decay_scale_frames))
    anchor = history[-1]
    result[2:] = anchor + weights[:, None, None] * (
        prediction[2:] - anchor
    )
    return result
