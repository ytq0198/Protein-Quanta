"""Rigid-motion-invariant trajectory features for temporal architecture screens."""

import numpy as np


FEATURE_NAMES = (
    "radius_of_gyration",
    "pair_distance_q10",
    "pair_distance_q25",
    "pair_distance_q50",
    "pair_distance_q75",
    "pair_distance_q90",
    "pair_distance_mean",
    "pair_distance_std",
    "step_magnitude_mean",
    "step_magnitude_std",
    "step_magnitude_q50",
    "step_magnitude_q90",
)


def invariant_trajectory_features(trajectory):
    """Map an xyz trajectory to fixed-size SE(3)-invariant frame features."""
    trajectory = np.asarray(trajectory, dtype=float)
    if trajectory.ndim != 3 or trajectory.shape[0] < 2 or trajectory.shape[-1] != 3:
        raise ValueError("trajectory must have shape (frames>=2, atoms, 3)")
    if trajectory.shape[1] < 2:
        raise ValueError("at least two atoms are required")
    if not np.isfinite(trajectory).all():
        raise ValueError("trajectory must be finite")

    centered = trajectory - trajectory.mean(axis=1, keepdims=True)
    radius_of_gyration = np.sqrt(
        np.mean(np.sum(np.square(centered), axis=-1), axis=1)
    )
    pair_i, pair_j = np.triu_indices(trajectory.shape[1], k=1)
    distances = np.linalg.norm(
        trajectory[:, pair_i, :] - trajectory[:, pair_j, :], axis=-1
    )
    pair_quantiles = np.quantile(distances, [0.10, 0.25, 0.50, 0.75, 0.90], axis=1).T
    pair_mean = np.mean(distances, axis=1)
    pair_std = np.std(distances, axis=1)

    steps = np.linalg.norm(np.diff(trajectory, axis=0), axis=-1)
    step_mean = np.concatenate([[0.0], np.mean(steps, axis=1)])
    step_std = np.concatenate([[0.0], np.std(steps, axis=1)])
    step_q50 = np.concatenate([[0.0], np.quantile(steps, 0.50, axis=1)])
    step_q90 = np.concatenate([[0.0], np.quantile(steps, 0.90, axis=1)])
    return np.column_stack(
        [
            radius_of_gyration,
            pair_quantiles,
            pair_mean,
            pair_std,
            step_mean,
            step_std,
            step_q50,
            step_q90,
        ]
    )
