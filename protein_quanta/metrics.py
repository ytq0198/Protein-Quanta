"""Reconstruction metrics used during baseline reproduction."""

import numpy as np


def _coordinate_errors(prediction, truth, atom_mask):
    prediction = np.asarray(prediction)
    truth = np.asarray(truth)
    if prediction.shape != truth.shape:
        raise ValueError("prediction and truth must have matching shapes")
    if prediction.ndim != 3 or prediction.shape[-1] != 3:
        raise ValueError("trajectories must have shape (frames, atoms, 3)")

    errors = prediction - truth
    if atom_mask is not None:
        mask = np.asarray(atom_mask, dtype=bool)
        if mask.shape != (prediction.shape[1],):
            raise ValueError(f"atom_mask must have shape ({prediction.shape[1]},)")
        errors = errors[:, mask, :]
    return errors


def coordinate_mae(prediction, truth, atom_mask=None):
    errors = _coordinate_errors(prediction, truth, atom_mask)
    return float(np.mean(np.abs(errors)))


def coordinate_rmse(prediction, truth, atom_mask=None):
    errors = _coordinate_errors(prediction, truth, atom_mask)
    return float(np.sqrt(np.mean(np.square(errors))))


def error_growth_summary(prediction, truth, atom_mask=None):
    """Summarize how coordinate error grows across a forecast horizon."""
    errors = _coordinate_errors(prediction, truth, atom_mask)
    per_frame = np.sqrt(np.mean(np.square(errors), axis=(1, 2)))
    if per_frame.size >= 3:
        early, middle, late = np.array_split(per_frame, 3)
    else:
        early = per_frame[:1]
        middle = per_frame[(per_frame.size - 1) // 2 : (per_frame.size - 1) // 2 + 1]
        late = per_frame[-1:]
    frame_index = np.arange(per_frame.shape[0], dtype=float)
    slope = np.polyfit(frame_index, per_frame, deg=1)[0] if per_frame.size > 1 else 0.0
    return {
        "coordinate_rmse_by_frame_angstrom": per_frame.tolist(),
        "coordinate_rmse_early_mean_angstrom": float(np.mean(early)),
        "coordinate_rmse_middle_mean_angstrom": float(np.mean(middle)),
        "coordinate_rmse_late_mean_angstrom": float(np.mean(late)),
        "coordinate_rmse_slope_angstrom_per_frame": float(slope),
    }


def distance_matching(prediction, truth, atom_mask=None):
    prediction, truth = _masked_trajectories(prediction, truth, atom_mask)
    distance_gap = _pairwise_distances(prediction) - _pairwise_distances(truth)
    return np.sqrt(np.mean(np.square(distance_gap), axis=(1, 2)))


def distance_stability(prediction, truth, threshold=0.5, atom_mask=None):
    prediction, truth = _masked_trajectories(prediction, truth, atom_mask)
    distance_gap = np.abs(
        _pairwise_distances(prediction) - _pairwise_distances(truth)
    )
    return 100.0 * np.mean(distance_gap <= threshold, axis=(1, 2))


def aligned_rmsd(prediction, truth, atom_mask=None):
    """Return per-frame heavy-atom RMSD after optimal rigid alignment."""
    prediction, truth = _masked_trajectories(prediction, truth, atom_mask)
    values = []
    for predicted_frame, true_frame in zip(prediction, truth):
        predicted_centered = predicted_frame - predicted_frame.mean(axis=0)
        true_centered = true_frame - true_frame.mean(axis=0)
        left, _, right_transpose = np.linalg.svd(
            predicted_centered.T @ true_centered
        )
        correction = np.eye(3)
        correction[-1, -1] = np.sign(
            np.linalg.det(left @ right_transpose)
        )
        rotation = left @ correction @ right_transpose
        residual = predicted_centered @ rotation - true_centered
        values.append(np.sqrt(np.mean(np.sum(np.square(residual), axis=-1))))
    return np.asarray(values)


def radius_of_gyration_error(prediction, truth, atom_mask=None):
    """Return per-frame absolute error in the unweighted radius of gyration."""
    prediction, truth = _masked_trajectories(prediction, truth, atom_mask)
    predicted_rg = _radius_of_gyration(prediction)
    true_rg = _radius_of_gyration(truth)
    return np.abs(predicted_rg - true_rg)


def rmsf_error(prediction, truth, atom_mask=None):
    """Return mean absolute error between per-atom fluctuation amplitudes."""
    prediction, truth = _masked_trajectories(prediction, truth, atom_mask)
    predicted_rmsf = _rmsf(prediction)
    true_rmsf = _rmsf(truth)
    return float(np.mean(np.abs(predicted_rmsf - true_rmsf)))


def _pearson_correlation(left, right):
    left = np.asarray(left, dtype=float).reshape(-1)
    right = np.asarray(right, dtype=float).reshape(-1)
    if left.shape != right.shape or not left.size:
        raise ValueError("correlation inputs must be non-empty and match")
    left_centered = left - np.mean(left)
    right_centered = right - np.mean(right)
    denominator = np.linalg.norm(left_centered) * np.linalg.norm(right_centered)
    if denominator <= np.finfo(float).eps:
        return 1.0 if np.allclose(left, right) else 0.0
    return float(np.dot(left_centered, right_centered) / denominator)


def _rankdata(values):
    values = np.asarray(values, dtype=float).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def _wasserstein_1d(left, right):
    """Return an equal-weight empirical 1-D Wasserstein distance."""
    left = np.asarray(left, dtype=float).reshape(-1)
    right = np.asarray(right, dtype=float).reshape(-1)
    if not left.size or not right.size:
        raise ValueError("Wasserstein inputs must be non-empty")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Wasserstein inputs must be finite")
    if left.size == right.size:
        return float(np.mean(np.abs(np.sort(left) - np.sort(right))))
    quantile_count = max(left.size, right.size)
    quantiles = (np.arange(quantile_count, dtype=float) + 0.5) / quantile_count
    left_values = np.quantile(left, quantiles)
    right_values = np.quantile(right, quantiles)
    return float(np.mean(np.abs(left_values - right_values)))


def _mean_column_wasserstein(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.shape != right.shape or left.ndim != 2 or not left.shape[1]:
        raise ValueError("column distributions must be matching non-empty matrices")
    return float(
        np.mean(
            [
                _wasserstein_1d(left[:, column], right[:, column])
                for column in range(left.shape[1])
            ]
        )
    )


def _velocity_autocorrelation(trajectory, maximum_lag):
    velocity = np.diff(trajectory, axis=0)
    energy = float(np.mean(np.sum(np.square(velocity), axis=-1)))
    if energy <= np.finfo(float).eps:
        return np.zeros(maximum_lag, dtype=float)
    values = []
    for lag in range(1, maximum_lag + 1):
        dot_products = np.sum(velocity[:-lag] * velocity[lag:], axis=-1)
        values.append(float(np.mean(dot_products) / energy))
    return np.asarray(values)


def dynamics_distribution_metrics(prediction, truth, atom_mask=None, maximum_lag=10):
    """Compare trajectory distributions without collapsing them to coordinates.

    Metrics intentionally remain project diagnostics rather than an organizer
    score. Lower is better except RMSF Pearson correlation and amplitude ratio,
    whose ideal values are one.
    """
    prediction, truth = _masked_trajectories(prediction, truth, atom_mask)
    if prediction.shape[0] < 3:
        raise ValueError("dynamics metrics require at least three frames")
    if not isinstance(maximum_lag, int) or maximum_lag <= 0:
        raise ValueError("maximum_lag must be a positive integer")

    predicted_rmsf = _rmsf(prediction)
    true_rmsf = _rmsf(truth)
    predicted_rg = _radius_of_gyration(prediction)
    true_rg = _radius_of_gyration(truth)
    pair_indices = np.triu_indices(prediction.shape[1], k=1)
    if pair_indices[0].size == 0:
        raise ValueError("dynamics metrics require at least two selected atoms")
    predicted_pairs = _pairwise_distances(prediction)[
        :, pair_indices[0], pair_indices[1]
    ]
    true_pairs = _pairwise_distances(truth)[:, pair_indices[0], pair_indices[1]]
    predicted_steps = np.linalg.norm(np.diff(prediction, axis=0), axis=-1)
    true_steps = np.linalg.norm(np.diff(truth, axis=0), axis=-1)
    lag_count = min(maximum_lag, prediction.shape[0] - 2)
    predicted_vacf = _velocity_autocorrelation(prediction, lag_count)
    true_vacf = _velocity_autocorrelation(truth, lag_count)
    true_amplitude = float(np.mean(true_steps))
    amplitude_ratio = (
        float(np.mean(predicted_steps) / true_amplitude)
        if true_amplitude > np.finfo(float).eps
        else (1.0 if np.allclose(predicted_steps, 0.0) else float("inf"))
    )
    return {
        "rmsf_profile_mae_angstrom": float(
            np.mean(np.abs(predicted_rmsf - true_rmsf))
        ),
        "rmsf_profile_pearson": _pearson_correlation(
            predicted_rmsf, true_rmsf
        ),
        "rmsf_profile_spearman": _pearson_correlation(
            _rankdata(predicted_rmsf), _rankdata(true_rmsf)
        ),
        "rg_wasserstein_angstrom": _wasserstein_1d(predicted_rg, true_rg),
        "pair_distance_wasserstein_angstrom": _mean_column_wasserstein(
            predicted_pairs, true_pairs
        ),
        "step_displacement_wasserstein_angstrom": _mean_column_wasserstein(
            predicted_steps, true_steps
        ),
        "step_amplitude_ratio": amplitude_ratio,
        "velocity_autocorrelation_mae": float(
            np.mean(np.abs(predicted_vacf - true_vacf))
        ),
        "velocity_autocorrelation_lags": lag_count,
    }


def contact_map_agreement(prediction, truth, cutoff, atom_mask=None):
    """Return per-frame agreement of unordered intramolecular contact pairs."""
    if cutoff <= 0:
        raise ValueError("cutoff must be positive")
    prediction, truth = _masked_trajectories(prediction, truth, atom_mask)
    if prediction.shape[1] < 2:
        raise ValueError("contact maps require at least two selected atoms")
    pair_indices = np.triu_indices(prediction.shape[1], k=1)
    predicted_contacts = _pairwise_distances(prediction)[:, pair_indices[0], pair_indices[1]] <= cutoff
    true_contacts = _pairwise_distances(truth)[:, pair_indices[0], pair_indices[1]] <= cutoff
    return np.mean(predicted_contacts == true_contacts, axis=1)


def _masked_trajectories(prediction, truth, atom_mask):
    prediction = np.asarray(prediction)
    truth = np.asarray(truth)
    _coordinate_errors(prediction, truth, atom_mask)
    if atom_mask is None:
        return prediction, truth
    mask = np.asarray(atom_mask, dtype=bool)
    return prediction[:, mask, :], truth[:, mask, :]


def _pairwise_distances(trajectory):
    differences = trajectory[:, :, None, :] - trajectory[:, None, :, :]
    return np.linalg.norm(differences, axis=-1)


def _radius_of_gyration(trajectory):
    centered = trajectory - trajectory.mean(axis=1, keepdims=True)
    return np.sqrt(np.mean(np.sum(np.square(centered), axis=-1), axis=1))


def _rmsf(trajectory):
    centered = trajectory - trajectory.mean(axis=0, keepdims=True)
    return np.sqrt(np.mean(np.sum(np.square(centered), axis=-1), axis=0))
