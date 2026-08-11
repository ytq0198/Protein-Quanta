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
