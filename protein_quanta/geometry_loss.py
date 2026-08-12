"""Geometry-aware training losses and deterministic gradient calibration."""

import numpy as np
import torch
import torch.nn.functional as functional


def pair_distance_smooth_l1(prediction, truth, atom_batch, beta=0.5):
    """Average Smooth-L1 distance residuals per complex, then per batch."""
    if not isinstance(prediction, torch.Tensor) or not isinstance(truth, torch.Tensor):
        raise ValueError("prediction and truth must be torch tensors")
    if prediction.shape != truth.shape:
        raise ValueError("prediction and truth must have matching shapes")
    if prediction.ndim != 3 or prediction.shape[-1] != 3:
        raise ValueError("coordinates must have shape (frames, atoms, 3)")
    if not isinstance(atom_batch, torch.Tensor) or atom_batch.ndim != 1:
        raise ValueError("atom_batch must be a one-dimensional torch tensor")
    if atom_batch.shape[0] != prediction.shape[1]:
        raise ValueError("atom_batch length must match the atom dimension")
    if atom_batch.device != prediction.device or truth.device != prediction.device:
        raise ValueError("coordinates and atom_batch must share a device")
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")

    complex_losses = []
    for complex_id in torch.unique(atom_batch):
        atom_indices = torch.nonzero(atom_batch == complex_id, as_tuple=False).flatten()
        atom_count = int(atom_indices.numel())
        if atom_count < 2:
            raise ValueError("each complex must contain at least two atoms")
        pair_i, pair_j = torch.triu_indices(
            atom_count,
            atom_count,
            offset=1,
            device=prediction.device,
        )
        selected_prediction = prediction[:, atom_indices, :]
        selected_truth = truth[:, atom_indices, :]
        predicted_distances = torch.linalg.vector_norm(
            selected_prediction[:, pair_i, :] - selected_prediction[:, pair_j, :],
            dim=-1,
        )
        true_distances = torch.linalg.vector_norm(
            selected_truth[:, pair_i, :] - selected_truth[:, pair_j, :],
            dim=-1,
        )
        complex_losses.append(
            functional.smooth_l1_loss(
                predicted_distances,
                true_distances,
                beta=float(beta),
                reduction="mean",
            )
        )
    if not complex_losses:
        raise ValueError("atom_batch must contain at least one complex")
    return torch.stack(complex_losses).mean()


def displacement_smooth_l1(prediction, truth, beta=0.5):
    """Match consecutive displacement vectors in an SE(3)-equivariant way."""
    if prediction.shape != truth.shape:
        raise ValueError("prediction and truth must have identical shapes")
    if prediction.ndim != 3 or prediction.shape[0] < 2 or prediction.shape[-1] != 3:
        raise ValueError("trajectories must have shape (frames>=2, atoms, 3)")
    if beta <= 0:
        raise ValueError("beta must be positive")
    predicted_displacement = prediction[1:] - prediction[:-1]
    true_displacement = truth[1:] - truth[:-1]
    return functional.smooth_l1_loss(
        predicted_displacement,
        true_displacement,
        beta=beta,
    )


def calibrated_auxiliary_weight(
    position_grad_norms,
    pair_grad_norms,
    target_fraction=0.1,
    epsilon=1e-12,
):
    """Return a frozen auxiliary weight from deterministic gradient ratios."""
    position = np.asarray(position_grad_norms, dtype=float)
    pair = np.asarray(pair_grad_norms, dtype=float)
    if position.ndim != 1 or pair.ndim != 1 or position.size == 0:
        raise ValueError("gradient norms must be nonempty one-dimensional sequences")
    if position.shape != pair.shape:
        raise ValueError("gradient norm sequences must have matching lengths")
    if not np.isfinite(position).all() or not np.isfinite(pair).all():
        raise ValueError("gradient norms must be finite")
    if np.any(position < 0) or np.any(pair < 0):
        raise ValueError("gradient norms cannot be negative")
    if not np.isfinite(target_fraction) or not 0 < target_fraction <= 1:
        raise ValueError("target_fraction must lie in (0, 1]")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    ratios = position / np.maximum(pair, epsilon)
    return float(target_fraction * np.median(ratios))
