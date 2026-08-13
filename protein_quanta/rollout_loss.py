"""Loss helpers for preregistered multiscale coordinate rollouts."""

import math
import random

import torch
import torch.nn.functional as functional


def validate_horizons(horizons, frame_count=None):
    """Return an immutable, duplicate-free sequence of positive horizons."""
    try:
        values = tuple(int(value) for value in horizons)
    except (TypeError, ValueError) as error:
        raise ValueError("horizons must be an iterable of integers") from error
    if not values or any(value <= 0 for value in values):
        raise ValueError("horizons must contain positive integers")
    if len(set(values)) != len(values):
        raise ValueError("horizons must not contain duplicates")
    if frame_count is not None:
        if int(frame_count) != frame_count or frame_count < 2:
            raise ValueError("frame_count must be an integer of at least two")
        if max(values) >= int(frame_count):
            raise ValueError("every horizon must be smaller than frame_count")
    return values


def sample_rollout_segment(frame_count, horizons, rng=None):
    """Sample a horizon uniformly, then a legal prefix uniformly.

    The returned interval is ``[start, end]`` and therefore contains
    ``horizon + 1`` states and exactly ``horizon`` supervised predictions.
    """
    values = validate_horizons(horizons, frame_count=frame_count)
    generator = random if rng is None else rng
    horizon = generator.choice(values)
    start = generator.randint(0, int(frame_count) - horizon - 1)
    return start, start + horizon, horizon


def multiscale_coordinate_smooth_l1(prediction, truth, beta=0.5):
    """Average coordinate Smooth-L1 over time, atoms and xyz dimensions."""
    if not isinstance(prediction, torch.Tensor) or not isinstance(truth, torch.Tensor):
        raise ValueError("prediction and truth must be torch tensors")
    if prediction.shape != truth.shape:
        raise ValueError("prediction and truth must have identical shapes")
    if prediction.ndim != 3 or prediction.shape[0] < 1 or prediction.shape[-1] != 3:
        raise ValueError("coordinates must have shape (frames>=1, atoms, 3)")
    if prediction.device != truth.device:
        raise ValueError("prediction and truth must share a device")
    if not math.isfinite(float(beta)) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    return functional.smooth_l1_loss(
        prediction,
        truth,
        beta=float(beta),
        reduction="mean",
    )


def combined_rollout_loss(local_loss, multiscale_loss, coefficient=0.25):
    """Add the candidate loss without perturbing the zero-weight baseline path."""
    if not isinstance(local_loss, torch.Tensor) or local_loss.ndim != 0:
        raise ValueError("local_loss must be a scalar torch tensor")
    if not math.isfinite(float(coefficient)) or coefficient < 0:
        raise ValueError("coefficient must be finite and non-negative")
    if coefficient == 0:
        return local_loss
    if not isinstance(multiscale_loss, torch.Tensor) or multiscale_loss.ndim != 0:
        raise ValueError("multiscale_loss must be a scalar torch tensor")
    if multiscale_loss.device != local_loss.device:
        raise ValueError("loss terms must share a device")
    return local_loss + float(coefficient) * multiscale_loss
