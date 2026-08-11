"""Numerically defensive optimization helpers for NeuralMD experiments."""

import math

import torch


def _gradient_norm(parameters):
    gradients = [
        parameter.grad.detach()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not gradients:
        return torch.tensor(0.0)
    norms = torch.stack(
        [torch.linalg.vector_norm(gradient, ord=2) for gradient in gradients]
    )
    return torch.linalg.vector_norm(norms, ord=2)


def stable_backward_step(loss, model, optimizer, max_grad_norm=0.0):
    """Backpropagate one scalar loss with optional global-norm clipping."""
    if not torch.is_tensor(loss) or loss.numel() != 1:
        raise ValueError("loss must be a scalar tensor")
    if max_grad_norm < 0 or not math.isfinite(float(max_grad_norm)):
        raise ValueError("max_grad_norm must be finite and non-negative")

    optimizer.zero_grad()
    loss_value = float(loss.detach().cpu())
    limit = float(max_grad_norm) if max_grad_norm > 0 else None
    if not math.isfinite(loss_value):
        return {
            "applied": False,
            "reason": "nonfinite_loss",
            "loss": loss_value,
            "grad_norm_before_clip": None,
            "max_grad_norm": limit,
            "clipped": False,
        }

    loss.backward()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if limit is None:
        grad_norm = _gradient_norm(parameters)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, limit)
    grad_norm_value = float(grad_norm.detach().cpu())
    if not math.isfinite(grad_norm_value):
        optimizer.zero_grad()
        return {
            "applied": False,
            "reason": "nonfinite_gradient",
            "loss": loss_value,
            "grad_norm_before_clip": grad_norm_value,
            "max_grad_norm": limit,
            "clipped": False,
        }

    optimizer.step()
    return {
        "applied": True,
        "reason": "applied",
        "loss": loss_value,
        "grad_norm_before_clip": grad_norm_value,
        "max_grad_norm": limit,
        "clipped": bool(limit is not None and grad_norm_value > limit),
    }
