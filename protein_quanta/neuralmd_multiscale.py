"""NeuralMD-specific multiscale ODE rollout helpers kept outside upstream code."""

import torch

from protein_quanta.rollout_loss import (
    multiscale_coordinate_smooth_l1,
    sample_rollout_segment,
)


def neuralmd_condition(batch):
    """Build the immutable semi-flexible condition tuple used by NeuralMD."""
    return (
        batch.ligand_x,
        batch.batch_ligand,
        batch.ligand_mass,
        batch.protein_pos[batch.mask_n],
        batch.protein_pos[batch.mask_ca],
        batch.protein_pos[batch.mask_c],
        batch.protein_backbone_residue,
        batch.batch_residue,
    )


def neuralmd_ode_rollout(
    binding_model,
    odeint,
    batch,
    start,
    horizon,
    scaling=100.0,
    step_size=0.05,
    method="euler",
    condition=None,
    use_mlp_velocity=False,
    initial_velocity_scale=1.0,
    initial_position=None,
):
    """Integrate one differentiable semi-flexible ligand trajectory segment.

    ``initial_velocity_scale=1`` preserves the upstream NeuralMD multi-trajectory
    convention.  When one trajectory frame represents one unit of physical time
    but the ODE grid is divided by ``scaling``, dimensional consistency instead
    requires ``initial_velocity_scale=scaling``.  The explicit argument keeps
    those two protocols auditable rather than silently changing old results.
    """
    frame_count = int(batch.ligand_trajectory_pos.shape[1])
    if start < 0 or horizon <= 0 or start + horizon >= frame_count:
        raise ValueError("requested rollout segment lies outside the trajectory")
    if scaling <= 0 or step_size <= 0 or initial_velocity_scale <= 0:
        raise ValueError(
            "scaling, step_size, and initial_velocity_scale must be positive"
        )
    condition = neuralmd_condition(batch) if condition is None else condition
    position = (
        batch.ligand_trajectory_pos[:, start, :].clone()
        if initial_position is None
        else initial_position
    )
    velocity = (
        batch.ligand_trajectory_pos[:, start + 1, :] - position
    ) * float(initial_velocity_scale)
    if use_mlp_velocity:
        _, velocity = binding_model.velocity_model(
            z=condition[0], pos=velocity, batch=condition[1]
        )
    times = torch.arange(
        horizon + 1,
        dtype=position.dtype,
        device=position.device,
    ) / float(scaling)
    output_velocity, output_position = odeint(
        binding_model,
        (velocity, position),
        times,
        condition=condition,
        method=method,
        options={"step_size": float(step_size)},
    )
    return output_velocity, output_position


def sampled_multiscale_neuralmd_loss(
    binding_model,
    odeint,
    batch,
    horizons=(5, 10, 20, 40),
    beta=0.5,
    scaling=100.0,
    step_size=0.05,
    method="euler",
    condition=None,
    use_mlp_velocity=False,
    initial_velocity_scale=1.0,
    rng=None,
):
    """Sample one frozen time scale and return its mean-normalized loss."""
    start, end, horizon = sample_rollout_segment(
        int(batch.ligand_trajectory_pos.shape[1]), horizons, rng=rng
    )
    _, prediction = neuralmd_ode_rollout(
        binding_model=binding_model,
        odeint=odeint,
        batch=batch,
        start=start,
        horizon=horizon,
        scaling=scaling,
        step_size=step_size,
        method=method,
        condition=condition,
        use_mlp_velocity=use_mlp_velocity,
        initial_velocity_scale=initial_velocity_scale,
    )
    truth = batch.ligand_trajectory_pos[:, start + 1 : end + 1, :].transpose(0, 1)
    loss = multiscale_coordinate_smooth_l1(prediction[1:], truth, beta=beta)
    return loss, {"start": start, "end": end, "horizon": horizon}
