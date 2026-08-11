"""Project-defined covalent-radius collision diagnostics."""

import numpy as np


def _trajectory(value, name):
    value = np.asarray(value, dtype=float)
    if value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (frames, atoms, 3)")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must contain finite values")
    return value


def _radii(value, atom_count, name):
    value = np.asarray(value, dtype=float)
    if value.shape != (atom_count,):
        raise ValueError(f"{name} must have shape ({atom_count},)")
    if not np.isfinite(value).all() or np.any(value <= 0):
        raise ValueError(f"{name} must be finite and positive")
    return value


def intramolecular_collision_rate(trajectory, covalent_radii):
    """Return per-frame percentage of colliding unordered atom pairs."""
    trajectory = _trajectory(trajectory, "trajectory")
    radii = _radii(covalent_radii, trajectory.shape[1], "covalent_radii")
    if trajectory.shape[1] < 2:
        raise ValueError("intramolecular collisions require at least two atoms")
    pair_i, pair_j = np.triu_indices(trajectory.shape[1], k=1)
    distances = np.linalg.norm(
        trajectory[:, pair_i, :] - trajectory[:, pair_j, :], axis=-1
    )
    thresholds = radii[pair_i] + radii[pair_j]
    return 100.0 * np.mean(distances < thresholds[None, :], axis=1)


def binding_collision_rate(
    ligand_trajectory,
    ligand_covalent_radii,
    protein_positions,
    protein_covalent_radii,
):
    """Return per-frame percentage of colliding ligand-protein atom pairs."""
    ligand = _trajectory(ligand_trajectory, "ligand_trajectory")
    protein = np.asarray(protein_positions, dtype=float)
    if protein.ndim != 2 or protein.shape[-1] != 3:
        raise ValueError("protein_positions must have shape (atoms, 3)")
    if not np.isfinite(protein).all():
        raise ValueError("protein_positions must contain finite values")
    ligand_radii = _radii(
        ligand_covalent_radii, ligand.shape[1], "ligand_covalent_radii"
    )
    protein_radii = _radii(
        protein_covalent_radii, protein.shape[0], "protein_covalent_radii"
    )
    distances = np.linalg.norm(
        ligand[:, :, None, :] - protein[None, None, :, :], axis=-1
    )
    thresholds = ligand_radii[:, None] + protein_radii[None, :]
    return 100.0 * np.mean(distances < thresholds[None, :, :], axis=(1, 2))
