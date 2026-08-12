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


def _bond_array(bonds, atom_count):
    bonds = np.asarray(bonds, dtype=int)
    if bonds.ndim != 2 or bonds.shape[1:] != (2,):
        raise ValueError("bonds must have shape (bonds, 2)")
    if bonds.size and (np.any(bonds < 0) or np.any(bonds >= atom_count)):
        raise ValueError("bond index is outside the atom range")
    if bonds.size and np.any(bonds[:, 0] == bonds[:, 1]):
        raise ValueError("self bonds are not allowed")
    return np.sort(bonds, axis=1)


def nonbonded_intramolecular_collision_rate(
    trajectory,
    covalent_radii,
    bonds,
    exclude_graph_distance=2,
):
    """Return definite clashes after excluding close covalent neighbours.

    A collision uses the conservative covalent-radius sum.  Graph distance two
    is excluded by default so normal bond angles are not counted as clashes.
    """
    trajectory = _trajectory(trajectory, "trajectory")
    radii = _radii(covalent_radii, trajectory.shape[1], "covalent_radii")
    bonds = _bond_array(bonds, trajectory.shape[1])
    if exclude_graph_distance not in (1, 2):
        raise ValueError("exclude_graph_distance must be 1 or 2")
    excluded = {tuple(pair) for pair in bonds.tolist()}
    if exclude_graph_distance == 2:
        neighbours = [set() for _ in range(trajectory.shape[1])]
        for left, right in bonds:
            neighbours[left].add(int(right))
            neighbours[right].add(int(left))
        for center in range(trajectory.shape[1]):
            adjacent = sorted(neighbours[center])
            for left_index in range(len(adjacent)):
                for right_index in range(left_index + 1, len(adjacent)):
                    excluded.add((adjacent[left_index], adjacent[right_index]))
    pair_i, pair_j = np.triu_indices(trajectory.shape[1], k=1)
    keep = np.array(
        [(int(left), int(right)) not in excluded for left, right in zip(pair_i, pair_j)],
        dtype=bool,
    )
    if not np.any(keep):
        raise ValueError("no nonbonded atom pairs remain")
    pair_i, pair_j = pair_i[keep], pair_j[keep]
    distances = np.linalg.norm(
        trajectory[:, pair_i, :] - trajectory[:, pair_j, :], axis=-1
    )
    thresholds = radii[pair_i] + radii[pair_j]
    return 100.0 * np.mean(distances < thresholds[None, :], axis=1)


def bond_length_diagnostics(prediction, truth, bonds, violation_fraction=0.20):
    """Compare predicted bond lengths with same-frame reference lengths."""
    prediction = _trajectory(prediction, "prediction")
    truth = _trajectory(truth, "truth")
    if prediction.shape != truth.shape:
        raise ValueError("prediction and truth must have identical shape")
    if not 0 < violation_fraction < 1:
        raise ValueError("violation_fraction must be between zero and one")
    bonds = _bond_array(bonds, prediction.shape[1])
    if not bonds.shape[0]:
        raise ValueError("at least one bond is required")
    left, right = bonds[:, 0], bonds[:, 1]
    predicted_lengths = np.linalg.norm(
        prediction[:, left, :] - prediction[:, right, :], axis=-1
    )
    truth_lengths = np.linalg.norm(truth[:, left, :] - truth[:, right, :], axis=-1)
    if np.any(truth_lengths <= 0):
        raise ValueError("truth contains a zero-length bond")
    absolute_error = np.abs(predicted_lengths - truth_lengths)
    relative_error = absolute_error / truth_lengths
    extreme = (predicted_lengths < 0.5 * truth_lengths) | (
        predicted_lengths > 1.5 * truth_lengths
    )
    return {
        "bond_length_mae_angstrom": float(np.mean(absolute_error)),
        "bond_length_relative_mae": float(np.mean(relative_error)),
        "bond_length_violation_percent": float(
            100.0 * np.mean(relative_error > violation_fraction)
        ),
        "extreme_bond_event_percent": float(100.0 * np.mean(extreme)),
    }


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
