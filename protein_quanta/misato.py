"""Lightweight schema auditing for MISATO molecular-dynamics HDF5 files."""

from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import h5py
import numpy as np

from protein_quanta.trajectory import Trajectory


REQUIRED_MD_FIELDS = (
    "trajectory_coordinates",
    "molecules_begin_atom_index",
    "atoms_number",
    "atoms_type",
    "atoms_residue",
    "frames_interaction_energy",
)


def load_ligand_trajectory(
    group: h5py.Group,
    heavy_atoms_only: bool = True,
) -> Trajectory:
    """Load one ligand trajectory using NeuralMD's centering and H filtering."""
    coordinates = np.asarray(group["trajectory_coordinates"])
    if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
        raise ValueError("trajectory_coordinates must have shape (frames, atoms, 3)")

    molecule_starts = np.asarray(group["molecules_begin_atom_index"])
    if molecule_starts.size == 0:
        raise ValueError("molecules_begin_atom_index must not be empty")
    ligand_begin_index = int(molecule_starts[-1])
    if not 0 <= ligand_begin_index <= coordinates.shape[1]:
        raise ValueError("ligand begin index is outside the atom range")

    center = coordinates.reshape(-1, 3).mean(axis=0)
    ligand_coordinates = coordinates[:, ligand_begin_index:, :] - center
    if heavy_atoms_only:
        atomic_numbers = np.asarray(group["atoms_number"])[ligand_begin_index:]
        ligand_coordinates = ligand_coordinates[:, atomic_numbers != 1, :]
    if ligand_coordinates.shape[1] == 0:
        raise ValueError("ligand trajectory contains no selected atoms")

    return Trajectory(
        sample_id=group.name.rsplit("/", 1)[-1],
        coordinates=ligand_coordinates,
    )


def _coordinate_statistics(dataset: h5py.Dataset, chunk_frames: int) -> Dict[str, Any]:
    finite = True
    minimum = np.inf
    maximum = -np.inf
    for start in range(0, dataset.shape[0], chunk_frames):
        values = np.asarray(dataset[start : start + chunk_frames])
        finite = finite and bool(np.isfinite(values).all())
        if values.size:
            minimum = min(minimum, float(np.nanmin(values)))
            maximum = max(maximum, float(np.nanmax(values)))
    return {
        "coordinates_finite": finite,
        "coordinate_min": None if np.isinf(minimum) else minimum,
        "coordinate_max": None if np.isinf(maximum) else maximum,
    }


def _audit_group(
    sample_id: str,
    group: h5py.Group,
    chunk_frames: int,
) -> Dict[str, Any]:
    dataset_metadata = {
        name: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in group.items()
        if isinstance(value, h5py.Dataset)
    }
    result: Dict[str, Any] = {
        "sample_id": sample_id,
        "datasets": dataset_metadata,
        "errors": [],
    }
    errors = result["errors"]

    missing = [name for name in REQUIRED_MD_FIELDS if name not in group]
    if missing:
        errors.append("missing required fields: " + ", ".join(missing))

    coordinates = group.get("trajectory_coordinates")
    if not isinstance(coordinates, h5py.Dataset) or (
        coordinates.ndim != 3 or coordinates.shape[-1] != 3
    ):
        errors.append("trajectory_coordinates must have shape (frames, atoms, 3)")
        result["valid"] = False
        return result

    frame_count, atom_count, _ = coordinates.shape
    result.update(
        {
            "frame_count": int(frame_count),
            "atom_count": int(atom_count),
        }
    )
    result.update(_coordinate_statistics(coordinates, chunk_frames))
    if not result["coordinates_finite"]:
        errors.append("trajectory_coordinates contains non-finite values")

    for field in ("atoms_number", "atoms_type", "atoms_residue"):
        dataset = group.get(field)
        if isinstance(dataset, h5py.Dataset) and dataset.shape != (atom_count,):
            errors.append(f"{field} must have shape ({atom_count},)")

    energies = group.get("frames_interaction_energy")
    if isinstance(energies, h5py.Dataset) and energies.shape != (frame_count,):
        errors.append(f"frames_interaction_energy must have shape ({frame_count},)")

    molecule_starts = group.get("molecules_begin_atom_index")
    if isinstance(molecule_starts, h5py.Dataset) and molecule_starts.size:
        ligand_begin_index = int(np.asarray(molecule_starts)[-1])
        result["ligand_begin_index"] = ligand_begin_index
        result["ligand_atom_count"] = atom_count - ligand_begin_index
        if not 0 <= ligand_begin_index <= atom_count:
            errors.append("ligand begin index is outside the atom range")
    elif isinstance(molecule_starts, h5py.Dataset):
        errors.append("molecules_begin_atom_index must not be empty")

    result["valid"] = not errors
    return result


def audit_misato_h5(
    path: Union[str, Path],
    max_samples: Optional[int] = None,
    chunk_frames: int = 16,
    selected_sample_ids: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Audit MISATO structure and values without loading the whole file at once."""
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive")
    if max_samples is not None and selected_sample_ids is not None:
        raise ValueError("max_samples and selected_sample_ids are mutually exclusive")
    if chunk_frames <= 0:
        raise ValueError("chunk_frames must be positive")

    path = Path(path)
    with h5py.File(path, "r") as handle:
        sample_ids = sorted(
            name for name, value in handle.items() if isinstance(value, h5py.Group)
        )
        if selected_sample_ids is None:
            selected_ids = sample_ids[:max_samples]
        else:
            selected_ids = [str(sample_id).upper() for sample_id in selected_sample_ids]
            missing = sorted(set(selected_ids) - set(sample_ids))
            if missing:
                raise KeyError("selected sample IDs absent from HDF5: " + ", ".join(missing))
        samples = [
            _audit_group(sample_id, handle[sample_id], chunk_frames)
            for sample_id in selected_ids
        ]

    frame_counts = Counter(
        str(sample["frame_count"])
        for sample in samples
        if "frame_count" in sample
    )
    valid_count = sum(bool(sample["valid"]) for sample in samples)
    return {
        "path": str(path.resolve()),
        "complex_count": len(sample_ids),
        "audited_complex_count": len(samples),
        "valid_complex_count": valid_count,
        "invalid_complex_count": len(samples) - valid_count,
        "frame_count_distribution": dict(sorted(frame_counts.items())),
        "samples": samples,
    }
