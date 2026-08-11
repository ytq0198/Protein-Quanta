"""Typed trajectory records shared by data, models, and metrics."""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Trajectory:
    sample_id: str
    coordinates: np.ndarray
    atom_mask: Optional[np.ndarray] = None
    unit: str = "angstrom"

    def __post_init__(self):
        coordinates = np.asarray(self.coordinates)
        if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
            raise ValueError("coordinates must have shape (frames, atoms, 3)")
        if not np.isfinite(coordinates).all():
            raise ValueError("coordinates must be finite")

        mask = self.atom_mask
        if mask is None:
            mask = np.ones(coordinates.shape[1], dtype=bool)
        else:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != (coordinates.shape[1],):
                raise ValueError(
                    f"atom_mask must have shape ({coordinates.shape[1]},)"
                )

        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "atom_mask", mask)
