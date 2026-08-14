"""Lazy, worker-safe access to full MISATO without an all-complex PyG cache."""

from __future__ import annotations

import csv
import pickle
from pathlib import Path

import h5py
from torch.utils.data import Dataset


ATOM_INDEX_TO_NAME = {
    1: "H", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 11: "Na",
    12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl",
    19: "K", 20: "Ca", 34: "Se", 35: "Br", 53: "I",
}


def read_ids(path: Path) -> list[str]:
    values = [
        line.strip().upper()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not values:
        raise ValueError(f"empty ID file: {path}")
    return values


def load_atomic_masses(periodic_table_path: Path) -> dict[int, float]:
    with Path(periodic_table_path).open(encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        masses = {
            int(row["AtomicNumber"]): float(row["AtomicMass"])
            for row in rows
            if row["AtomicMass"]
        }
    missing = sorted(set(range(1, 119)) - set(masses))
    if missing:
        raise ValueError(f"periodic table is missing atomic masses: {missing}")
    return masses


class StreamingMISATODataset(Dataset):
    """Parse one complex at a time using NeuralMD's official preprocessing.

    The HDF5 handle is opened lazily in each process.  This makes the object safe
    to construct before a PyTorch DataLoader forks or spawns workers and avoids
    serializing an h5py handle.
    """

    def __init__(
        self,
        h5_path: Path,
        split_path: Path,
        peptides_path: Path,
        neuralmd_utils_dir: Path,
        periodic_table_path: Path,
        parser=None,
    ):
        self.h5_path = Path(h5_path)
        self.split_path = Path(split_path)
        self.peptides_path = Path(peptides_path)
        self.neuralmd_utils_dir = Path(neuralmd_utils_dir)
        self.periodic_table_path = Path(periodic_table_path)
        peptides = set(read_ids(self.peptides_path))
        self.sample_ids = [
            sample_id for sample_id in read_ids(self.split_path)
            if sample_id not in peptides
        ]
        if not self.sample_ids:
            raise ValueError("peptide filtering removed the entire split")
        self._parser = parser
        self._h5 = None
        self._resources = None

    def __len__(self):
        return len(self.sample_ids)

    def _load_resources(self):
        if self._resources is None:
            with (self.neuralmd_utils_dir / "atoms_residue_map.pickle").open("rb") as handle:
                residue_map = pickle.load(handle)
            with (self.neuralmd_utils_dir / "atoms_type_map.pickle").open("rb") as handle:
                atom_type_map = pickle.load(handle)
            with (self.neuralmd_utils_dir / "atoms_name_map_for_pdb.pickle").open("rb") as handle:
                atom_name_map = pickle.load(handle)
            self._resources = {
                "atom_index2name_dict": ATOM_INDEX_TO_NAME,
                "atom_num2atom_mass": load_atomic_masses(self.periodic_table_path),
                "residue_index2name_dict": residue_map,
                "protein_atom_index2standard_name_dict": atom_type_map,
                "atom_reisdue2standard_atom_name_dict": atom_name_map,
            }
        return self._resources

    def _handle(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def _resolve_parser(self):
        if self._parser is not None:
            return self._parser
        from NeuralMD.datasets.MISATO.dataset_MISATO_semi_flexible import (
            parse_MISATO_data,
        )
        return parse_MISATO_data

    def __getitem__(self, index):
        sample_id = self.sample_ids[index]
        group = self._handle().get(sample_id)
        if group is None:
            raise KeyError(f"split ID absent from HDF5: {sample_id}")
        data = self._resolve_parser()(group, **self._load_resources())
        data.sample_id = sample_id
        return data

    def close(self):
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def __del__(self):
        self.close()
