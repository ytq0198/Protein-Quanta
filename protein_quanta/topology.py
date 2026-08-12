"""Traceable ligand-topology recovery from PDB records.

The helper only accepts atom-order mappings that exactly match the MISATO
heavy-element sequence.  It never infers chemical bonds from coordinates;
bonds must be present in PDB CONECT records.
"""

from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ELEMENT_NUMBERS = {
    "H": 1,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "NA": 11,
    "MG": 12,
    "SI": 14,
    "P": 15,
    "S": 16,
    "CL": 17,
    "K": 19,
    "CA": 20,
    "MN": 25,
    "FE": 26,
    "CU": 29,
    "ZN": 30,
    "SE": 34,
    "BR": 35,
    "I": 53,
}

WATER_RESIDUES = {"HOH", "WAT"}


@dataclass(frozen=True)
class PdbAtom:
    record: str
    serial: int
    name: str
    altloc: str
    residue: str
    chain: str
    residue_number: str
    insertion: str
    element: str
    coordinates: tuple

    @property
    def residue_key(self):
        return (self.chain, self.residue_number, self.insertion, self.residue)


def _element(line):
    value = line[76:78].strip().upper()
    if value:
        return value
    letters = "".join(character for character in line[12:16] if character.isalpha())
    if not letters:
        raise ValueError("PDB atom record has no element")
    return letters[0].upper()


def parse_pdb(path):
    """Return atoms and undirected CONECT edges from one PDB file."""
    atoms = []
    edges = set()
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            atoms.append(
                PdbAtom(
                    record=line[:6].strip(),
                    serial=int(line[6:11]),
                    name=line[12:16].strip(),
                    altloc=line[16:17].strip(),
                    residue=line[17:20].strip(),
                    chain=line[21:22].strip(),
                    residue_number=line[22:26].strip(),
                    insertion=line[26:27].strip(),
                    element=_element(line),
                    coordinates=(
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    ),
                )
            )
        elif line.startswith("CONECT"):
            values = []
            for start in range(6, len(line), 5):
                try:
                    values.append(int(line[start : start + 5]))
                except ValueError:
                    continue
            if values:
                for neighbour in values[1:]:
                    if neighbour != values[0]:
                        edges.add(tuple(sorted((values[0], neighbour))))
    return atoms, edges


def _altloc_views(atoms):
    labels = sorted({atom.altloc for atom in atoms if atom.altloc})
    if not labels:
        return [("blank", atoms)]
    return [
        (label, [atom for atom in atoms if not atom.altloc or atom.altloc == label])
        for label in labels
    ]


def _contiguous_windows(residues, target_heavy_count):
    for start in range(len(residues)):
        heavy_count = 0
        for end in range(start, len(residues)):
            heavy_count += sum(atom.element != "H" for atom in residues[end])
            if heavy_count == target_heavy_count:
                yield residues[start : end + 1]
            if heavy_count >= target_heavy_count:
                break


def _candidate_groups(atoms, target_heavy_count):
    by_residue = OrderedDict()
    for atom in atoms:
        if atom.residue in WATER_RESIDUES:
            continue
        by_residue.setdefault(atom.residue_key, []).append(atom)

    candidates = []
    # Single/multi-residue hetero ligands, including polysaccharides.
    for chain in sorted({key[0] for key in by_residue}):
        chain_residues = [
            value
            for key, value in by_residue.items()
            if key[0] == chain and all(atom.record == "HETATM" for atom in value)
        ]
        for window in _contiguous_windows(chain_residues, target_heavy_count):
            flattened = [atom for residue in window for atom in residue]
            for altloc, view in _altloc_views(flattened):
                candidates.append(("hetero", altloc, view))

    # A peptide-like ligand may contain both ATOM and HETATM residues.
    for chain in sorted({key[0] for key in by_residue}):
        chain_residues = [value for key, value in by_residue.items() if key[0] == chain]
        for window in _contiguous_windows(chain_residues, target_heavy_count):
            flattened = [atom for residue in window for atom in residue]
            for altloc, view in _altloc_views(flattened):
                candidates.append(("chain", altloc, view))
    return candidates


def _pairwise_distance_mae(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] != 3:
        raise ValueError("coordinate arrays must have matching shape (atoms, 3)")
    if left.shape[0] < 2:
        return 0.0
    pair_i, pair_j = np.triu_indices(left.shape[0], k=1)
    left_distances = np.linalg.norm(left[pair_i] - left[pair_j], axis=1)
    right_distances = np.linalg.norm(right[pair_i] - right[pair_j], axis=1)
    return float(np.mean(np.abs(left_distances - right_distances)))


def recover_ligand_topology(
    pdb_path,
    atomic_numbers,
    reference_coordinates=None,
    maximum_pairwise_distance_mae=2.0,
):
    """Recover a bond graph only when atom order and CONECT coverage agree.

    Multiple crystallographic copies are accepted only when their indexed bond
    topology is identical.  Coordinates are used to validate atom order, never
    to infer bonds.
    """
    atomic_numbers = np.asarray(atomic_numbers, dtype=int)
    if atomic_numbers.ndim != 1 or not atomic_numbers.size:
        raise ValueError("atomic_numbers must be a non-empty vector")
    heavy_numbers = atomic_numbers[atomic_numbers != 1]
    if reference_coordinates is not None:
        reference_coordinates = np.asarray(reference_coordinates, dtype=float)
        if reference_coordinates.shape != (heavy_numbers.size, 3):
            raise ValueError("reference_coordinates must match the heavy atoms")

    atoms, serial_edges = parse_pdb(pdb_path)
    accepted = []
    for source, altloc, candidate in _candidate_groups(atoms, heavy_numbers.size):
        heavy_atoms = [atom for atom in candidate if atom.element != "H"]
        candidate_numbers = np.array(
            [ELEMENT_NUMBERS.get(atom.element, -1) for atom in heavy_atoms], dtype=int
        )
        if not np.array_equal(candidate_numbers, heavy_numbers):
            continue
        serial_to_index = {atom.serial: index for index, atom in enumerate(heavy_atoms)}
        bonds = sorted(
            {
                tuple(sorted((serial_to_index[left], serial_to_index[right])))
                for left, right in serial_edges
                if left in serial_to_index and right in serial_to_index
            }
        )
        covered = {index for bond in bonds for index in bond}
        if len(covered) != heavy_numbers.size:
            continue
        coordinate_mae = None
        if reference_coordinates is not None:
            coordinate_mae = _pairwise_distance_mae(
                reference_coordinates,
                [atom.coordinates for atom in heavy_atoms],
            )
            if coordinate_mae > maximum_pairwise_distance_mae:
                continue
        accepted.append(
            {
                "source": source,
                "altloc": altloc,
                "residues": [
                    {
                        "name": key[3],
                        "chain": key[0],
                        "number": key[1],
                        "insertion": key[2],
                    }
                    for key in OrderedDict((atom.residue_key, None) for atom in heavy_atoms)
                ],
                "bonds": bonds,
                "pairwise_distance_mae_angstrom": coordinate_mae,
            }
        )

    signatures = {tuple(map(tuple, item["bonds"])) for item in accepted}
    if not accepted:
        return {
            "status": "unmatched_or_incomplete",
            "heavy_atom_count": int(heavy_numbers.size),
            "candidate_count": 0,
        }
    if len(signatures) != 1:
        return {
            "status": "ambiguous_topology",
            "heavy_atom_count": int(heavy_numbers.size),
            "candidate_count": len(accepted),
        }
    best = min(
        accepted,
        key=lambda item: (
            float("inf")
            if item["pairwise_distance_mae_angstrom"] is None
            else item["pairwise_distance_mae_angstrom"]
        ),
    )
    return {
        "status": "matched",
        "heavy_atom_count": int(heavy_numbers.size),
        "candidate_count": len(accepted),
        "equivalent_candidate_count": len(accepted),
        "bond_count": len(best["bonds"]),
        "bonds": best["bonds"],
        "mapping": {key: value for key, value in best.items() if key != "bonds"},
    }
