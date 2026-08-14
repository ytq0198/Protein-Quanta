import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import h5py

from protein_quanta.streaming_misato import (
    StreamingMISATODataset,
    load_atomic_masses,
)


class StreamingMISATOTests(unittest.TestCase):
    def test_atomic_mass_table_requires_all_elements(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "periodic.csv"
            path.write_text(
                "AtomicNumber,AtomicMass\n" +
                "".join(f"{number},{float(number)}\n" for number in range(1, 119)),
                encoding="utf-8",
            )
            masses = load_atomic_masses(path)
        self.assertEqual(masses[1], 1.0)
        self.assertEqual(masses[118], 118.0)

    def test_dataset_filters_peptides_and_opens_hdf5_lazily(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            h5_path = root / "MD.hdf5"
            with h5py.File(h5_path, "w") as handle:
                handle.create_group("A")
                handle.create_group("B")
            split = root / "train.txt"
            split.write_text("a\nb\n", encoding="utf-8")
            peptides = root / "peptides.txt"
            peptides.write_text("b\n", encoding="utf-8")
            utils = root / "utils"
            utils.mkdir()
            for name in (
                "atoms_residue_map.pickle", "atoms_type_map.pickle",
                "atoms_name_map_for_pdb.pickle",
            ):
                with (utils / name).open("wb") as handle:
                    pickle.dump({}, handle)
            periodic = root / "periodic.csv"
            periodic.write_text(
                "AtomicNumber,AtomicMass\n" +
                "".join(f"{number},{float(number)}\n" for number in range(1, 119)),
                encoding="utf-8",
            )
            seen = []
            def parser(group, **resources):
                seen.append((group.name, resources["atom_num2atom_mass"][6]))
                return SimpleNamespace(name=group.name)
            dataset = StreamingMISATODataset(
                h5_path, split, peptides, utils, periodic, parser=parser
            )
            try:
                self.assertIsNone(dataset._h5)
                item = dataset[0]
                self.assertEqual(len(dataset), 1)
                self.assertEqual(dataset.sample_ids, ["A"])
                self.assertEqual(item.name, "/A")
                self.assertEqual(seen, [("/A", 6.0)])
                self.assertIsNotNone(dataset._h5)
                state = dataset.__getstate__()
                self.assertIsNone(state["_h5"])
            finally:
                dataset.close()
            self.assertIsNone(dataset._h5)


if __name__ == "__main__":
    unittest.main()
