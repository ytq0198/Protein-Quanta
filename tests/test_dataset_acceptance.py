import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from protein_quanta.dataset_acceptance import (
    audit_full_misato_acceptance,
    evenly_spaced_ids,
)


class DatasetAcceptanceTests(unittest.TestCase):
    def _write_ids(self, root, name, values):
        path = root / name
        path.write_text("\n".join(values) + "\n", encoding="utf-8")
        return path

    def _write_group(self, handle, sample_id):
        group = handle.create_group(sample_id)
        group.create_dataset(
            "trajectory_coordinates", data=np.zeros((2, 3, 3), dtype=np.float32)
        )
        group.create_dataset("molecules_begin_atom_index", data=[0, 2])
        group.create_dataset("atoms_number", data=[6, 7, 8])
        group.create_dataset("atoms_type", data=[1, 2, 3])
        group.create_dataset("atoms_residue", data=[0, 0, 1])
        group.create_dataset("frames_interaction_energy", data=np.zeros(2))

    def test_evenly_spaced_ids_is_deterministic_and_covers_endpoints(self):
        self.assertEqual(evenly_spaced_ids(["d", "b", "a", "c"], 3), ["A", "B", "D"])
        self.assertEqual(evenly_spaced_ids(["a", "b"], 3), ["A", "B"])
        with self.assertRaises(ValueError):
            evenly_spaced_ids(["a"], 0)

    def test_acceptance_links_filtered_splits_to_hdf5(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            split_paths = {
                "train": self._write_ids(root, "train.txt", ["1aaa", "1aab"]),
                "val": self._write_ids(root, "val.txt", ["2aaa"]),
                "test": self._write_ids(root, "test.txt", ["3aaa"]),
            }
            peptides = self._write_ids(root, "peptides.txt", ["1aab"])
            h5_path = root / "MD.hdf5"
            with h5py.File(h5_path, "w") as handle:
                for sample_id in ("1AAA", "1AAB", "2AAA", "3AAA"):
                    self._write_group(handle, sample_id)

            report = audit_full_misato_acceptance(
                h5_path,
                split_paths,
                peptides,
                samples_per_split=1,
                observed_md5="ABCD",
                expected_md5="abcd",
            )

        self.assertEqual(report["hdf5"]["group_count"], 4)
        self.assertEqual(
            report["membership_differences"]["raw_ids_missing_from_hdf5_count"], 0
        )
        self.assertTrue(report["checks"]["all_retained_ids_present"])
        self.assertTrue(report["checks"]["sampled_schema_and_finite_values_pass"])
        self.assertTrue(report["checks"]["md5_matches_published_value"])
        self.assertFalse(report["checks"]["updated_guide_counts_match"])
        self.assertFalse(report["checks"]["all_passed"])

    def test_digest_arguments_must_be_paired(self):
        with self.assertRaises(ValueError):
            audit_full_misato_acceptance(
                Path("missing.h5"), {}, Path("missing.txt"), observed_md5="abcd"
            )


if __name__ == "__main__":
    unittest.main()
