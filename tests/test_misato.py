import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from protein_quanta.misato import audit_misato_h5, load_ligand_trajectory


class MisatoAuditTests(unittest.TestCase):
    def test_load_ligand_trajectory_matches_neuralmd_preprocessing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "ligand.hdf5"
            coordinates = np.arange(3 * 4 * 3, dtype=np.float32).reshape(3, 4, 3)
            with h5py.File(path, "w") as handle:
                group = handle.create_group("sample")
                group.create_dataset("trajectory_coordinates", data=coordinates)
                group.create_dataset("molecules_begin_atom_index", data=[0, 2])
                group.create_dataset("atoms_number", data=[6, 7, 1, 8])

            with h5py.File(path, "r") as handle:
                trajectory = load_ligand_trajectory(handle["sample"])

        center = coordinates.reshape(-1, 3).mean(axis=0)
        self.assertEqual(trajectory.sample_id, "sample")
        self.assertEqual(trajectory.coordinates.shape, (3, 1, 3))
        np.testing.assert_allclose(
            trajectory.coordinates[:, 0, :],
            coordinates[:, 3, :] - center,
        )

    def test_audit_summarizes_valid_complexes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tiny.hdf5"
            with h5py.File(path, "w") as handle:
                group = handle.create_group("1abc")
                group.create_dataset(
                    "trajectory_coordinates",
                    data=np.zeros((4, 3, 3), dtype=np.float32),
                )
                group.create_dataset("molecules_begin_atom_index", data=[0, 2])
                group.create_dataset("atoms_number", data=[6, 7, 8])
                group.create_dataset("atoms_type", data=[1, 2, 3])
                group.create_dataset("atoms_residue", data=[0, 0, 1])
                group.create_dataset("frames_interaction_energy", data=np.zeros(4))

            report = audit_misato_h5(path)

        self.assertEqual(report["complex_count"], 1)
        self.assertEqual(report["valid_complex_count"], 1)
        self.assertEqual(report["invalid_complex_count"], 0)
        self.assertEqual(report["frame_count_distribution"], {"4": 1})
        sample = report["samples"][0]
        self.assertEqual(sample["sample_id"], "1abc")
        self.assertEqual(sample["atom_count"], 3)
        self.assertEqual(sample["ligand_begin_index"], 2)
        self.assertEqual(sample["ligand_atom_count"], 1)
        self.assertTrue(sample["coordinates_finite"])

    def test_audit_reports_missing_fields_and_shape_errors(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "invalid.hdf5"
            with h5py.File(path, "w") as handle:
                group = handle.create_group("broken")
                group.create_dataset(
                    "trajectory_coordinates",
                    data=np.zeros((4, 3), dtype=np.float32),
                )

            report = audit_misato_h5(path)

        self.assertEqual(report["valid_complex_count"], 0)
        self.assertEqual(report["invalid_complex_count"], 1)
        errors = report["samples"][0]["errors"]
        self.assertTrue(any("missing required fields" in error for error in errors))
        self.assertIn("trajectory_coordinates must have shape (frames, atoms, 3)", errors)

    def test_audit_can_limit_number_of_samples(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "many.hdf5"
            with h5py.File(path, "w") as handle:
                for sample_id in ("a", "b"):
                    group = handle.create_group(sample_id)
                    group.create_dataset(
                        "trajectory_coordinates",
                        data=np.zeros((2, 1, 3), dtype=np.float32),
                    )
                    group.create_dataset("molecules_begin_atom_index", data=[0])
                    group.create_dataset("atoms_number", data=[6])
                    group.create_dataset("atoms_type", data=[1])
                    group.create_dataset("atoms_residue", data=[0])
                    group.create_dataset("frames_interaction_energy", data=np.zeros(2))

            report = audit_misato_h5(path, max_samples=1)

        self.assertEqual(report["complex_count"], 2)
        self.assertEqual(report["audited_complex_count"], 1)
        self.assertEqual(len(report["samples"]), 1)


if __name__ == "__main__":
    unittest.main()
