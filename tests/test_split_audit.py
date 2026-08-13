import tempfile
import unittest
from pathlib import Path

from protein_quanta.split_audit import audit_filtered_splits


class SplitAuditTests(unittest.TestCase):
    def _write(self, root, name, values):
        path = root / name
        path.write_text("\n".join(values) + "\n", encoding="utf-8")
        return path

    def test_counts_exclusions_and_overlap_are_explicit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = {
                "train": self._write(root, "train.txt", ["1abc", "2def"]),
                "val": self._write(root, "val.txt", ["3ghi"]),
                "test": self._write(root, "test.txt", ["4jkl"]),
            }
            peptides = self._write(root, "peptides.txt", ["2DEF", "9xyz"])
            report = audit_filtered_splits(paths, peptides)

        self.assertEqual(report["splits"]["train"]["retained_count"], 1)
        self.assertEqual(report["splits"]["train"]["excluded_peptide_count"], 1)
        self.assertEqual(report["peptide_ids_present_in_any_split"], 1)
        self.assertEqual(report["peptide_ids_outside_all_splits"], 1)
        self.assertTrue(report["checks"]["splits_disjoint"])
        self.assertFalse(report["checks"]["all_passed"])

    def test_duplicates_and_cross_split_overlap_fail_checks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = {
                "train": self._write(root, "train.txt", ["1abc", "1ABC"]),
                "val": self._write(root, "val.txt", ["1abc"]),
                "test": self._write(root, "test.txt", ["2def"]),
            }
            peptides = self._write(root, "peptides.txt", ["9xyz"])
            report = audit_filtered_splits(paths, peptides)

        self.assertFalse(report["checks"]["no_duplicate_rows"])
        self.assertFalse(report["checks"]["splits_disjoint"])
        self.assertEqual(report["cross_split_overlap"]["train_val"]["count"], 1)

    def test_requires_exact_split_names(self):
        with self.assertRaises(ValueError):
            audit_filtered_splits({}, Path("missing"))


if __name__ == "__main__":
    unittest.main()
