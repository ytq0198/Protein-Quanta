import tempfile
import unittest
from pathlib import Path

import numpy as np

from protein_quanta.topology import recover_ligand_topology


PDB = """HETATM    1  C1  LIG A   1       0.000   0.000   0.000  1.00  0.00           C  
HETATM    2  N1  LIG A   1       1.300   0.000   0.000  1.00  0.00           N  
HETATM    3  O1  LIG A   1       2.500   0.000   0.000  1.00  0.00           O  
CONECT    1    2
CONECT    2    1    3
CONECT    3    2
END
"""


class TopologyRecoveryTests(unittest.TestCase):
    def test_recovers_exact_element_order_and_conect_graph(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.pdb"
            path.write_text(PDB, encoding="utf-8")
            result = recover_ligand_topology(
                path,
                np.array([6, 7, 8]),
                np.array([[0.0, 0.0, 0.0], [1.3, 0.0, 0.0], [2.5, 0.0, 0.0]]),
            )
        self.assertEqual(result["status"], "matched")
        self.assertEqual(result["bonds"], [(0, 1), (1, 2)])

    def test_rejects_element_order_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.pdb"
            path.write_text(PDB, encoding="utf-8")
            result = recover_ligand_topology(path, np.array([6, 8, 7]))
        self.assertEqual(result["status"], "unmatched_or_incomplete")


if __name__ == "__main__":
    unittest.main()
