import unittest


class TopologyAuditContractTests(unittest.TestCase):
    def test_placeholder_for_cli_contract(self):
        # Full HDF5/PDB integration is exercised on the server dataset.  This
        # test keeps the module importable in the lightweight local suite.
        import scripts.audit_ligand_topologies  # noqa: F401


if __name__ == "__main__":
    unittest.main()
