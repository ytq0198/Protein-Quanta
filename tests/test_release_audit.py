import json
import tempfile
import unittest
from pathlib import Path

from protein_quanta.release_audit import audit_release


class ReleaseAuditTests(unittest.TestCase):
    def _workspace(self):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        (root / "configs").mkdir()
        (root / "reports").mkdir()
        manifest = {
            "status": "proxy metrics only; not an official competition score",
            "selection": {"official_score_claimed": False},
            "physical_diagnostics": {"official_score_claimed": False},
            "evidence": {"validation_report": "reports/validation.json"},
        }
        (root / "configs" / "frozen_candidate.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        (root / "reports" / "validation.json").write_text("{}", encoding="utf-8")
        return temporary, root

    def test_clean_release_passes_with_license_warning(self):
        temporary, root = self._workspace()
        self.addCleanup(temporary.cleanup)

        report = audit_release(
            root,
            tracked_paths=[
                "configs/frozen_candidate.json",
                "reports/validation.json",
            ],
        )

        self.assertTrue(report["passed"])
        self.assertEqual(report["errors"], [])
        self.assertTrue(any("LICENSE" in value for value in report["warnings"]))

    def test_rejects_tracked_model_data_and_secret_patterns(self):
        temporary, root = self._workspace()
        self.addCleanup(temporary.cleanup)
        (root / "model.pth").write_bytes(b"weights")
        (root / "notes.txt").write_text(
            "github" + "_pat_" + "example", encoding="utf-8"
        )

        report = audit_release(
            root,
            tracked_paths=["model.pth", "notes.txt", "configs/frozen_candidate.json"],
        )

        self.assertFalse(report["passed"])
        self.assertTrue(any("model.pth" in value for value in report["errors"]))
        self.assertTrue(any("secret-like" in value for value in report["errors"]))

    def test_rejects_missing_evidence_and_official_score_claim(self):
        temporary, root = self._workspace()
        self.addCleanup(temporary.cleanup)
        manifest_path = root / "configs" / "frozen_candidate.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["selection"]["official_score_claimed"] = True
        manifest["evidence"]["validation_report"] = "reports/missing.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        report = audit_release(root, tracked_paths=["configs/frozen_candidate.json"])

        self.assertFalse(report["passed"])
        self.assertTrue(any("official score" in value for value in report["errors"]))
        self.assertTrue(any("missing.json" in value for value in report["errors"]))


if __name__ == "__main__":
    unittest.main()
