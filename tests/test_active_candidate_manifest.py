import json
import unittest
from pathlib import Path


class ActiveCandidateManifestTests(unittest.TestCase):
    def test_active_candidate_is_unanchored_and_does_not_claim_official_score(self):
        path = Path(__file__).resolve().parents[1] / "configs" / "active_candidate.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))

        self.assertTrue(manifest["active_for_submission"])
        self.assertEqual(manifest["checkpoint"]["epoch"], 5)
        self.assertEqual(manifest["inference"]["postprocessing"], "none")
        self.assertFalse(manifest["selection"]["official_score_claimed"])
        self.assertIn("not yet eligible", manifest["promotion_state"]["overall"])


if __name__ == "__main__":
    unittest.main()
