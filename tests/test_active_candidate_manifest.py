import json
import unittest
from pathlib import Path


class ActiveCandidateManifestTests(unittest.TestCase):
    def test_active_candidate_is_unanchored_and_does_not_claim_official_score(self):
        path = Path(__file__).resolve().parents[1] / "configs" / "active_candidate.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))

        self.assertFalse(manifest["active_for_submission"])
        self.assertEqual(manifest["checkpoint"]["epoch"], 5)
        self.assertEqual(manifest["inference"]["postprocessing"], "none")
        self.assertFalse(manifest["selection"]["official_score_claimed"])
        self.assertIn("updated-guide T1", manifest["promotion_state"]["overall"])
        self.assertEqual(
            manifest["updated_guide_protocol"]["T1"],
            {"observed": [0, 9], "initializer": [8, 9], "target": [10, 19]},
        )
        self.assertFalse(manifest["updated_guide_protocol"]["weights_published"])
        self.assertEqual(
            manifest["updated_guide_validation"]["status"],
            "rerun complete; epoch 5 is a research candidate but not a comprehensive winner",
        )
        self.assertIn(
            "published NeuralMD",
            manifest["updated_guide_validation"]["decision"],
        )


if __name__ == "__main__":
    unittest.main()
