import json
import unittest
from pathlib import Path


class FrozenCandidateManifestTests(unittest.TestCase):
    def test_manifest_freezes_reproducible_selection_and_inference(self):
        path = Path("configs/frozen_candidate.json")
        manifest = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(manifest["method"], "scenario-aware-earlystop-static-anchor")
        self.assertEqual(manifest["checkpoint"]["epoch"], 5)
        self.assertRegex(manifest["checkpoint"]["sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(manifest["postprocessing"]["beta"], 1.0)
        self.assertEqual(manifest["postprocessing"]["decay_scale_frames"], 98.0)
        self.assertEqual(
            manifest["selection"]["split"],
            "MISATO-100 validation (10 complexes)",
        )
        self.assertFalse(manifest["selection"]["official_score_claimed"])
        self.assertEqual(manifest["competition_weights"]["T1_T2_T3"], [0.5, 0.3, 0.2])
        self.assertFalse(manifest["active_for_submission"])
        self.assertIn("bond-aware validation", manifest["demotion"]["reason"])


if __name__ == "__main__":
    unittest.main()
