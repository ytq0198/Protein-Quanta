import json
import unittest
from pathlib import Path


class E17PreregistrationTests(unittest.TestCase):
    def test_preregistration_is_single_value_validation_only(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "e17_displacement_preregistration.json"
        )
        config = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(config["training"]["epochs"], 5)
        self.assertEqual(config["training"]["displacement_loss_coefficient"], 1.0)
        self.assertEqual(config["decision_split"], "MISATO-100 validation only")
        self.assertIn("prohibited", config["test_access"])
        self.assertIn("do not tune", config["stop_rule"])


if __name__ == "__main__":
    unittest.main()
