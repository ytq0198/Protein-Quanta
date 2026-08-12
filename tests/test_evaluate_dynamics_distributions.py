import unittest

from scripts.evaluate_dynamics_distributions import _parse_models


class EvaluateDynamicsDistributionTests(unittest.TestCase):
    def test_model_specs_require_unique_name_directory_pairs(self):
        models = _parse_models(["epoch5=/tmp/a", "published=/tmp/b"])
        self.assertEqual(set(models), {"epoch5", "published"})
        with self.assertRaises(ValueError):
            _parse_models(["epoch5=/tmp/a", "epoch5=/tmp/b"])
        with self.assertRaises(ValueError):
            _parse_models(["missing-separator"])


if __name__ == "__main__":
    unittest.main()
