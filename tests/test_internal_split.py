import unittest

from protein_quanta.internal_split import deterministic_hash_split


class InternalSplitTests(unittest.TestCase):
    def test_split_is_deterministic_disjoint_and_order_preserving(self):
        identifiers = [f"ID{index}" for index in range(10)]
        first = deterministic_hash_split(identifiers, 2, "salt")
        second = deterministic_hash_split(identifiers, 2, "salt")
        self.assertEqual(first, second)
        development, holdout = first
        self.assertEqual(len(development), 8)
        self.assertEqual(len(holdout), 2)
        self.assertFalse(set(development) & set(holdout))
        self.assertEqual(development, [value for value in identifiers if value in development])

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            deterministic_hash_split(["A", "A"], 1, "salt")
        with self.assertRaises(ValueError):
            deterministic_hash_split(["A", "B"], 0, "salt")
        with self.assertRaises(ValueError):
            deterministic_hash_split(["A", "B"], 1, "")


if __name__ == "__main__":
    unittest.main()
