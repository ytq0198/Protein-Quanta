import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from protein_quanta.dense_equivariant_dynamics import DenseEquivariantAcceleration
from protein_quanta.effect_gate_barrier import verify_training_artifacts


class EffectGateBarrierTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.training = root / "training"
        self.checkpoints = root / "checkpoints"
        self.training.mkdir()
        self.checkpoints.mkdir()
        self.config = {"pairing": {"seeds": [0, 42, 123]}}

    def tearDown(self):
        self.temporary.cleanup()

    def write_seed(self, seed, corrupt=False):
        hashes = {}
        for arm in ("control", "candidate"):
            path = self.checkpoints / f"{arm}_seed_{seed}_final.pth"
            path.write_bytes(f"{arm}-{seed}".encode())
            hashes[arm] = hashlib.sha256(path.read_bytes()).hexdigest()
        if corrupt:
            hashes["candidate"] = "0" * 64
        (self.training / f"seed_{seed}.json").write_text(json.dumps({
            "status": "training_complete_holdout_unread",
            "seed": seed,
            "checkpoint_sha256": hashes,
        }), encoding="utf-8")

    def test_rejects_incomplete_six_checkpoint_set(self):
        self.write_seed(0)
        with self.assertRaises(FileNotFoundError):
            verify_training_artifacts(
                self.config, self.training, self.checkpoints
            )

    def test_rejects_hash_mismatch(self):
        self.write_seed(0)
        self.write_seed(42, corrupt=True)
        self.write_seed(123)
        with self.assertRaises(ValueError):
            verify_training_artifacts(
                self.config, self.training, self.checkpoints
            )

    def test_accepts_complete_verified_set(self):
        for seed in self.config["pairing"]["seeds"]:
            self.write_seed(seed)
        verified = verify_training_artifacts(
            self.config, self.training, self.checkpoints
        )
        self.assertEqual(set(verified), {0, 42, 123})


class EffectGatePairingTests(unittest.TestCase):
    def test_deepcopied_arms_start_bitwise_identical_and_independent(self):
        import copy

        torch.manual_seed(42)
        initial = DenseEquivariantAcceleration(hidden_dim=32)
        control = copy.deepcopy(initial)
        candidate = copy.deepcopy(initial)

        for left, right in zip(control.parameters(), candidate.parameters()):
            self.assertTrue(torch.equal(left, right))
        with torch.no_grad():
            next(control.parameters()).add_(1.0)
        self.assertFalse(torch.equal(
            next(control.parameters()), next(candidate.parameters())
        ))


if __name__ == "__main__":
    unittest.main()
