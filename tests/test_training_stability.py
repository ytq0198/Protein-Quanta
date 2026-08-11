import unittest

import torch

from protein_quanta.training_stability import stable_backward_step


class StableBackwardStepTests(unittest.TestCase):
    def test_large_finite_gradient_is_clipped_and_update_stays_finite(self):
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loss = (model(torch.tensor([[1000.0]])) ** 2).mean()

        result = stable_backward_step(
            loss, model, optimizer, max_grad_norm=1.0
        )

        self.assertTrue(result["applied"])
        self.assertTrue(result["clipped"])
        self.assertGreater(result["grad_norm_before_clip"], 1.0)
        self.assertTrue(torch.isfinite(model.weight).all())
        self.assertAlmostEqual(model.weight.item(), 0.9, places=5)

    def test_nonfinite_loss_skips_update_without_changing_parameters(self):
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        before = model.weight.detach().clone()
        loss = model.weight.sum() * torch.tensor(float("nan"))

        result = stable_backward_step(
            loss, model, optimizer, max_grad_norm=1.0
        )

        self.assertFalse(result["applied"])
        self.assertEqual(result["reason"], "nonfinite_loss")
        torch.testing.assert_close(model.weight, before)

    def test_disabled_clipping_reports_unclipped_update(self):
        model = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        loss = (model(torch.tensor([[2.0]])) ** 2).mean()

        result = stable_backward_step(
            loss, model, optimizer, max_grad_norm=0.0
        )

        self.assertTrue(result["applied"])
        self.assertFalse(result["clipped"])
        self.assertIsNone(result["max_grad_norm"])


if __name__ == "__main__":
    unittest.main()
