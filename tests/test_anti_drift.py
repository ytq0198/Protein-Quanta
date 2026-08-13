import unittest

import torch

from protein_quanta.anti_drift import (
    BridgeInterpolator,
    EndpointForecaster,
    block_rollout,
    parameter_count,
)


class AntiDriftTests(unittest.TestCase):
    def test_network_contract_and_parameter_cap(self):
        start = torch.randn(3, 12)
        endpoint = torch.randn(3, 12)
        step = torch.tensor([0.0, 1.0, 2.0])
        horizon = torch.tensor([2.0, 3.0, 4.0])
        interpolator = BridgeInterpolator(12)
        forecaster = EndpointForecaster(12)
        self.assertEqual(interpolator(start, endpoint, step, horizon).shape, start.shape)
        self.assertEqual(forecaster(start, start, step, horizon).shape, start.shape)
        self.assertLess(parameter_count(interpolator), 10000)
        self.assertLess(parameter_count(forecaster), 10000)

    def test_interpolator_preserves_start_boundary(self):
        start = torch.randn(2, 12)
        endpoint = torch.randn(2, 12)
        model = BridgeInterpolator(12)
        torch.testing.assert_close(model(start, endpoint, 0, 4), start)

    def test_alternation_changes_only_call_order_not_weights(self):
        class LinearInterpolator:
            def __call__(self, start, endpoint, step, horizon):
                ratio = (step / horizon).unsqueeze(-1)
                return start + ratio * (endpoint - start)

        class AddOneForecaster:
            def __call__(self, intermediate, start, step, horizon):
                return intermediate + 1.0

        observed = torch.zeros(1, 2, 1)
        control = block_rollout(
            LinearInterpolator(), AddOneForecaster(), observed, 2,
            maximum_horizon=2, alternating=False,
        )
        candidate = block_rollout(
            LinearInterpolator(), AddOneForecaster(), observed, 2,
            maximum_horizon=2, alternating=True,
        )
        torch.testing.assert_close(control, torch.tensor([[[0.5], [1.0]]]))
        torch.testing.assert_close(candidate, torch.tensor([[[0.75], [1.5]]]))

    def test_nonmultiple_horizon_has_exact_output_length(self):
        interpolator = BridgeInterpolator(12, hidden_dim=8)
        forecaster = EndpointForecaster(12, hidden_dim=8)
        observed = torch.randn(2, 10, 12)
        output = block_rollout(
            interpolator, forecaster, observed, 10, maximum_horizon=4
        )
        self.assertEqual(output.shape, (2, 10, 12))
        self.assertTrue(torch.isfinite(output).all())

    def test_invalid_step_is_rejected(self):
        model = BridgeInterpolator(12)
        values = torch.randn(2, 12)
        with self.assertRaises(ValueError):
            model(values, values, 4, 4)


if __name__ == "__main__":
    unittest.main()
