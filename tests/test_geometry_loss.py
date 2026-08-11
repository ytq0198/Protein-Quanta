import unittest

import torch

from protein_quanta.geometry_loss import (
    calibrated_auxiliary_weight,
    pair_distance_smooth_l1,
)


class PairDistanceLossTests(unittest.TestCase):
    def test_zero_on_truth_and_finite_backward(self):
        truth = torch.randn(3, 4, 3)
        prediction = truth.clone().requires_grad_(True)
        batch = torch.tensor([0, 0, 1, 1])

        loss = pair_distance_smooth_l1(prediction, truth, batch)
        loss.backward()

        self.assertEqual(loss.item(), 0.0)
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_is_invariant_to_rigid_translation_and_rotation(self):
        truth = torch.tensor(
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]]
        )
        rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        prediction = truth @ rotation.T + torch.tensor([3.0, -4.0, 2.0])

        loss = pair_distance_smooth_l1(
            prediction, truth, torch.zeros(3, dtype=torch.long)
        )

        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_uses_one_unordered_nondiagonal_pair(self):
        truth = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
        prediction = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])

        loss = pair_distance_smooth_l1(
            prediction, truth, torch.tensor([0, 0]), beta=0.5
        )

        self.assertAlmostEqual(loss.item(), 0.75)

    def test_balances_complexes_instead_of_pairs(self):
        truth = torch.tensor(
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]
        )
        prediction = truth.clone()
        prediction[:, 1, 0] = 2.0

        loss = pair_distance_smooth_l1(
            prediction, truth, torch.tensor([0, 0, 1, 1, 1]), beta=0.5
        )

        self.assertAlmostEqual(loss.item(), 0.375)

    def test_rejects_invalid_inputs_and_single_atom_complexes(self):
        trajectory = torch.zeros(2, 2, 3)
        cases = (
            (trajectory[:, :, :2], trajectory[:, :, :2], torch.tensor([0, 0]), 0.5),
            (trajectory, trajectory[:1], torch.tensor([0, 0]), 0.5),
            (trajectory, trajectory, torch.tensor([0]), 0.5),
            (trajectory, trajectory, torch.tensor([0, 1]), 0.5),
            (trajectory, trajectory, torch.tensor([0, 0]), 0.0),
        )
        for prediction, truth, batch, beta in cases:
            with self.subTest(shape=tuple(prediction.shape), beta=beta):
                with self.assertRaises(ValueError):
                    pair_distance_smooth_l1(prediction, truth, batch, beta=beta)


class AuxiliaryWeightCalibrationTests(unittest.TestCase):
    def test_uses_target_fraction_times_median_gradient_ratio(self):
        weight = calibrated_auxiliary_weight([2.0, 8.0, 6.0], [1.0, 2.0, 3.0])
        self.assertAlmostEqual(weight, 0.2)

    def test_zero_pair_gradient_uses_epsilon_floor(self):
        weight = calibrated_auxiliary_weight(
            [2.0], [0.0], target_fraction=0.1, epsilon=0.5
        )
        self.assertAlmostEqual(weight, 0.4)

    def test_rejects_invalid_calibration_inputs(self):
        cases = (
            ([], [], 0.1, 1e-12),
            ([1.0], [1.0, 2.0], 0.1, 1e-12),
            ([float("nan")], [1.0], 0.1, 1e-12),
            ([1.0], [-1.0], 0.1, 1e-12),
            ([1.0], [1.0], 0.0, 1e-12),
            ([1.0], [1.0], 1.1, 1e-12),
            ([1.0], [1.0], 0.1, 0.0),
        )
        for position, pair, fraction, epsilon in cases:
            with self.subTest(position=position, pair=pair):
                with self.assertRaises(ValueError):
                    calibrated_auxiliary_weight(
                        position,
                        pair,
                        target_fraction=fraction,
                        epsilon=epsilon,
                    )


if __name__ == "__main__":
    unittest.main()
