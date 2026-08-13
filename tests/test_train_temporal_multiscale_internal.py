import unittest

import numpy as np
import torch

from scripts.train_temporal_multiscale_internal import train_multiscale_seed


class MultiscaleInternalTrainingTests(unittest.TestCase):
    def test_short_cpu_run_returns_final_epoch(self):
        rng = np.random.default_rng(7)
        train = rng.normal(size=(4, 100, 12)).astype(np.float32)
        holdout = rng.normal(size=(2, 100, 12)).astype(np.float32)
        result = train_multiscale_seed(
            train,
            holdout,
            {"hidden_dim": 8, "layers": 1, "heads": 2},
            {
                "learning_rate": 0.001,
                "epochs": 1,
                "batch_size": 2,
                "gradient_clip_norm": 1.0,
                "one_step_loss_weight": 1.0,
                "closed_loop_loss_weight": 0.25,
                "closed_loop_horizons": [5, 10],
            },
            seed=0,
            device=torch.device("cpu"),
            static_step_values=np.zeros(4),
        )
        self.assertEqual(result["final"]["epoch"], 1)
        self.assertTrue(np.isfinite(result["final"]["macro_scenario_rmse"]))


if __name__ == "__main__":
    unittest.main()
