import unittest

import numpy as np

from protein_quanta.scenarios import competition_scenarios, scenario_time_grid


class CompetitionScenarioTests(unittest.TestCase):
    def test_scenarios_have_exact_observation_and_target_frames(self):
        scenarios = {scenario.name: scenario for scenario in competition_scenarios()}

        self.assertEqual(scenarios["T1"].initializer_indices, (8, 9))
        self.assertEqual(scenarios["T1"].target_indices, tuple(range(10, 20)))
        self.assertEqual(scenarios["T2"].initializer_indices, (78, 79))
        self.assertEqual(scenarios["T2"].target_indices, tuple(range(80, 100)))
        self.assertEqual(scenarios["T3"].initializer_indices, (18, 19))
        self.assertEqual(scenarios["T3"].target_indices, tuple(range(20, 100)))

    def test_time_grid_starts_at_zero_and_covers_initializer_to_target_end(self):
        scenario = {item.name: item for item in competition_scenarios()}["T2"]

        grid = scenario_time_grid(scenario, scaling=100.0)

        np.testing.assert_allclose(grid, np.arange(22) / 100.0)

    def test_scenarios_reject_insufficient_frames_and_nonpositive_scaling(self):
        with self.assertRaises(ValueError):
            competition_scenarios(total_frames=99)
        scenario = competition_scenarios()[0]
        with self.assertRaises(ValueError):
            scenario_time_grid(scenario, scaling=0)


if __name__ == "__main__":
    unittest.main()
