"""Competition-aligned trajectory scenario definitions."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrajectoryScenario:
    """Inclusive observed and target frame ranges for one trajectory task."""

    name: str
    observed_start: int
    observed_end: int
    target_start: int
    target_end: int

    def __post_init__(self):
        if not self.name:
            raise ValueError("scenario name must be nonempty")
        if self.observed_start < 0 or self.observed_end < self.observed_start + 1:
            raise ValueError("a scenario needs at least two observed frames")
        if self.target_start != self.observed_end + 1:
            raise ValueError("target frames must immediately follow observations")
        if self.target_end < self.target_start:
            raise ValueError("a scenario needs at least one target frame")

    @property
    def initializer_indices(self):
        return (self.observed_end - 1, self.observed_end)

    @property
    def target_indices(self):
        return tuple(range(self.target_start, self.target_end + 1))


def competition_scenarios(total_frames=100):
    """Return the fixed project proxies for competition T1, T2 and T3."""
    if total_frames != 100:
        raise ValueError("competition scenario proxies require exactly 100 frames")
    return (
        TrajectoryScenario("T1", 0, 1, 2, 19),
        TrajectoryScenario("T2", 0, 79, 80, 99),
        TrajectoryScenario("T3", 0, 19, 20, 99),
    )


def scenario_time_grid(scenario, scaling):
    """Return local ODE times from the first initializer through the final target."""
    if scaling <= 0 or not np.isfinite(scaling):
        raise ValueError("scaling must be finite and positive")
    first_frame = scenario.initializer_indices[0]
    frame_count = scenario.target_end - first_frame + 1
    return np.arange(frame_count, dtype=np.float32) / float(scaling)
