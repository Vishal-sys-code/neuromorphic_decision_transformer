"""
Dataset modules for offline RL and D4RL trajectory data processing.
"""

from offline_rl.datasets.d4rl_loader import (
    D4RLLoader,
    Trajectory,
    split_d4rl_episodes,
    validate_episodes,
)

__all__ = [
    "D4RLLoader",
    "Trajectory",
    "split_d4rl_episodes",
    "validate_episodes",
    "TrajectoryDataset",
    "TrajectorySequenceDataset",
]


def __getattr__(name: str):
    if name in ("TrajectoryDataset", "TrajectorySequenceDataset"):
        from offline_rl.datasets.trajectory_dataset import TrajectoryDataset, TrajectorySequenceDataset
        if name == "TrajectoryDataset":
            return TrajectoryDataset
        return TrajectorySequenceDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
