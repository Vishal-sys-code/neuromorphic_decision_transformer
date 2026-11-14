"""
Unit tests for offline_rl dataset loading and sequence trajectory batching.
"""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from offline_rl.datasets.d4rl_loader import D4RLLoader, Trajectory
from offline_rl.datasets.trajectory_dataset import TrajectorySequenceDataset


def test_d4rl_loader_synthetic_dataset():
    """Verify synthetic trajectory creation produces valid dimensions and RTG."""
    state_dim = 11
    act_dim = 3
    num_trajs = 5
    trajs = D4RLLoader.create_synthetic_dataset(
        num_trajectories=num_trajs,
        traj_len=50,
        state_dim=state_dim,
        act_dim=act_dim,
        seed=123,
    )

    assert len(trajs) == num_trajs
    for t in trajs:
        assert isinstance(t, Trajectory)
        assert t.observations.shape[1] == state_dim
        assert t.actions.shape[1] == act_dim
        assert len(t.rewards) == t.length
        assert len(t.returns_to_go) == t.length
        assert len(t.timesteps) == t.length
        # Check RTG calculation: RTG[0] == sum of all rewards
        np.testing.assert_allclose(t.returns_to_go[0], np.sum(t.rewards), rtol=1e-5)
        # Check RTG[-1] == last reward
        np.testing.assert_allclose(t.returns_to_go[-1], t.rewards[-1], rtol=1e-5)


def test_state_statistics():
    """Verify normalization statistics computation."""
    trajs = D4RLLoader.create_synthetic_dataset(num_trajectories=4, traj_len=30, state_dim=5, act_dim=2)
    mean, std = D4RLLoader.compute_state_statistics(trajs)
    assert mean.shape == (5,)
    assert std.shape == (5,)
    assert np.all(std > 0)


def test_trajectory_sequence_dataset_batching():
    """Verify sequence padding, attention masks, and DataLoader integration."""
    state_dim = 11
    act_dim = 3
    context_len = 20
    trajs = D4RLLoader.create_synthetic_dataset(
        num_trajectories=10,
        traj_len=40,
        state_dim=state_dim,
        act_dim=act_dim,
        seed=42,
    )

    dataset = TrajectorySequenceDataset(
        trajectories=trajs,
        context_len=context_len,
        scale=1000.0,
    )

    sample = dataset[0]
    assert "states" in sample
    assert "actions" in sample
    assert "returns_to_go" in sample
    assert "timesteps" in sample
    assert "attention_mask" in sample

    assert sample["states"].shape == (context_len, state_dim)
    assert sample["actions"].shape == (context_len, act_dim)
    assert sample["returns_to_go"].shape == (context_len, 1)
    assert sample["timesteps"].shape == (context_len,)
    assert sample["attention_mask"].shape == (context_len,)

    # Test DataLoader
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(loader))
    assert batch["states"].shape == (4, context_len, state_dim)
    assert batch["actions"].shape == (4, context_len, act_dim)
    assert batch["returns_to_go"].shape == (4, context_len, 1)
    assert batch["timesteps"].shape == (4, context_len)
    assert batch["attention_mask"].shape == (4, context_len)


if __name__ == "__main__":
    test_d4rl_loader_synthetic_dataset()
    test_state_statistics()
    test_trajectory_sequence_dataset_batching()
    print("All dataset tests passed successfully!")
