"""
Unit tests for offline_rl dataset loading and episode splitting.
Gate 2: Dataset integrity verification.
"""

import numpy as np
import pytest

from offline_rl.datasets.d4rl_loader import (
    D4RLLoader,
    Trajectory,
    split_d4rl_episodes,
    validate_episodes,
)


def test_episode_split():
    dataset = {
        "observations": np.random.randn(10, 11).astype(np.float32),
        "actions": np.random.randn(10, 3).astype(np.float32),
        "rewards": np.ones(10, dtype=np.float32),
        "terminals": np.array(
            [0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
            dtype=bool,
        ),
        "timeouts": np.zeros(10, dtype=bool),
    }
    episodes = split_d4rl_episodes(dataset)
    assert len(episodes) == 3
    assert [len(ep["observations"]) for ep in episodes] == [3, 4, 3]
    assert validate_episodes(episodes)
    assert np.allclose(
        episodes[0]["returns_to_go"],
        [3.0, 2.0, 1.0],
    )


def test_episode_split_timeouts_and_max_len():
    """Verify episode boundary triggered by timeouts and max_episode_length."""
    dataset = {
        "observations": np.zeros((15, 4), dtype=np.float32),
        "actions": np.zeros((15, 2), dtype=np.float32),
        "rewards": np.full(15, 2.0, dtype=np.float32),
        "terminals": np.zeros(15, dtype=bool),
        "timeouts": np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=bool),
    }
    episodes = split_d4rl_episodes(dataset, max_episode_length=5)
    # Step 0..4 (len 5) -> timeout at index 4
    # Step 5..9 (len 5) -> max_episode_length 5 at index 9
    # Step 10..14 (len 5) -> dataset end (t == n - 1)
    assert len(episodes) == 3
    assert [len(ep["observations"]) for ep in episodes] == [5, 5, 5]
    assert validate_episodes(episodes)
    assert np.allclose(episodes[0]["returns_to_go"], [10.0, 8.0, 6.0, 4.0, 2.0])


def test_validate_episodes_catches_invalid_data():
    """Verify validate_episodes rejects corrupted inputs (Gate 2 integrity checks)."""
    # 1. Missing keys
    corrupted_missing_keys = [{"observations": np.zeros((3, 2))}]
    assert not validate_episodes(corrupted_missing_keys)

    # 2. Length mismatch
    corrupted_mismatch = [{
        "observations": np.zeros((3, 2), dtype=np.float32),
        "actions": np.zeros((2, 1), dtype=np.float32),
        "rewards": np.ones(3, dtype=np.float32),
        "terminals": np.zeros(3, dtype=bool),
        "timeouts": np.zeros(3, dtype=bool),
        "returns_to_go": np.ones(3, dtype=np.float32),
        "timesteps": np.arange(3, dtype=np.int64),
    }]
    assert not validate_episodes(corrupted_mismatch)

    # 3. NaNs in observations
    corrupted_nan = [{
        "observations": np.array([[np.nan, 1.0], [0.0, 0.0]], dtype=np.float32),
        "actions": np.zeros((2, 1), dtype=np.float32),
        "rewards": np.ones(2, dtype=np.float32),
        "terminals": np.zeros(2, dtype=bool),
        "timeouts": np.zeros(2, dtype=bool),
        "returns_to_go": np.array([2.0, 1.0], dtype=np.float32),
        "timesteps": np.arange(2, dtype=np.int64),
    }]
    assert not validate_episodes(corrupted_nan)

    # 4. Incorrect returns-to-go
    corrupted_rtg = [{
        "observations": np.zeros((2, 2), dtype=np.float32),
        "actions": np.zeros((2, 1), dtype=np.float32),
        "rewards": np.array([1.0, 1.0], dtype=np.float32),
        "terminals": np.zeros(2, dtype=bool),
        "timeouts": np.zeros(2, dtype=bool),
        "returns_to_go": np.array([5.0, 1.0], dtype=np.float32),
        "timesteps": np.arange(2, dtype=np.int64),
    }]
    assert not validate_episodes(corrupted_rtg)


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
        np.testing.assert_allclose(t.returns_to_go[0], np.sum(t.rewards), rtol=1e-5)
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
    import torch
    from torch.utils.data import DataLoader
    from offline_rl.datasets.trajectory_dataset import TrajectorySequenceDataset

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

    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(loader))
    assert batch["states"].shape == (4, context_len, state_dim)
    assert batch["actions"].shape == (4, context_len, act_dim)
    assert batch["returns_to_go"].shape == (4, context_len, 1)
    assert batch["timesteps"].shape == (4, context_len)
    assert batch["attention_mask"].shape == (4, context_len)


if __name__ == "__main__":
    test_episode_split()
    print("Dataset test passed.")
