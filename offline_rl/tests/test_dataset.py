"""
Unit tests for offline_rl dataset loading and episode splitting.
Gate 2: Dataset integrity verification.
"""

import numpy as np
import pytest
import torch

from offline_rl.datasets.d4rl_loader import (
    D4RLLoader,
    Trajectory,
    split_d4rl_episodes,
    validate_episodes,
)
from offline_rl.datasets.trajectory_dataset import (
    TrajectoryDataset,
    TrajectorySequenceDataset,
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


def test_gate3_trajectory_dataset_batch_shapes():
    """
    Phase 3 / Gate 3: Batch shapes are correct.
    Expected:
        states: (8, 20, 11)
        actions: (8, 20, 3)
        returns_to_go: (8, 20, 1)
        timesteps: (8, 20)
        attention_mask: (8, 20)
    Pass condition:
        All five tensors have correct shapes, no NaNs, and padded timesteps are masked.
    """
    import torch
    from torch.utils.data import DataLoader

    np.random.seed(42)
    torch.manual_seed(42)

    flat_dataset = {
        "observations": np.random.randn(300, 11).astype(np.float32),
        "actions": np.random.uniform(-1, 1, size=(300, 3)).astype(np.float32),
        "rewards": np.random.uniform(0, 5, size=(300,)).astype(np.float32),
        "terminals": np.zeros(300, dtype=bool),
        "timeouts": np.zeros(300, dtype=bool),
    }
    flat_dataset["terminals"][45] = True
    flat_dataset["terminals"][110] = True
    flat_dataset["terminals"][175] = True
    flat_dataset["terminals"][220] = True
    flat_dataset["terminals"][299] = True

    episodes = split_d4rl_episodes(flat_dataset)
    assert len(episodes) >= 4

    dataset = TrajectoryDataset(
        episodes,
        context_length=20,
    )
    loader = DataLoader(
        dataset,
        batch_size=8,
        num_workers=0,
    )
    batch = next(iter(loader))

    expected_shapes = {
        "states": (8, 20, 11),
        "actions": (8, 20, 3),
        "returns_to_go": (8, 20, 1),
        "timesteps": (8, 20),
        "attention_mask": (8, 20),
    }

    # 1. Shape checks
    for key, shape in expected_shapes.items():
        assert key in batch, f"Missing key {key} in batch"
        assert batch[key].shape == shape, f"Shape mismatch for {key}: expected {shape}, got {batch[key].shape}"

    # 2. No NaNs or Infs
    for key in expected_shapes:
        tensor = batch[key]
        assert not torch.isnan(tensor).any(), f"NaN detected in {key}"
        assert not torch.isinf(tensor).any(), f"Inf detected in {key}"

    # 3. Padded timesteps are masked
    mask = batch["attention_mask"]
    assert torch.all((mask == 0.0) | (mask == 1.0))

    for b in range(8):
        for t in range(20):
            if mask[b, t] == 0.0:
                assert torch.all(batch["states"][b, t] == 0.0), f"Unzeroed state at masked timestep ({b}, {t})"
                assert torch.all(batch["actions"][b, t] == 0.0), f"Unzeroed action at masked timestep ({b}, {t})"
                assert torch.all(batch["returns_to_go"][b, t] == 0.0), f"Unzeroed rtg at masked timestep ({b}, {t})"
                assert batch["timesteps"][b, t] == 0, f"Unzeroed timestep at masked timestep ({b}, {t})"


def test_sampling_proportional_to_length():
    """Verify episode sampling probability P(tau_i) = T_i / sum_j T_j."""
    episodes = [
        {
            "observations": np.zeros((10, 11), dtype=np.float32),
            "actions": np.zeros((10, 3), dtype=np.float32),
            "rewards": np.zeros(10, dtype=np.float32),
            "returns_to_go": np.zeros(10, dtype=np.float32),
            "timesteps": np.arange(10, dtype=np.int64),
        },
        {
            "observations": np.zeros((30, 11), dtype=np.float32),
            "actions": np.zeros((30, 3), dtype=np.float32),
            "rewards": np.zeros(30, dtype=np.float32),
            "returns_to_go": np.zeros(30, dtype=np.float32),
            "timesteps": np.arange(30, dtype=np.int64),
        },
        {
            "observations": np.zeros((60, 11), dtype=np.float32),
            "actions": np.zeros((60, 3), dtype=np.float32),
            "rewards": np.zeros(60, dtype=np.float32),
            "returns_to_go": np.zeros(60, dtype=np.float32),
            "timesteps": np.arange(60, dtype=np.int64),
        },
    ]
    dataset = TrajectoryDataset(episodes, context_length=20)
    expected_probs = np.array([10 / 100.0, 30 / 100.0, 60 / 100.0])
    np.testing.assert_allclose(dataset.sampling_probs, expected_probs, rtol=1e-5)


def test_left_padding_for_short_episode():
    """Verify an episode shorter than context_length is correctly left-padded and masked."""
    short_ep = [
        {
            "observations": np.ones((5, 11), dtype=np.float32) * 3.0,
            "actions": np.ones((5, 3), dtype=np.float32) * 0.5,
            "rewards": np.ones(5, dtype=np.float32),
            "returns_to_go": np.array([5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float32),
            "timesteps": np.arange(5, dtype=np.int64),
        }
    ]
    dataset = TrajectoryDataset(short_ep, context_length=20)
    sample = dataset[0]

    assert sample["states"].shape == (20, 11)
    assert sample["actions"].shape == (20, 3)
    assert sample["returns_to_go"].shape == (20, 1)
    assert sample["timesteps"].shape == (20,)
    assert sample["attention_mask"].shape == (20,)

    mask = sample["attention_mask"]
    num_valid = int(mask.sum().item())
    assert 1 <= num_valid <= 5
    pad_len = 20 - num_valid

    # All padded steps must have mask=0 and zeroed state, action, rtg, timestep
    if pad_len > 0:
        assert torch.all(mask[:pad_len] == 0.0)
        assert torch.all(sample["states"][:pad_len] == 0.0)
        assert torch.all(sample["actions"][:pad_len] == 0.0)
        assert torch.all(sample["returns_to_go"][:pad_len] == 0.0)
        assert torch.all(sample["timesteps"][:pad_len] == 0)

    # All valid steps must have mask=1
    assert torch.all(mask[pad_len:] == 1.0)


if __name__ == "__main__":
    test_episode_split()
    test_gate3_trajectory_dataset_batch_shapes()
    test_sampling_proportional_to_length()
    test_left_padding_for_short_episode()
    print("Gate 3 tests passed.")
