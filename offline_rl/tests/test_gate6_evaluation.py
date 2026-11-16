"""
Unit and integration tests for Phase 6: MuJoCo Evaluation & Gate 6: Policy produces valid trajectories.

Tests Gate 6 criteria:
1. D4RL-v2 reference scores match exact published benchmarks without substitution.
2. Normalized score formula:
   Normalized Score = 100 * (R_policy - R_random) / (R_expert - R_random)
3. Action clipping ensures all actions lie within environment action bounds.
4. Deterministic evaluation generates identical trajectories for the same seed.
5. Autoregressive RTG updates: R_{t+1} = (R_t * scale - r_t) / scale.
6. Context length truncation: Slices never exceed context_len K.
7. All 6 required metrics are tracked:
   - mean_return
   - std_return
   - normalized_score
   - episode_length
   - spike_count
   - inference_latency
8. Gate 6 trajectory validation detects NaNs, Infs, out-of-bound actions, or collapses.
9. Target return tuning routine correctly identifies sensitivity and optimal conditioning.
"""

import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import pytest
import torch

from offline_rl.evaluation.evaluate_mujoco import (
    D4RL_REF_SCORES,
    EpisodeResult,
    TrajectoryRollout,
    compute_d4rl_normalized_score,
    evaluate_episode,
    evaluate_mujoco,
    evaluate_seed,
    tune_target_returns,
    validate_gate6_policy,
)
from offline_rl.models.snn_dt_mujoco import ContinuousSnnDecisionTransformer


@pytest.fixture
def mock_hopper_model():
    """Deterministic lightweight SNN-DT model for fast testing."""
    torch.manual_seed(42)
    model = ContinuousSnnDecisionTransformer(
        state_dim=11,
        act_dim=3,
        embed_dim=32,
        n_layer=2,
        n_head=2,
        snn_timesteps=2,
        dropout=0.0,
        action_tanh=True,
    )
    model.eval()
    return model


@pytest.fixture
def dummy_stats():
    """Zero-mean unit-variance normalization statistics."""
    state_mean = np.zeros(11, dtype=np.float32)
    state_std = np.ones(11, dtype=np.float32)
    return state_mean, state_std


def test_d4rl_reference_scores_exactness():
    """
    Verify reference scores associated with exact D4RL-v2 environments.
    Strictly forbids substitution with newer MuJoCo versions.
    """
    assert "hopper-medium-v2" in D4RL_REF_SCORES
    ref_hopper = D4RL_REF_SCORES["hopper-medium-v2"]
    assert np.isclose(ref_hopper["random"], -20.272305, atol=1e-5)
    assert np.isclose(ref_hopper["expert"], 3234.3, atol=1e-2)

    assert "halfcheetah-medium-v2" in D4RL_REF_SCORES
    ref_cheetah = D4RL_REF_SCORES["halfcheetah-medium-v2"]
    assert np.isclose(ref_cheetah["random"], -280.178953, atol=1e-5)
    assert np.isclose(ref_cheetah["expert"], 12135.0, atol=1e-2)

    assert "walker2d-medium-v2" in D4RL_REF_SCORES
    ref_walker = D4RL_REF_SCORES["walker2d-medium-v2"]
    assert np.isclose(ref_walker["random"], 1.629008, atol=1e-5)
    assert np.isclose(ref_walker["expert"], 4592.3, atol=1e-2)


def test_normalized_score_calculation():
    """
    Verify normalized score formula:
        Normalized Score = 100 * (R_policy - R_random) / (R_expert - R_random)
    """
    # Exact expert return -> 100.0
    expert_norm = compute_d4rl_normalized_score(3234.3, "hopper-medium-v2")
    assert np.isclose(expert_norm, 100.0, atol=1e-4)

    # Exact random return -> 0.0
    random_norm = compute_d4rl_normalized_score(-20.272305, "hopper-medium-v2")
    assert np.isclose(random_norm, 0.0, atol=1e-4)

    # Intermediate return (1500.0)
    # 100 * (1500 - (-20.272305)) / (3234.3 - (-20.272305))
    expected = 100.0 * (1500.0 - (-20.272305)) / (3234.3 - (-20.272305))
    inter_norm = compute_d4rl_normalized_score(1500.0, "hopper-medium-v2")
    assert np.isclose(inter_norm, expected, atol=1e-4)

    # Unknown environment raises KeyError
    with pytest.raises(KeyError):
        compute_d4rl_normalized_score(100.0, "nonexistent-env-v0")


def test_action_clipping_bounds(mock_hopper_model, dummy_stats):
    """
    Verify actions are strictly clipped to environment bounds.
    """
    state_mean, state_std = dummy_stats
    env = gym.make("Hopper-v4")

    # Run episode with action_clip=True
    res = evaluate_episode(
        env=env,
        model=mock_hopper_model,
        state_mean=state_mean,
        state_std=state_std,
        target_return=3000.0,
        max_ep_len=20,
        action_clip=True,
        seed=100,
    )
    env.close()

    traj = res.trajectory
    assert traj is not None
    assert np.all(traj.actions >= env.action_space.low)
    assert np.all(traj.actions <= env.action_space.high)


def test_deterministic_evaluation(mock_hopper_model, dummy_stats):
    """
    Verify deterministic evaluation produces identical action and state sequences for the same seed.
    """
    state_mean, state_std = dummy_stats
    env = gym.make("Hopper-v4")

    res1 = evaluate_episode(
        env=env,
        model=mock_hopper_model,
        state_mean=state_mean,
        state_std=state_std,
        target_return=3000.0,
        max_ep_len=25,
        deterministic=True,
        seed=42,
    )

    res2 = evaluate_episode(
        env=env,
        model=mock_hopper_model,
        state_mean=state_mean,
        state_std=state_std,
        target_return=3000.0,
        max_ep_len=25,
        deterministic=True,
        seed=42,
    )
    env.close()

    assert np.isclose(res1.total_reward, res2.total_reward, atol=1e-5)
    assert res1.episode_length == res2.episode_length
    assert np.allclose(res1.trajectory.actions, res2.trajectory.actions, atol=1e-5)
    assert np.allclose(res1.trajectory.states, res2.trajectory.states, atol=1e-5)


def test_autoregressive_rtg_update(mock_hopper_model, dummy_stats):
    """
    Verify return-to-go updates conform to DT inference:
        R_{t+1} = (R_t * scale - r_t) / scale
    """
    state_mean, state_std = dummy_stats
    env = gym.make("Hopper-v4")
    target_return = 3000.0
    scale = 1000.0

    res = evaluate_episode(
        env=env,
        model=mock_hopper_model,
        state_mean=state_mean,
        state_std=state_std,
        target_return=target_return,
        scale=scale,
        max_ep_len=15,
        seed=100,
    )
    env.close()

    rtgs = res.trajectory.returns_to_go
    rewards = res.trajectory.rewards

    # First RTG is initial target return
    assert np.isclose(rtgs[0], target_return, atol=1e-3)

    # Consecutive RTG values step down by reward
    for t in range(len(rewards) - 1):
        expected_next = rtgs[t] - rewards[t]
        assert np.isclose(rtgs[t + 1], expected_next, atol=1e-3)


def test_metrics_collection_completeness(mock_hopper_model, dummy_stats):
    """
    Verify all required metrics are recorded:
    - mean_return
    - std_return
    - normalized_score
    - episode_length
    - spike_count
    - inference_latency
    """
    state_mean, state_std = dummy_stats
    summary = evaluate_mujoco(
        env_name="Hopper-v4",
        model=mock_hopper_model,
        state_mean=state_mean,
        state_std=state_std,
        seeds=[100, 200],
        num_episodes=2,
        target_return=3000.0,
        max_ep_len=15,
        d4rl_env_name="hopper-medium-v2",
    )

    required_metrics = [
        "mean_return",
        "std_return",
        "normalized_score",
        "episode_length",
        "spike_count",
        "inference_latency",
    ]

    for m in required_metrics:
        assert m in summary, f"Required metric '{m}' missing from evaluation summary"
        assert np.isfinite(summary[m]), f"Metric '{m}' is non-finite: {summary[m]}"

    # Verify per-step latency is positive and in ms
    assert summary["inference_latency"] > 0.0
    # Verify spike count is tracked
    assert summary["spike_count"] >= 0.0
    # Verify episode length > 0
    assert summary["episode_length"] > 0


def test_trajectory_rollout_validation():
    """
    Verify TrajectoryRollout.validate correctly identifies valid trajectories
    and flags NaNs, Infs, and bound violations.
    """
    act_low = np.array([-1.0, -1.0, -1.0])
    act_high = np.array([1.0, 1.0, 1.0])

    # 1. Valid trajectory
    valid_traj = TrajectoryRollout(
        states=np.random.randn(10, 11).astype(np.float32),
        actions=np.clip(np.random.randn(10, 3).astype(np.float32), -1.0, 1.0),
        raw_actions=np.random.randn(10, 3).astype(np.float32),
        rewards=np.ones(10, dtype=np.float32),
        returns_to_go=np.linspace(3000, 2991, 10, dtype=np.float32),
        timesteps=np.arange(10, dtype=np.int64),
        step_latencies_ms=[5.0] * 10,
        step_spike_counts=[10.0] * 10,
        step_spike_rates=[0.2] * 10,
    )
    val = valid_traj.validate(act_low, act_high)
    assert val["is_valid"] is True
    assert len(val["issues"]) == 0

    # 2. Corrupt trajectory with NaNs
    nan_traj = TrajectoryRollout(
        states=np.full((5, 11), np.nan, dtype=np.float32),
        actions=np.zeros((5, 3), dtype=np.float32),
        raw_actions=np.zeros((5, 3), dtype=np.float32),
        rewards=np.ones(5, dtype=np.float32),
        returns_to_go=np.ones(5, dtype=np.float32),
        timesteps=np.arange(5, dtype=np.int64),
        step_latencies_ms=[5.0] * 5,
        step_spike_counts=[10.0] * 5,
        step_spike_rates=[0.2] * 5,
    )
    val_nan = nan_traj.validate(act_low, act_high)
    assert val_nan["is_valid"] is False
    assert any("NaN" in issue for issue in val_nan["issues"])

    # 3. Action out of bounds
    oob_traj = TrajectoryRollout(
        states=np.zeros((5, 11), dtype=np.float32),
        actions=np.full((5, 3), 50.0, dtype=np.float32),  # Exceeds [-1, 1]
        raw_actions=np.full((5, 3), 50.0, dtype=np.float32),
        rewards=np.ones(5, dtype=np.float32),
        returns_to_go=np.ones(5, dtype=np.float32),
        timesteps=np.arange(5, dtype=np.int64),
        step_latencies_ms=[5.0] * 5,
        step_spike_counts=[10.0] * 5,
        step_spike_rates=[0.2] * 5,
    )
    val_oob = oob_traj.validate(act_low, act_high)
    assert val_oob["is_valid"] is False
    assert any("bounds" in issue for issue in val_oob["issues"])


def test_gate6_policy_validation_function():
    """Verify validate_gate6_policy certifies valid trajectories and catches failures."""
    valid_summary = {
        "gate_6_status": "PASSED",
        "gate_6_issues": [],
        "episode_length": 25.0,
        "mean_return": 35.0,
        "normalized_score": 1.7,
        "spike_count": 8.5,
        "inference_latency": 12.0,
    }
    passed, issues = validate_gate6_policy(valid_summary, min_length_threshold=5.0)
    assert passed is True
    assert len(issues) == 0

    failing_summary = {
        "gate_6_status": "FAILED",
        "gate_6_issues": ["Action out of bounds"],
        "episode_length": 2.0,  # Below threshold
        "mean_return": np.nan,  # Non-finite
        "normalized_score": 0.0,
        "spike_count": 0.0,
        "inference_latency": 0.0,
    }
    passed_fail, issues_fail = validate_gate6_policy(failing_summary, min_length_threshold=5.0)
    assert passed_fail is False
    assert len(issues_fail) >= 2


def test_target_return_tuning_routine(mock_hopper_model, dummy_stats):
    """Verify tune_target_returns sweeps across candidate targets."""
    state_mean, state_std = dummy_stats
    tuning = tune_target_returns(
        env_name="Hopper-v4",
        model=mock_hopper_model,
        state_mean=state_mean,
        state_std=state_std,
        target_returns=[1000.0, 3000.0],
        seeds=[100],
        num_episodes_per_seed=1,
        context_len=20,
        d4rl_env_name="hopper-medium-v2",
    )
    assert "best_target_return" in tuning
    assert "best_mean_return" in tuning
    assert len(tuning["sweep_results"]) == 2
    assert tuning["best_target_return"] in [1000.0, 3000.0]


def test_episode_result_backward_compatibility(mock_hopper_model, dummy_stats):
    """
    Verify EpisodeResult can be unpacked as 3-tuple (total_reward, episode_length, avg_spike_rate)
    for seamless integration with legacy code.
    """
    state_mean, state_std = dummy_stats
    env = gym.make("Hopper-v4")
    res = evaluate_episode(
        env=env,
        model=mock_hopper_model,
        state_mean=state_mean,
        state_std=state_std,
        target_return=3000.0,
        max_ep_len=10,
        seed=100,
    )
    env.close()

    # Unpack as tuple
    reward, ep_len, sr = res
    assert isinstance(reward, float)
    assert isinstance(ep_len, int)
    assert isinstance(sr, float)
    assert ep_len > 0
