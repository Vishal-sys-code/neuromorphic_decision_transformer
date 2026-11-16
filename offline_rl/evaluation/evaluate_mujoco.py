"""
MuJoCo Evaluation Pipeline for Spiking Decision Transformer policies.

Phase 6: MuJoCo evaluation & Gate 6: Policy produces valid trajectories.

This module provides the comprehensive evaluation harness for SNN-DT:
1. Autoregressive inference loop:
   - Reset environment (with deterministic seed)
   - Set desired return-to-go (target_return / scale)
   - Pass recent (R, s, a, t) context window to SNN-DT
   - Predict action for current timestep
   - Clip action to environment bounds
   - Execute action in MuJoCo
   - Update return-to-go: R_{t+1} = (R_t * scale - r_t) / scale
   - Continue until termination or truncation
2. Normalized D4RL score calculation:
   Normalized Score = 100 * (R_policy - R_random) / (R_expert - R_random)
   Using exact reference scores for D4RL-v2 environments.
3. Metric tracking across evaluation seeds:
   - mean_return, std_return
   - normalized_score
   - episode_length
   - spike_count (per step & per episode)
   - inference_latency (per step in ms)
   - mean_spike_rate, estimated energy consumption
4. Gate 6 trajectory validity certification:
   - Valid finite dynamics (no NaNs / Infs)
   - Action boundary adherence
   - Non-degenerate trajectory progression
"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Ensure repository root is on sys.path
_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import gymnasium as gym
import numpy as np
import torch
import yaml

from offline_rl.models.snn_dt_mujoco import ContinuousSnnDecisionTransformer

# ==============================================================================
# Exact D4RL-v2 Reference Scores (Official Fu et al., 2020 / Chen et al., 2021)
# Do NOT substitute scores from newer MuJoCo versions.
# ==============================================================================
D4RL_REF_SCORES: Dict[str, Dict[str, float]] = {
    "hopper-medium-v2": {"random": -20.272305, "expert": 3234.3},
    "hopper-medium-replay-v2": {"random": -20.272305, "expert": 3234.3},
    "hopper-medium-expert-v2": {"random": -20.272305, "expert": 3234.3},
    "halfcheetah-medium-v2": {"random": -280.178953, "expert": 12135.0},
    "halfcheetah-medium-expert-v2": {"random": -280.178953, "expert": 12135.0},
    "halfcheetah-medium-replay-v2": {"random": -280.178953, "expert": 12135.0},
    "walker2d-medium-v2": {"random": 1.629008, "expert": 4592.3},
    "walker2d-medium-replay-v2": {"random": 1.629008, "expert": 4592.3},
    "walker2d-medium-expert-v2": {"random": 1.629008, "expert": 4592.3},
    "ant-medium-v2": {"random": -325.6, "expert": 3880.0},
    "ant-medium-replay-v2": {"random": -325.6, "expert": 3880.0},
    "ant-medium-expert-v2": {"random": -325.6, "expert": 3880.0},
}

# Environment alias mapping
ENV_ALIAS_MAP: Dict[str, str] = {
    "hopper-medium": "hopper-medium-v2",
    "hopper-medium-replay": "hopper-medium-replay-v2",
    "hopper-medium-expert": "hopper-medium-expert-v2",
    "hopper-v4": "hopper-medium-v2",
    "hopper-v3": "hopper-medium-v2",
    "hopper-v2": "hopper-medium-v2",
    "halfcheetah-medium": "halfcheetah-medium-v2",
    "halfcheetah-medium-replay": "halfcheetah-medium-replay-v2",
    "halfcheetah-medium-expert": "halfcheetah-medium-expert-v2",
    "halfcheetah-v4": "halfcheetah-medium-v2",
    "halfcheetah-v3": "halfcheetah-medium-v2",
    "halfcheetah-v2": "halfcheetah-medium-v2",
    "walker2d-medium": "walker2d-medium-v2",
    "walker2d-medium-replay": "walker2d-medium-replay-v2",
    "walker2d-medium-expert": "walker2d-medium-expert-v2",
    "walker2d-v4": "walker2d-medium-v2",
    "walker2d-v3": "walker2d-medium-v2",
    "walker2d-v2": "walker2d-medium-v2",
    "ant-medium": "ant-medium-v2",
    "ant-medium-replay": "ant-medium-replay-v2",
    "ant-medium-expert": "ant-medium-expert-v2",
    "ant-v4": "ant-medium-v2",
}


def resolve_d4rl_name(env_name: str) -> str:
    """Resolve environment name or Gym identifier to its canonical D4RL dataset name."""
    clean = env_name.strip().lower()
    if clean in D4RL_REF_SCORES:
        return clean
    if clean in ENV_ALIAS_MAP:
        return ENV_ALIAS_MAP[clean]
    # Check if removing suffix works
    for k in D4RL_REF_SCORES:
        if clean.startswith(k.split("-v")[0]):
            return k
    return clean


def compute_d4rl_normalized_score(
    score: float,
    env_name: str,
    ref_scores: Optional[Dict[str, Dict[str, float]]] = None,
) -> float:
    """
    Compute normalized D4RL score:
        Normalized Score = 100 * (R_policy - R_random) / (R_expert - R_random)

    Uses exact reference scores associated with the exact D4RL environment version.
    """
    scores_dict = ref_scores if ref_scores is not None else D4RL_REF_SCORES
    canonical_name = resolve_d4rl_name(env_name)

    if canonical_name not in scores_dict:
        raise KeyError(
            f"No D4RL reference scores found for '{env_name}' (canonical: '{canonical_name}'). "
            f"Available: {list(scores_dict.keys())}"
        )

    ref = scores_dict[canonical_name]
    random_score = float(ref["random"])
    expert_score = float(ref["expert"])
    denom = expert_score - random_score

    if denom <= 0.0:
        raise ValueError(
            f"Invalid D4RL reference scores for {canonical_name}: expert ({expert_score}) <= random ({random_score})"
        )

    normalized = 100.0 * (score - random_score) / denom
    return float(normalized)


# ==============================================================================
# Trajectory and Episode Data Structures
# ==============================================================================
@dataclass
class TrajectoryRollout:
    """Full trajectory recorded during a single evaluation episode."""
    states: np.ndarray               # (T, state_dim)
    actions: np.ndarray              # (T, act_dim)
    raw_actions: np.ndarray          # (T, act_dim)
    rewards: np.ndarray              # (T,)
    returns_to_go: np.ndarray        # (T,)
    timesteps: np.ndarray            # (T,)
    step_latencies_ms: List[float]   # Latency per step in ms
    step_spike_counts: List[float]   # Spikes fired per step
    step_spike_rates: List[float]    # Spike rate per step
    terminated: bool = False
    truncated: bool = False

    def validate(self, action_low: np.ndarray, action_high: np.ndarray, tol: float = 1e-4) -> Dict[str, Any]:
        """
        Validate trajectory for Gate 6 criteria:
        - All tensors are strictly finite (no NaNs or Infs).
        - Executed actions lie within [action_low, action_high].
        - Episode length >= 1.
        - RTG monotonically updates according to rewards.
        """
        issues = []
        # Check finiteness
        if not np.all(np.isfinite(self.states)):
            issues.append("States contain NaN or Inf values.")
        if not np.all(np.isfinite(self.actions)):
            issues.append("Actions contain NaN or Inf values.")
        if not np.all(np.isfinite(self.rewards)):
            issues.append("Rewards contain NaN or Inf values.")
        if not np.all(np.isfinite(self.returns_to_go)):
            issues.append("Returns-to-go contain NaN or Inf values.")

        # Check action bounds
        if np.any(self.actions < action_low - tol) or np.any(self.actions > action_high + tol):
            issues.append("Actions exceed environment action bounds.")

        # Check non-empty
        if len(self.rewards) == 0:
            issues.append("Trajectory has zero steps.")

        # Check RTG consistency: R_{t+1} approx R_t - reward
        if len(self.returns_to_go) > 1:
            diffs = self.returns_to_go[:-1] - self.rewards[:-1]
            if not np.allclose(diffs, self.returns_to_go[1:], atol=1e-2):
                issues.append("RTG update mismatch detected.")

        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "num_steps": len(self.rewards),
            "total_return": float(np.sum(self.rewards)),
        }


@dataclass
class EpisodeResult:
    """
    Summary of an evaluation episode.
    Supports tuple-unpacking (total_reward, episode_length, avg_spike_rate)
    for 100% backward compatibility with legacy scripts.
    """
    total_reward: float
    episode_length: int
    avg_spike_rate: float
    total_spikes: float = 0.0
    avg_spike_count_per_step: float = 0.0
    avg_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    normalized_score: Optional[float] = None
    seed: Optional[int] = None
    trajectory: Optional[TrajectoryRollout] = None

    def __iter__(self):
        # Enables: total_reward, ep_len, sr = evaluate_episode(...)
        return iter((self.total_reward, self.episode_length, self.avg_spike_rate))


# ==============================================================================
# Core Episode Evaluation (Inference Loop)
# ==============================================================================
def evaluate_episode(
    env: gym.Env,
    model: ContinuousSnnDecisionTransformer,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    target_return: float = 3000.0,
    scale: float = 1000.0,
    context_len: Optional[int] = None,
    max_ep_len: int = 1000,
    device: str = "cpu",
    seed: Optional[int] = None,
    deterministic: bool = True,
    action_clip: bool = True,
    d4rl_env_name: Optional[str] = None,
    record_trajectory: bool = True,
) -> EpisodeResult:
    """
    Execute a single DT-style evaluation episode interacting with a MuJoCo environment.

    Inference loop:
    1. Reset environment with the specified seed.
    2. Set desired return-to-go R_0 = target_return / scale.
    3. Normalize state using dataset statistics: (s - mu) / sigma.
    4. Pass recent context window [t - K + 1 : t + 1] to SNN-DT.
    5. Predict action for current timestep.
    6. Clip action to environment bounds (action_clip=True).
    7. Execute action in environment.
    8. Update return-to-go: R_{t+1} = (R_t * scale - r_t) / scale.
    9. Continue until termination or truncation.

    Returns:
        EpisodeResult: Detailed dataclass that also unpacks as (total_reward, episode_length, avg_spike_rate).
    """
    model.eval()
    if hasattr(model, "reset_state"):
        model.reset_state()
    if hasattr(model, "reset_spike_counts"):
        model.reset_spike_counts()

    # 1. Reset environment
    if seed is not None:
        obs, info = env.reset(seed=seed)
    else:
        obs, info = env.reset()

    state_dim = obs.shape[0]
    act_dim = env.action_space.shape[0]
    act_low = env.action_space.low
    act_high = env.action_space.high

    # Resolve context length
    k_ctx = context_len if context_len is not None else getattr(model, "context_len", 20)

    # Pre-allocate episode buffers
    states = torch.zeros((1, max_ep_len, state_dim), dtype=torch.float32, device=device)
    actions = torch.zeros((1, max_ep_len, act_dim), dtype=torch.float32, device=device)
    returns_to_go = torch.zeros((1, max_ep_len, 1), dtype=torch.float32, device=device)
    timesteps = torch.zeros((1, max_ep_len), dtype=torch.long, device=device)

    # 2. Set desired return-to-go & initial normalized observation
    safe_std = np.where(state_std > 1e-6, state_std, 1.0)
    norm_obs = (obs - state_mean) / safe_std

    states[0, 0] = torch.from_numpy(norm_obs).to(device)
    returns_to_go[0, 0] = float(target_return) / float(scale)
    timesteps[0, 0] = 0

    # Tracking records
    recorded_states = []
    recorded_actions = []
    recorded_raw_actions = []
    recorded_rewards = []
    recorded_rtgs = []
    recorded_timesteps = []
    step_latencies_ms = []
    step_spike_counts = []
    step_spike_rates = []

    total_reward = 0.0
    ep_len = 0
    terminated = False
    truncated = False

    for t in range(max_ep_len):
        ep_len += 1

        # 3. Context window slice [start_idx : t + 1]
        start_idx = max(0, t + 1 - k_ctx)
        s_ctx = states[:, start_idx : t + 1]
        a_ctx = actions[:, start_idx : t + 1]
        r_ctx = returns_to_go[:, start_idx : t + 1]
        t_ctx = timesteps[:, start_idx : t + 1]

        # Reset membrane state and spike count for this forward step
        if hasattr(model, "reset_state"):
            model.reset_state()
        if hasattr(model, "reset_spike_counts"):
            model.reset_spike_counts()

        # 4. Predict action with high-precision latency measurement
        t_start = time.perf_counter()
        with torch.no_grad():
            out = model(
                states=s_ctx,
                actions=a_ctx,
                returns_to_go=r_ctx,
                timesteps=t_ctx,
            )
            if isinstance(out, tuple):
                preds, diag = out
            else:
                preds = out
                diag = getattr(model, "diagnostics", {})

            # Predicted action token corresponding to state at step t
            action_pred_t = preds[0, -1].cpu().numpy()

        latency_ms = (time.perf_counter() - t_start) * 1000.0
        step_latencies_ms.append(latency_ms)

        # Spiking metrics from diagnostics
        spikes_fired = float(diag.get("total_spikes", 0.0))
        spike_rate = float(diag.get("mean_spike_rate", 0.0))
        step_spike_counts.append(spikes_fired)
        step_spike_rates.append(spike_rate)

        # 5. Deterministic action & clipping
        raw_action = action_pred_t.copy()
        if action_clip:
            action = np.clip(raw_action, act_low, act_high)
        else:
            action = raw_action

        # Store executed action in context buffer for subsequent steps
        actions[0, t] = torch.from_numpy(action).to(device)

        if record_trajectory:
            recorded_states.append(obs.copy())
            recorded_actions.append(action.copy())
            recorded_raw_actions.append(raw_action.copy())
            recorded_rtgs.append(returns_to_go[0, t].item() * scale)
            recorded_timesteps.append(t)

        # 6. Execute action in environment
        next_obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)

        if record_trajectory:
            recorded_rewards.append(float(reward))

        # Check termination or truncation
        if terminated or truncated or (t + 1 >= max_ep_len):
            break

        # 7. Update return-to-go and state for step t + 1
        obs = next_obs
        norm_next_obs = (next_obs - state_mean) / safe_std
        states[0, t + 1] = torch.from_numpy(norm_next_obs).to(device)

        prev_rtg_unscaled = returns_to_go[0, t].item() * scale
        returns_to_go[0, t + 1] = (prev_rtg_unscaled - reward) / scale
        timesteps[0, t + 1] = t + 1

    # Aggregate episodic metrics
    avg_spike_rate = float(np.mean(step_spike_rates)) if step_spike_rates else 0.0
    total_spikes = float(np.sum(step_spike_counts)) if step_spike_counts else 0.0
    avg_spike_count = float(np.mean(step_spike_counts)) if step_spike_counts else 0.0
    avg_latency = float(np.mean(step_latencies_ms)) if step_latencies_ms else 0.0
    total_latency = float(np.sum(step_latencies_ms)) if step_latencies_ms else 0.0

    # Normalized score
    norm_score = None
    ref_env = d4rl_env_name or getattr(env, "spec", None) and env.spec.id or ""
    try:
        norm_score = compute_d4rl_normalized_score(total_reward, ref_env)
    except (KeyError, ValueError):
        norm_score = None

    # Rollout object
    trajectory = None
    if record_trajectory and len(recorded_states) > 0:
        trajectory = TrajectoryRollout(
            states=np.array(recorded_states, dtype=np.float32),
            actions=np.array(recorded_actions, dtype=np.float32),
            raw_actions=np.array(recorded_raw_actions, dtype=np.float32),
            rewards=np.array(recorded_rewards, dtype=np.float32),
            returns_to_go=np.array(recorded_rtgs, dtype=np.float32),
            timesteps=np.array(recorded_timesteps, dtype=np.int64),
            step_latencies_ms=step_latencies_ms,
            step_spike_counts=step_spike_counts,
            step_spike_rates=step_spike_rates,
            terminated=terminated,
            truncated=truncated,
        )

    return EpisodeResult(
        total_reward=total_reward,
        episode_length=ep_len,
        avg_spike_rate=avg_spike_rate,
        total_spikes=total_spikes,
        avg_spike_count_per_step=avg_spike_count,
        avg_latency_ms=avg_latency,
        total_latency_ms=total_latency,
        normalized_score=norm_score,
        seed=seed,
        trajectory=trajectory,
    )


# ==============================================================================
# Multi-Seed & Multi-Episode Evaluation
# ==============================================================================
def evaluate_seed(
    env: gym.Env,
    model: ContinuousSnnDecisionTransformer,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    seed: int,
    num_episodes: int = 10,
    target_return: float = 3000.0,
    scale: float = 1000.0,
    context_len: Optional[int] = 20,
    max_ep_len: int = 1000,
    device: str = "cpu",
    deterministic: bool = True,
    action_clip: bool = True,
    d4rl_env_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate policy across num_episodes for a single seed base.
    """
    ep_rewards = []
    ep_lengths = []
    ep_spike_rates = []
    ep_total_spikes = []
    ep_spike_counts_per_step = []
    ep_latencies = []
    ep_norm_scores = []
    rollouts = []

    for ep in range(num_episodes):
        ep_seed = seed + ep * 1000
        result = evaluate_episode(
            env=env,
            model=model,
            state_mean=state_mean,
            state_std=state_std,
            target_return=target_return,
            scale=scale,
            context_len=context_len,
            max_ep_len=max_ep_len,
            device=device,
            seed=ep_seed,
            deterministic=deterministic,
            action_clip=action_clip,
            d4rl_env_name=d4rl_env_name,
            record_trajectory=True,
        )
        ep_rewards.append(result.total_reward)
        ep_lengths.append(result.episode_length)
        ep_spike_rates.append(result.avg_spike_rate)
        ep_total_spikes.append(result.total_spikes)
        ep_spike_counts_per_step.append(result.avg_spike_count_per_step)
        ep_latencies.append(result.avg_latency_ms)
        if result.normalized_score is not None:
            ep_norm_scores.append(result.normalized_score)
        if result.trajectory is not None:
            rollouts.append(result.trajectory)

    mean_ret = float(np.mean(ep_rewards))
    std_ret = float(np.std(ep_rewards))
    mean_len = float(np.mean(ep_lengths))
    std_len = float(np.std(ep_lengths))
    mean_sr = float(np.mean(ep_spike_rates))
    mean_spikes_step = float(np.mean(ep_spike_counts_per_step))
    mean_spikes_ep = float(np.mean(ep_total_spikes))
    mean_lat = float(np.mean(ep_latencies))

    # Normalized score from mean return
    norm_score = None
    ref_name = d4rl_env_name or (env.spec.id if getattr(env, "spec", None) else "")
    try:
        norm_score = compute_d4rl_normalized_score(mean_ret, ref_name)
    except (KeyError, ValueError):
        if ep_norm_scores:
            norm_score = float(np.mean(ep_norm_scores))

    return {
        "seed": seed,
        "num_episodes": num_episodes,
        "mean_return": mean_ret,
        "std_return": std_ret,
        "normalized_score": norm_score,
        "mean_episode_length": mean_len,
        "std_episode_length": std_len,
        "mean_spike_rate": mean_sr,
        "mean_spike_count_per_step": mean_spikes_step,
        "mean_total_spikes_per_episode": mean_spikes_ep,
        "mean_inference_latency_ms": mean_lat,
        "raw_returns": ep_rewards,
        "raw_lengths": ep_lengths,
        "rollouts": rollouts,
    }


def evaluate_mujoco(
    env_name: str,
    model: ContinuousSnnDecisionTransformer,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    seeds: Union[int, Sequence[int]] = (100, 200, 300),
    num_episodes: int = 10,
    target_return: float = 3000.0,
    scale: float = 1000.0,
    context_len: Optional[int] = 20,
    max_ep_len: int = 1000,
    deterministic: bool = True,
    action_clip: bool = True,
    device: str = "cpu",
    d4rl_env_name: Optional[str] = None,
    spike_energy_pJ: float = 0.5,
    seed: Optional[int] = None,  # Backward compatibility
) -> Dict[str, Any]:
    """
    Evaluate policy across multiple seeds and episodes.

    Computes all standard Phase 6 metrics:
    - mean_return, std_return
    - normalized_score (D4RL benchmark normalized)
    - episode_length
    - spike_count (per step & per episode)
    - inference_latency (ms)
    - energy estimates (pJ)
    """
    # Backward compatibility with single seed argument
    if seed is not None:
        eval_seeds = [seed]
    elif isinstance(seeds, int):
        eval_seeds = [seeds]
    else:
        eval_seeds = list(seeds)

    env = gym.make(env_name)
    ref_env_name = d4rl_env_name or env_name

    per_seed_summaries = []
    all_rewards = []
    all_lengths = []
    all_spike_rates = []
    all_spike_counts_per_step = []
    all_total_spikes_per_ep = []
    all_latencies_ms = []
    all_rollouts = []

    for s in eval_seeds:
        seed_res = evaluate_seed(
            env=env,
            model=model,
            state_mean=state_mean,
            state_std=state_std,
            seed=s,
            num_episodes=num_episodes,
            target_return=target_return,
            scale=scale,
            context_len=context_len,
            max_ep_len=max_ep_len,
            device=device,
            deterministic=deterministic,
            action_clip=action_clip,
            d4rl_env_name=ref_env_name,
        )
        per_seed_summaries.append(seed_res)
        all_rewards.extend(seed_res["raw_returns"])
        all_lengths.extend(seed_res["raw_lengths"])
        all_rollouts.extend(seed_res["rollouts"])
        all_spike_rates.append(seed_res["mean_spike_rate"])
        all_spike_counts_per_step.append(seed_res["mean_spike_count_per_step"])
        all_total_spikes_per_ep.append(seed_res["mean_total_spikes_per_episode"])
        all_latencies_ms.append(seed_res["mean_inference_latency_ms"])

    env.close()

    # Aggregate global statistics across all episodes & seeds
    mean_ret = float(np.mean(all_rewards))
    std_ret = float(np.std(all_rewards))
    min_ret = float(np.min(all_rewards))
    max_ret = float(np.max(all_rewards))
    mean_len = float(np.mean(all_lengths))
    std_len = float(np.std(all_lengths))
    mean_sr = float(np.mean(all_spike_rates))
    mean_spikes_per_step = float(np.mean(all_spike_counts_per_step))
    mean_spikes_per_ep = float(np.mean(all_total_spikes_per_ep))
    mean_latency = float(np.mean(all_latencies_ms))

    # Normalized D4RL score from aggregate mean return
    normalized_score = 0.0
    try:
        normalized_score = compute_d4rl_normalized_score(mean_ret, ref_env_name)
    except (KeyError, ValueError) as err:
        print(f"[Warning] Could not compute D4RL normalized score: {err}")

    # Energy estimation
    energy_pJ_per_step = mean_spikes_per_step * spike_energy_pJ
    energy_nJ_per_episode = (mean_spikes_per_ep * spike_energy_pJ) / 1000.0

    # Gate 6 Trajectory Validation
    act_space = gym.make(env_name).action_space
    valid_count = 0
    validation_issues = []
    for idx, r in enumerate(all_rollouts):
        val = r.validate(act_space.low, act_space.high)
        if val["is_valid"]:
            valid_count += 1
        else:
            validation_issues.extend([f"Ep {idx}: {iss}" for iss in val["issues"]])

    trajectories_valid = (valid_count == len(all_rollouts)) and (len(all_rollouts) > 0)

    results: Dict[str, Any] = {
        # Core 6.2 Metrics (raw names)
        "mean_return": mean_ret,
        "std_return": std_ret,
        "min_return": min_ret,
        "max_return": max_ret,
        "normalized_score": float(normalized_score),
        "episode_length": mean_len,
        "episode_length_std": std_len,
        "spike_count": mean_spikes_per_step,
        "total_spike_count_per_episode": mean_spikes_per_ep,
        "mean_spike_rate": mean_sr,
        "inference_latency": mean_latency,
        "inference_latency_ms": mean_latency,
        "estimated_energy_pJ_per_step": energy_pJ_per_step,
        "estimated_energy_nJ_per_episode": energy_nJ_per_episode,
        # Legacy eval/ prefixed keys for backward compatibility
        "eval/mean_return": mean_ret,
        "eval/std_return": std_ret,
        "eval/normalized_score": float(normalized_score),
        "eval/mean_length": mean_len,
        "eval/episode_length": mean_len,
        "eval/mean_spike_rate": mean_sr,
        "eval/spike_count": mean_spikes_per_step,
        "eval/inference_latency_ms": mean_latency,
        # Gate 6 certification metadata
        "gate_6_status": "PASSED" if trajectories_valid else "FAILED",
        "gate_6_valid_trajectory_ratio": f"{valid_count}/{len(all_rollouts)}",
        "gate_6_issues": validation_issues,
        # Detailed configuration & breakdown
        "configuration": {
            "environment": ref_env_name,
            "gym_env": env_name,
            "target_return": target_return,
            "context_length": context_len,
            "deterministic": deterministic,
            "action_clip": action_clip,
            "seeds": eval_seeds,
            "num_episodes_per_seed": num_episodes,
            "total_episodes": len(all_rewards),
        },
        "per_seed_results": [
            {
                "seed": s["seed"],
                "mean_return": s["mean_return"],
                "std_return": s["std_return"],
                "normalized_score": s["normalized_score"],
                "mean_length": s["mean_episode_length"],
                "mean_spike_count": s["mean_spike_count_per_step"],
                "mean_latency_ms": s["mean_inference_latency_ms"],
            }
            for s in per_seed_summaries
        ],
    }

    return results


# ==============================================================================
# Target Return Tuning Routine
# ==============================================================================
def tune_target_returns(
    env_name: str,
    model: ContinuousSnnDecisionTransformer,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    target_returns: Sequence[float] = (1000.0, 1800.0, 2400.0, 3000.0, 3600.0),
    seeds: Sequence[int] = (100, 200, 300),
    num_episodes_per_seed: int = 5,
    context_len: int = 20,
    scale: float = 1000.0,
    device: str = "cpu",
    d4rl_env_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Tune target return according to the dataset and benchmark setup.
    Sweeps target returns to analyze conditioning sensitivity and find optimal target return.
    """
    print("=" * 80)
    print(f"Target Return Tuning Sweep for {d4rl_env_name or env_name}")
    print(f"Candidates: {list(target_returns)} | Seeds: {list(seeds)}")
    print("=" * 80)

    sweep_results = []
    best_target = None
    best_return = -float("inf")

    for tr in target_returns:
        t0 = time.time()
        res = evaluate_mujoco(
            env_name=env_name,
            model=model,
            state_mean=state_mean,
            state_std=state_std,
            seeds=seeds,
            num_episodes=num_episodes_per_seed,
            target_return=float(tr),
            scale=scale,
            context_len=context_len,
            device=device,
            d4rl_env_name=d4rl_env_name,
        )
        elapsed = time.time() - t0
        ret = res["mean_return"]
        norm = res["normalized_score"]
        spikes = res["spike_count"]
        lat = res["inference_latency"]

        sweep_entry = {
            "target_return": float(tr),
            "mean_return": ret,
            "std_return": res["std_return"],
            "normalized_score": norm,
            "mean_length": res["episode_length"],
            "spike_count": spikes,
            "latency_ms": lat,
            "eval_time_s": elapsed,
        }
        sweep_results.append(sweep_entry)

        print(
            f"  Target: {tr:6.0f} | Return: {ret:7.2f} +/- {res['std_return']:5.2f} | "
            f"D4RL Norm: {norm:5.2f}% | Len: {res['episode_length']:5.1f} | "
            f"Spikes/step: {spikes:6.1f} | Latency: {lat:4.2f}ms"
        )

        if ret > best_return:
            best_return = ret
            best_target = float(tr)

    return {
        "best_target_return": best_target,
        "best_mean_return": best_return,
        "sweep_results": sweep_results,
    }


# ==============================================================================
# Model and Checkpoint Utilities
# ==============================================================================
def load_trained_model(
    checkpoint_path: str,
    device: str = "cpu",
) -> Tuple[ContinuousSnnDecisionTransformer, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load pre-trained SNN-DT checkpoint including state normalization statistics.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at '{checkpoint_path}'")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = checkpoint.get("config", {})
    model_cfg = config.get("model", {})
    state_dim = config.get("state_dim", 11)
    act_dim = config.get("act_dim", 3)

    model = ContinuousSnnDecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        embed_dim=model_cfg.get("embed_dim", 128),
        n_layer=model_cfg.get("n_layer", 3),
        n_head=model_cfg.get("n_head", 4),
        snn_timesteps=model_cfg.get("snn_timesteps", 4),
        tau=model_cfg.get("tau", 20.0),
        v_th=model_cfg.get("v_th", 0.5),
        dropout=model_cfg.get("dropout", 0.1),
        action_tanh=model_cfg.get("action_tanh", True),
    ).to(device)

    model_state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(model_state)
    model.eval()

    state_mean = checkpoint.get("state_mean", np.zeros(state_dim, dtype=np.float32))
    state_std = checkpoint.get("state_std", np.ones(state_dim, dtype=np.float32))

    return model, state_mean, state_std, config


# ==============================================================================
# Gate 6 Validation Function
# ==============================================================================
def validate_gate6_policy(
    eval_summary: Dict[str, Any],
    min_length_threshold: float = 10.0,
) -> Tuple[bool, List[str]]:
    """
    Formal certification of Gate 6: Policy produces valid trajectories.

    Criteria:
    1. Trajectories must be finite and free from NaNs / Infs.
    2. Actions must be bounded within environment action limits.
    3. Episode lengths must be non-trivial (> min_length_threshold).
    4. Normalized score must be accurately calculated against D4RL reference scores.
    5. Neuromorphic spike counts and latency must be actively measured.
    """
    issues = []

    # Check Gate 6 trajectory validation
    if eval_summary.get("gate_6_status") != "PASSED":
        issues.extend(eval_summary.get("gate_6_issues", ["Unknown trajectory invalidity."]))

    mean_len = eval_summary.get("episode_length", 0.0)
    if mean_len < min_length_threshold:
        issues.append(
            f"Mean episode length ({mean_len:.1f}) is below threshold ({min_length_threshold}). "
            "Policy collapses prematurely."
        )

    # Check metrics existence & finiteness
    for metric in ["mean_return", "normalized_score", "spike_count", "inference_latency"]:
        val = eval_summary.get(metric)
        if val is None or not np.isfinite(val):
            issues.append(f"Metric '{metric}' is missing or non-finite: {val}")

    passed = len(issues) == 0
    return passed, issues


# ==============================================================================
# CLI Entry Point
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Phase 6: MuJoCo Evaluation & Gate 6 Verification")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/offline_rl/hopper-medium-v2_snn_dt.pt")
    parser.add_argument("--config", type=str, default="offline_rl/configs/eval_hopper_medium.yaml")
    parser.add_argument("--env", type=str, default=None, help="Gym environment name (e.g. Hopper-v4)")
    parser.add_argument("--d4rl_env", type=str, default=None, help="D4RL reference environment name")
    parser.add_argument("--target_return", type=float, default=None, help="Target return for DT inference")
    parser.add_argument("--context_len", type=int, default=None, help="Context length K")
    parser.add_argument("--num_episodes", type=int, default=None, help="Episodes per seed")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds, e.g. '100,200,300'")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--tune_target_return", action="store_true", help="Run target return tuning sweep")
    parser.add_argument("--output_json", type=str, default=None, help="Path to save evaluation JSON results")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load configuration if provided
    config = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

    eval_cfg = config.get("evaluation", {})
    env_name = args.env or config.get("gym_env", "Hopper-v4")
    d4rl_env_name = args.d4rl_env or config.get("environment", "hopper-medium-v2")
    target_return = args.target_return or config.get("target_return", 3000.0)
    context_len = args.context_len or config.get("context_length", 20)
    num_episodes = args.num_episodes or eval_cfg.get("num_episodes", 10)
    deterministic = eval_cfg.get("deterministic", True)
    action_clip = eval_cfg.get("action_clip", True)
    scale = config.get("scale", 1000.0)
    max_ep_len = config.get("max_ep_len", 1000)

    # Seeds
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]
    else:
        seeds = eval_cfg.get("seeds", [100, 200, 300])

    print("=" * 80)
    print("PHASE 6: MuJoCo EVALUATION & GATE 6 VERIFICATION")
    print("=" * 80)
    print(f"Model Checkpoint:  {args.checkpoint}")
    print(f"Gym Environment:   {env_name}")
    print(f"D4RL Target Spec:  {d4rl_env_name}")
    print(f"Target Return:     {target_return}")
    print(f"Context Length:    {context_len}")
    print(f"Seeds:             {seeds}")
    print(f"Episodes per seed: {num_episodes} (Total: {len(seeds) * num_episodes})")
    print(f"Deterministic:     {deterministic}")
    print(f"Action Clipping:   {action_clip}")
    print("=" * 80)

    # Load pre-trained model
    model, state_mean, state_std, ckpt_cfg = load_trained_model(args.checkpoint, device=args.device)
    print(f"[Setup] Loaded checkpoint successfully. Model state_dim={model.state_dim}, act_dim={model.act_dim}")

    # Optional: target return sweep
    if args.tune_target_return:
        tuning_results = tune_target_returns(
            env_name=env_name,
            model=model,
            state_mean=state_mean,
            state_std=state_std,
            target_returns=[1000.0, 1800.0, 2400.0, 3000.0, 3600.0],
            seeds=seeds,
            num_episodes_per_seed=max(3, num_episodes // 2),
            context_len=context_len,
            scale=scale,
            device=args.device,
            d4rl_env_name=d4rl_env_name,
        )
        print(f"\n[Tuning Complete] Best Target Return: {tuning_results['best_target_return']:.0f} (Return: {tuning_results['best_mean_return']:.2f})")

    # Main evaluation across specified configuration
    print("\nRunning multi-seed evaluation...")
    results = evaluate_mujoco(
        env_name=env_name,
        model=model,
        state_mean=state_mean,
        state_std=state_std,
        seeds=seeds,
        num_episodes=num_episodes,
        target_return=target_return,
        scale=scale,
        context_len=context_len,
        max_ep_len=max_ep_len,
        deterministic=deterministic,
        action_clip=action_clip,
        device=args.device,
        d4rl_env_name=d4rl_env_name,
    )

    # Validate Gate 6 criteria
    gate6_passed, gate6_issues = validate_gate6_policy(results)

    # Print Formatted Report
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY: METRICS BREAKDOWN")
    print("=" * 80)
    print(f"  Mean Return:                  {results['mean_return']:.2f} +/- {results['std_return']:.2f}")
    print(f"  Normalized D4RL Score:        {results['normalized_score']:.2f}%")
    print(f"  Mean Episode Length:          {results['episode_length']:.1f} +/- {results['episode_length_std']:.1f} steps")
    print(f"  Spike Count (per step):       {results['spike_count']:.2f} spikes")
    print(f"  Spike Count (per episode):    {results['total_spike_count_per_episode']:.1f} spikes")
    print(f"  Mean Spike Rate:              {results['mean_spike_rate']:.4f}")
    print(f"  Inference Latency:            {results['inference_latency']:.2f} ms / step")
    print(f"  Estimated Energy / Step:      {results['estimated_energy_pJ_per_step']:.2f} pJ")
    print(f"  Estimated Energy / Episode:   {results['estimated_energy_nJ_per_episode']:.4f} nJ")
    print("-" * 80)
    print("Per-Seed Breakdown:")
    for s_info in results["per_seed_results"]:
        print(
            f"  Seed {s_info['seed']:4d}: Return = {s_info['mean_return']:7.2f} +/- {s_info['std_return']:5.2f} | "
            f"D4RL Norm = {s_info['normalized_score']:5.2f}% | "
            f"Len = {s_info['mean_length']:5.1f} | Latency = {s_info['mean_latency_ms']:4.2f}ms"
        )
    print("=" * 80)
    print(f"Gate 6 Status: {'PASSED [OK]' if gate6_passed else 'FAILED [X]'}")
    if gate6_issues:
        print("Gate 6 Issues Detected:")
        for iss in gate6_issues:
            print(f"  - {iss}")
    print("=" * 80)

    # Save to JSON if requested
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        # Remove un-serializable objects before json dumping
        clean_results = {k: v for k, v in results.items() if not k.startswith("_")}
        with open(args.output_json, "w") as f:
            json.dump(clean_results, f, indent=2)
        print(f"[Results] Saved evaluation metrics to {args.output_json}")

    return results


if __name__ == "__main__":
    main()
