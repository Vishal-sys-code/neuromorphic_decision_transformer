"""
MuJoCo Evaluation Pipeline for Decision Transformer policies.

Rolls out the SNN Decision Transformer in a Gymnasium MuJoCo environment,
maintaining the autoregressive context buffer (R, s, a), updating returns-to-go,
and logging cumulative episodic returns and normalized scores.
"""

from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch

from offline_rl.models.snn_dt_mujoco import ContinuousSnnDecisionTransformer

# D4RL reference score bounds for score normalization: 100 * (score - random) / (expert - random)
D4RL_REF_SCORES = {
    "hopper-medium-v2": {"random": -20.272305, "expert": 3234.3},
    "hopper-medium-replay-v2": {"random": -20.272305, "expert": 3234.3},
    "hopper-medium-expert-v2": {"random": -20.272305, "expert": 3234.3},
    "halfcheetah-medium-v2": {"random": -280.178953, "expert": 12135.0},
    "halfcheetah-medium-expert-v2": {"random": -280.178953, "expert": 12135.0},
    "walker2d-medium-v2": {"random": 1.629008, "expert": 4592.3},
    "walker2d-medium-expert-v2": {"random": 1.629008, "expert": 4592.3},
    "ant-medium-v2": {"random": -325.6, "expert": 3880.0},
}


def evaluate_episode(
    env: gym.Env,
    model: ContinuousSnnDecisionTransformer,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    target_return: float = 3600.0,
    scale: float = 1000.0,
    max_ep_len: int = 1000,
    device: str = "cpu",
) -> Tuple[float, int, float]:
    """
    Run a single evaluation episode with autoregressive SNN-DT policy.
    Returns:
        (total_reward, episode_length, avg_spike_rate)
    """
    model.eval()
    obs, info = env.reset()
    state_dim = obs.shape[0]
    act_dim = env.action_space.shape[0]

    states = torch.zeros((1, max_ep_len, state_dim), dtype=torch.float32, device=device)
    actions = torch.zeros((1, max_ep_len, act_dim), dtype=torch.float32, device=device)
    returns_to_go = torch.zeros((1, max_ep_len, 1), dtype=torch.float32, device=device)
    timesteps = torch.zeros((1, max_ep_len), dtype=torch.long, device=device)

    # Initialize first step
    norm_obs = (obs - state_mean) / state_std
    states[0, 0] = torch.from_numpy(norm_obs).to(device)
    returns_to_go[0, 0] = target_return / scale
    timesteps[0, 0] = 0

    total_reward = 0.0
    ep_len = 0
    spike_rates = []

    for t in range(max_ep_len):
        ep_len += 1

        # Context window slice [0 : t+1]
        start_idx = max(0, t + 1 - model.context_len)
        s_ctx = states[:, start_idx : t + 1]
        a_ctx = actions[:, start_idx : t + 1]
        r_ctx = returns_to_go[:, start_idx : t + 1]
        t_ctx = timesteps[:, start_idx : t + 1]

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
            # Action corresponding to state token at timestep t
            action_pred = preds[0, -1].cpu().numpy()
            spike_rates.append(diag.get("mean_spike_rate", 0.0))

        # Clip action to environment bounds
        action = np.clip(action_pred, env.action_space.low, env.action_space.high)
        actions[0, t] = torch.from_numpy(action).to(device)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        if terminated or truncated or (t + 1 >= max_ep_len):
            break

        # Next step state and target return update
        norm_next_obs = (next_obs - state_mean) / state_std
        states[0, t + 1] = torch.from_numpy(norm_next_obs).to(device)
        prev_rtg = returns_to_go[0, t].item() * scale
        returns_to_go[0, t + 1] = (prev_rtg - reward) / scale
        timesteps[0, t + 1] = t + 1

    avg_spike_rate = float(np.mean(spike_rates)) if spike_rates else 0.0
    return total_reward, ep_len, avg_spike_rate


def evaluate_mujoco(
    env_name: str,
    model: ContinuousSnnDecisionTransformer,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    num_episodes: int = 10,
    target_return: float = 3600.0,
    scale: float = 1000.0,
    max_ep_len: int = 1000,
    device: str = "cpu",
    seed: int = 42,
    d4rl_env_name: Optional[str] = None,
) -> Dict[str, float]:
    """
    Evaluate policy across multiple episodes and return mean metrics.
    """
    env = gym.make(env_name)
    rewards = []
    lengths = []
    spike_rates = []

    for ep in range(num_episodes):
        env.reset(seed=seed + ep)
        ep_reward, ep_len, sr = evaluate_episode(
            env=env,
            model=model,
            state_mean=state_mean,
            state_std=state_std,
            target_return=target_return,
            scale=scale,
            max_ep_len=max_ep_len,
            device=device,
        )
        rewards.append(ep_reward)
        lengths.append(ep_len)
        spike_rates.append(sr)

    env.close()

    mean_ret = float(np.mean(rewards))
    std_ret = float(np.std(rewards))
    mean_len = float(np.mean(lengths))
    mean_sr = float(np.mean(spike_rates))

    metrics = {
        "eval/mean_return": mean_ret,
        "eval/std_return": std_ret,
        "eval/mean_length": mean_len,
        "eval/mean_spike_rate": mean_sr,
    }

    ref_key = d4rl_env_name or env_name.lower()
    if ref_key in D4RL_REF_SCORES:
        ref = D4RL_REF_SCORES[ref_key]
        normalized_score = 100.0 * (mean_ret - ref["random"]) / (ref["expert"] - ref["random"])
        metrics["eval/normalized_score"] = float(normalized_score)

    return metrics
