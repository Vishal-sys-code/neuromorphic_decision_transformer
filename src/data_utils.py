"""
Minimal data_utils.py providing dsf_collect_trajectories(...)

This provides a safe default implementation for the helper used by
run_experiment.py. It collects a mixture of "expert" and random rollouts
until `offline_steps` environment steps are gathered, and segments them
into episodes/clips of at most `max_length`.

Each trajectory is a dict:
  {
    "observations": [obs0, obs1, ...],
    "actions": [a0, a1, ...],
    "rewards": [r0, r1, ...],
    "dones": [d0, d1, ...]   # optional but helpful
  }

Returns:
  (trajectories, act_dim)
"""

import gym
import numpy as np
import random
from collections import deque
from typing import List, Tuple, Dict


def cartpole_heuristic_action(obs):
    # obs = [x, x_dot, theta, theta_dot]
    theta = obs[2]
    # push right if pole angle > 0
    return 1 if theta > 0 else 0


def pendulum_heuristic_action(obs):
    # obs is [-cos(theta), -sin(theta), theta_dot] in Gym Pendulum-v1
    # approximate angle and angular velocity
    cos_th, sin_th, thdot = obs
    theta = np.arctan2(sin_th, cos_th)
    # simple PD controller: torque = -k_p * theta - k_d * theta_dot
    kp, kd = 2.0, 0.1
    torque = -kp * theta - kd * thdot
    # clip to action space [-2, 2]
    torque = float(np.clip(torque, -2.0, 2.0))
    return np.array([torque], dtype=np.float32)


def default_expert_action(env, obs):
    """Return an 'expert' action if we have a heuristic for the env, else None."""
    name = env.spec.id
    if name.startswith("CartPole"):
        return cartpole_heuristic_action(obs)
    if name.startswith("Pendulum"):
        return pendulum_heuristic_action(obs)
    # For other envs (MountainCar/Acrobot) we do not have a heuristic here.
    return None


def dsf_collect_trajectories(env_name: str,
                             offline_steps: int = 10000,
                             max_length: int = 50,
                             expert_frac: float = 0.5,
                             seed: int = 42) -> Tuple[List[Dict], int]:
    """
    Collect trajectories for offline training.

    Args:
      env_name: gym environment id, e.g. "CartPole-v1"
      offline_steps: total number of env steps to collect
      max_length: maximum timesteps per stored trajectory/clip
      expert_frac: fraction of offline_steps to use expert (heuristic) policy;
                   remaining fraction is random actions.
      seed: random seed.

    Returns:
      trajectories: list of dicts with keys ('observations','actions','rewards','dones')
      act_dim: dimension of action (int) for continuous or discrete:
               - for discrete: env.action_space.n
               - for continuous: env.action_space.shape[0]
    """

    random.seed(seed)
    np.random.seed(seed)

    env = gym.make(env_name)
    env.seed(seed)
    obs_space = env.observation_space
    act_space = env.action_space

    # action dimension
    if isinstance(act_space, gym.spaces.Discrete):
        act_dim = int(act_space.n)
    else:
        act_dim = int(np.prod(act_space.shape))

    trajectories = []
    steps_collected = 0
    expert_steps_target = int(offline_steps * float(expert_frac))
    random_steps_target = offline_steps - expert_steps_target

    def collect_rollouts(target_steps, policy_type="random"):
        nonlocal steps_collected, trajectories
        steps = 0
        while steps < target_steps:
            obs = env.reset()
            ep_obs = []
            ep_actions = []
            ep_rewards = []
            ep_dones = []
            for t in range(max_length):
                if policy_type == "expert":
                    a = default_expert_action(env, obs)
                    if a is None:
                        # fallback to random if no heuristic available
                        if isinstance(act_space, gym.spaces.Discrete):
                            a = env.action_space.sample()
                        else:
                            a = env.action_space.sample()
                    # ensure correct dtype/shape
                    if isinstance(act_space, gym.spaces.Box):
                        a = np.array(a, dtype=act_space.dtype)
                    else:
                        a = int(a)
                else:
                    # random policy
                    a = env.action_space.sample()

                next_obs, r, done, info = env.step(a)

                ep_obs.append(np.array(obs, dtype=np.float32))
                # store action as scalar or array depending on action space
                if isinstance(act_space, gym.spaces.Box):
                    ep_actions.append(np.array(a, dtype=np.float32))
                else:
                    ep_actions.append(int(a))
                ep_rewards.append(float(r))
                ep_dones.append(bool(done))

                obs = next_obs
                steps += 1
                steps_collected += 1

                if done or steps >= target_steps:
                    break

            trajectories.append({
                "observations": ep_obs,
                "actions": ep_actions,
                "rewards": ep_rewards,
                "dones": ep_dones
            })

            # safety break in case something odd (shouldn't happen)
            if steps_collected >= offline_steps:
                break

    # collect expert-ish rollouts first (where heuristic exists; otherwise random)
    if expert_steps_target > 0:
        collect_rollouts(expert_steps_target, policy_type="expert")
    # then collect the rest as random
    if random_steps_target > 0:
        collect_rollouts(random_steps_target, policy_type="random")

    env.close()

    # Simple postprocessing: if any trajectory length is > max_length, split it.
    processed = []
    for traj in trajectories:
        L = len(traj["observations"])
        if L <= max_length:
            processed.append(traj)
        else:
            # slice into chunks
            for i in range(0, L, max_length):
                chunk = {
                    "observations": traj["observations"][i:i+max_length],
                    "actions": traj["actions"][i:i+max_length],
                    "rewards": traj["rewards"][i:i+max_length],
                    "dones": traj["dones"][i:i+max_length]
                }
                processed.append(chunk)

    # If processed is empty for some reason, no-op fallback: generate a tiny random trajectory
    if len(processed) == 0:
        obs = env.reset()
        ep_obs = [np.array(obs, dtype=np.float32)]
        a = env.action_space.sample()
        ep_actions = [np.array(a, dtype=np.float32) if isinstance(act_space, gym.spaces.Box) else int(a)]
        ep_rewards = [0.0]
        ep_dones = [True]
        processed.append({
            "observations": ep_obs,
            "actions": ep_actions,
            "rewards": ep_rewards,
            "dones": ep_dones
        })

    # Return processed trajectories and action dimension
    return processed, act_dim
