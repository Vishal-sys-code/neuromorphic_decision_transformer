import os
import sys
import random
import pickle
import gym
import numpy as np
import torch
from torch.utils.data import Dataset

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if not hasattr(np, "bool8"): np.bool8 = np.bool_
os.environ["GYM_DISABLE_ENV_CHECKER"] = "true"

import src.setup_paths
from src.utils.trajectory_buffer import TrajectoryBuffer
from src.utils.helpers import compute_returns_to_go

class TrajectoryDataset(Dataset):
    def __init__(self, trajectories, max_length, gamma):
        self.max_length = max_length
        self.gamma = gamma
        self.seqs = []
        for traj in trajectories:
            self.add_trajectory(traj)

    def add_trajectory(self, traj):
        states = traj["states"]
        actions = traj["actions"].reshape(-1, 1)
        returns = compute_returns_to_go(traj["rewards"], self.gamma).reshape(-1, 1)
        timesteps = np.arange(len(states)).reshape(-1, 1)
        for i in range(1, len(states) + 1):
            start = max(0, i - self.max_length)
            self.seqs.append({
                "states": states[start:i], "actions": actions[start:i],
                "returns": returns[start:i], "timesteps": timesteps[start:i]
            })

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        s = self.seqs[idx]
        pad_len = self.max_length - len(s["states"])
        return {
            "states": torch.from_numpy(np.vstack([np.zeros((pad_len, s["states"].shape[1])), s["states"]])).float(),
            "actions": torch.from_numpy(np.vstack([np.zeros((pad_len, 1)), s["actions"]])).long(),
            "returns_to_go": torch.from_numpy(np.vstack([np.zeros((pad_len, 1)), s["returns"]])).float(),
            "timesteps": torch.from_numpy(np.vstack([np.zeros((pad_len, 1)), s["timesteps"]]).squeeze(-1)).long()
        }

def set_seed(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

def collect_trajectories(env_name, offline_steps, max_length):
    """
    Collect offline trajectories from `env_name` until `offline_steps` steps are gathered.
    Works with both old gym (reset() -> obs) and new gym/gymnasium (reset() -> (obs, info)),
    and supports env.step(...) returning either 4-tuple (obs, reward, done, info)
    or 5-tuple (obs, reward, terminated, truncated, info).
    """
    env = gym.make(env_name)
    is_continuous = isinstance(env.action_space, gym.spaces.Box)
    act_dim = env.action_space.shape[0] if is_continuous else 1

    trajectories = []
    buf = TrajectoryBuffer(max_length, env.observation_space.shape[0], act_dim)
    steps = 0

    # Robust reset: handle both single-return and (obs, info)
    reset_out = env.reset()
    if isinstance(reset_out, tuple):
        # e.g., (obs, info)
        obs = reset_out[0]
        info = reset_out[1] if len(reset_out) > 1 else {}
    else:
        obs = reset_out
        info = {}

    # main collection loop
    while steps < offline_steps:
        action = env.action_space.sample()

        step_out = env.step(action)
        # handle either 4-tuple or 5-tuple
        if isinstance(step_out, tuple) and len(step_out) == 5:
            next_obs, reward, terminated, truncated, info = step_out
            done = bool(terminated or truncated)
        elif isinstance(step_out, tuple) and len(step_out) == 4:
            next_obs, reward, done, info = step_out
            done = bool(done)
        else:
            # fallback: try to unpack safely
            try:
                next_obs, reward, done, info = step_out
                done = bool(done)
            except Exception:
                raise RuntimeError(f"Unexpected env.step() return shape: {type(step_out)} {step_out}")

        # normalize shapes for buffer
        if not is_continuous:
            # ensure action stored as scalar or shape-consistent array
            act_to_store = np.array([action])
        else:
            act_to_store = np.asarray(action)

        buf.add(np.asarray(obs).flatten(), act_to_store, float(reward))
        obs = next_obs

        if done:
            trajectories.append(buf.get_trajectory())
            buf.reset()
            # reset env robustly again
            reset_out = env.reset()
            if isinstance(reset_out, tuple):
                obs = reset_out[0]
                info = reset_out[1] if len(reset_out) > 1 else {}
            else:
                obs = reset_out
                info = {}

        steps += 1

    # Return collected trajectories and action dimension (for discrete envs, return n)
    if is_continuous:
        return trajectories, env.action_space.shape[0]
    else:
        return trajectories, env.action_space.n

