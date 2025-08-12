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
    env = gym.make(env_name)
    is_continuous = isinstance(env.action_space, gym.spaces.Box)
    act_dim = env.action_space.shape[0] if is_continuous else 1
    
    trajectories = []
    buf = TrajectoryBuffer(max_length, env.observation_space.shape[0], act_dim)
    steps = 0
    
    obs, _ = env.reset()
    
    while steps < offline_steps:
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        buf.add(obs.flatten(), np.array([action]) if not is_continuous else action, reward)
        obs = next_obs if not done else env.reset()[0]
        steps += 1
        
        if done:
            trajectories.append(buf.get_trajectory())
            buf.reset()
            
    return trajectories, env.action_space.shape[0] if is_continuous else env.action_space.n