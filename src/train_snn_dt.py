import os
import sys
import random
import pickle
import argparse
import gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if not hasattr(np, "bool8"): np.bool8 = np.bool_
os.environ["GYM_DISABLE_ENV_CHECKER"] = "true"

import src.setup_paths
from src.utils.trajectory_buffer import TrajectoryBuffer
from src.utils.helpers import compute_returns_to_go, simple_logger, save_checkpoint, evaluate_model
from src.models.snn_dt_patch import SNNDecisionTransformer

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

# (The rest of the file is removed as its functionality is now in run_experiment.py)
