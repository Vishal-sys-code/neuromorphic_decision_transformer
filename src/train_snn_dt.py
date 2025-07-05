"""
Author: Vishal Pandey
Email: pandeyvishal.mlprof@gmail.com

Offline Decision Transformer training on a Gym environment.
Collect trajectories first, then train in batch (return-conditioned).
"""

# Offline Decision Transformer Training
import os
import sys # Added sys import
import random
import pickle

# --- Prepend Project Root to sys.path ---
# This allows Python to find the 'src' package when running this script directly,
# especially in environments like Kaggle or when the script is not run as a module from the root.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path: # Restoring this crucial part
    sys.path.insert(0, PROJECT_ROOT) # Restoring this crucial part
# --- End sys.path modification ---

import gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

os.environ["GYM_DISABLE_ENV_CHECKER"] = "true"

# ensure our packages are on PYTHONPATH
import src.setup_paths

from src.config import (
    DEVICE, SEED,
    offline_steps,   # total env steps to collect
    batch_size,
    dt_epochs,       # training epochs on the offline dataset
    gamma,
    max_length,
    lr,
    dt_config,
)
from src.utils.trajectory_buffer import TrajectoryBuffer
from src.utils.helpers import compute_returns_to_go, simple_logger, save_checkpoint
# from src.models.snn_dt_patch import SNNDecisionTransformer  # or DecisionTransformer
# from .models.snn_dt_patch import SNNDecisionTransformer  # or DecisionTransformer
# from ..src.models.snn_dt_patch import SNNDecisionTransformer  # or DecisionTransformer # Original problematic import
from src.models.snn_dt_patch import SNNDecisionTransformer  # Corrected import
# Uncomment the following line to use the original DecisionTransformer
# from src.models.snn_dt_gpt2_attention import SNNDecisionTransformer  # or DecisionTransformer

# 1) Define an offline dataset of DT sequences
class TrajectoryDataset(Dataset):
    def __init__(self, trajectories, max_length):
        """
        trajectories: list of dicts with 'states', 'actions', 'rewards'
        """
        self.seqs = []
        for traj in trajectories:
            states = traj["states"]
            actions = traj["actions"].reshape(-1,1)
            returns = compute_returns_to_go(traj["rewards"], gamma=gamma).reshape(-1,1)
            T = len(states)
            timesteps = np.arange(T).reshape(-1,1)
            # pad up to max_length on the left
            for i in range(1, T+1):
                start = max(0, i - max_length)
                self.seqs.append({
                    "states":    states[start:i],
                    "actions":   actions[start:i],
                    "returns":   returns[start:i],
                    "timesteps": timesteps[start:i],
                })

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        s = self.seqs[idx]
        # pad each field to max_length
        pad_len = max_length - len(s["states"])
        pad_s = np.zeros((pad_len, s["states"].shape[1]), dtype=np.float32)
        pad_a = np.zeros((pad_len, s["actions"].shape[1]), dtype=np.int64)
        pad_r = np.zeros((pad_len, 1), dtype=np.float32)
        pad_t = np.zeros((pad_len, 1), dtype=np.int64)

        states = np.vstack([pad_s, s["states"]])
        actions= np.vstack([pad_a, s["actions"]])
        returns= np.vstack([pad_r, s["returns"]])
        timesteps = np.vstack([pad_t, s["timesteps"]])

        return {
            "states": torch.tensor(states, dtype=torch.float32),
            "actions": torch.tensor(actions, dtype=torch.long),
            "returns_to_go": torch.tensor(returns, dtype=torch.float32),
            "timesteps": torch.tensor(timesteps.squeeze(-1), dtype=torch.long),
        }

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)

def collect_trajectories(env_name="CartPole-v1"):
    """Collect offline_steps of data with a random policy."""
    env = gym.make(env_name)
    trajectories = []
    buf = TrajectoryBuffer(max_length, env.observation_space.shape[0], env.action_space.n)
    steps = 0
    obs = env.reset()[0]
    while steps < offline_steps:
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buf.add(np.array(obs, dtype=np.float32), action, reward)
        obs = next_obs if not done else env.reset()[0]
        steps += 1
        if done:
            trajectories.append(buf.get_trajectory())
            buf.reset()
    return trajectories, env.action_space.n

def train_offline_dt(env_name="CartPole-v1"):
    set_seed(SEED)
    os.makedirs("checkpoints", exist_ok=True)

    # 1. Collect & save
    print("Collecting trajectories...")
    trajectories, act_dim_from_env = collect_trajectories(env_name)
    # Save only trajectories, not act_dim
    with open("offline_data.pkl","wb") as f:
        pickle.dump(trajectories, f)

    # 2. Build dataset & loader
    dataset = TrajectoryDataset(trajectories, max_length)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 3. Model & optimizer
    dt_conf = dt_config.copy()
    dt_conf.update(
        state_dim=dataset[0]["states"].shape[-1],
        act_dim=act_dim_from_env, # Use action dim from environment
        max_length=max_length,
    )
    model = SNNDecisionTransformer(**dt_conf).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # 4. Offline training epochs
    for epoch in range(dt_epochs):
        total_loss = 0.0
        for batch in loader:
            states = batch["states"].to(DEVICE)           # [B, S, state_dim]
            actions= batch["actions"].to(DEVICE)          # [B, S, 1]
            returns= batch["returns_to_go"].to(DEVICE)    # [B, S, 1]
            times  = batch["timesteps"].to(DEVICE)        # [B, S]

            # one-hot actions for input embedding
            actions_in = torch.nn.functional.one_hot(
                actions.squeeze(-1), num_classes=dt_conf["act_dim"]
            ).to(torch.float32)

            # forward: predict next actions
            _, action_preds, _ = model(states, actions_in, None, returns, times)
            # compute CE loss on all positions
            logits = action_preds.view(-1, dt_conf["act_dim"])
            targets= actions.view(-1)
            loss = loss_fn(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        simple_logger({"epoch": epoch, "avg_offline_loss": avg_loss}, epoch)
        save_checkpoint(model, optimizer, f"checkpoints/offline_dt_{env_name}_{epoch}.pt")

    print("✅ Offline Decision Transformer training complete.")

if __name__ == "__main__":
    train_offline_dt()