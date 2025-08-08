"""
Offline Decision Transformer training on a Gym environment.
This script is adapted for the DecisionSpikeFormer model.
"""

import os
import sys
import random
import pickle

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

os.environ["GYM_DISABLE_ENV_CHECKER"] = "true"

import src.setup_paths
from src.config import (
    DEVICE, SEED,
    offline_steps,
    batch_size,
    dt_epochs,
    gamma,
    max_length,
    lr,
    dt_config,
)
from src.utils.trajectory_buffer import TrajectoryBuffer
from src.utils.helpers import compute_returns_to_go, simple_logger, save_checkpoint
from src.models.dsf_dt import DecisionSpikeFormer

class TrajectoryDataset(Dataset):
    def __init__(self, trajectories, max_length):
        self.seqs = []
        for traj in trajectories:
            states = traj["states"]
            actions = traj["actions"].reshape(-1,1)
            returns = compute_returns_to_go(traj["rewards"], gamma=gamma).reshape(-1,1)
            T = len(states)
            timesteps = np.arange(T).reshape(-1,1)
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
    env = gym.make(env_name)
    trajectories = []
    buf = TrajectoryBuffer(max_length, env.observation_space.shape[0], 1) # act_dim is 1 for discrete actions
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

def train_offline_dsf(env_name="CartPole-v1"):
    set_seed(SEED)
    os.makedirs("checkpoints", exist_ok=True)

    print("Collecting trajectories for DSF...")
    trajectories, act_dim_from_env = collect_trajectories(env_name)
    with open(f"offline_data_{env_name}_dsf.pkl","wb") as f:
        pickle.dump(trajectories, f)

    dataset = TrajectoryDataset(trajectories, max_length)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    dt_conf = dt_config.copy()
    dt_conf.update(
        state_dim=dataset[0]["states"].shape[-1],
        act_dim=act_dim_from_env,
        max_length=max_length,
        # dsf specific params from its config can be added here
    )
    # Add num_training_steps to the config for the model
    dt_conf['num_training_steps'] = dt_epochs * len(loader)
    dt_conf['warmup_ratio'] = 0.1 # Example, should be in config

    model = DecisionSpikeFormer(**dt_conf).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(dt_epochs):
        total_loss = 0.0
        for batch in loader:
            states = batch["states"].to(DEVICE)
            actions= batch["actions"].to(DEVICE)
            returns= batch["returns_to_go"].to(DEVICE)
            times  = batch["timesteps"].to(DEVICE)

            # dsf expects actions to be float
            actions_in = actions.to(torch.float32)

            _, action_preds, _ = model(states, actions_in, returns, times)
            
            logits = action_preds.view(-1, dt_conf["act_dim"])
            targets= actions.view(-1)
            loss = loss_fn(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        simple_logger({"epoch": epoch, "avg_offline_loss": avg_loss}, epoch)
        save_checkpoint(model, optimizer, f"checkpoints/offline_dsf_{env_name}_{epoch}.pt")

    print("Offline DecisionSpikeFormer training complete.")

if __name__ == "__main__":
    train_offline_dsf()