# src/train_offline_dt.py
"""
Interactive Offline Decision Transformer training on a chosen Gym environment.
1=CartPole-v1, 2=MountainCar-v0, 3=LunarLander-v2, 4=Acrobot-v1, 5=Pendulum-v1, 6=Exit
"""
import os, sys, argparse, pickle, random
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Patch numpy for Gym’s checker
import numpy as np
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

if not hasattr(np, "float_"):
    np.float_ = np.float64
    
os.environ["GYM_DISABLE_ENV_CHECKER"] = "true"

import gym
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import src.setup_paths
from src.config import DEVICE, SEED, offline_steps, batch_size, dt_epochs, gamma, max_length, lr, dt_config
from src.utils.trajectory_buffer import TrajectoryBuffer
from src.utils.helpers import compute_returns_to_go, simple_logger
from src.models.snn_dt_patch import SNNDecisionTransformer

ENV_MAP = {
    "1": "CartPole-v1",
    "2": "MountainCar-v0",
    "3": "LunarLander-v2",
    "4": "Acrobot-v1",
    "5": "Pendulum-v1",
}


class TrajectoryDataset(Dataset):
    def __init__(self, trajectories):
        self.seqs = []
        for traj in trajectories:
            states  = traj["states"]                                         # [T, S]
            actions = traj["actions"]                                        # [T,] or [T,dim]
            returns = compute_returns_to_go(traj["rewards"], gamma=gamma).reshape(-1,1)
            timesteps = np.arange(len(states)).reshape(-1,1)
            T = len(states)
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
        L = len(s["states"])
        pad = max_length - L

        pad_s = np.zeros((pad, s["states"].shape[1]), dtype=np.float32)
        pad_r = np.zeros((pad, 1), dtype=np.float32)
        pad_t = np.zeros((pad, 1), dtype=np.int64)

        states    = np.vstack([pad_s, s["states"]]).astype(np.float32)
        returns   = np.vstack([pad_r, s["returns"].astype(np.float32)])
        timesteps = np.vstack([pad_t, s["timesteps"]]).astype(np.int64)

        # actions may be shape [L,] or [L,dim]
        a = s["actions"]
        if a.ndim == 1:
            pad_a = np.zeros((pad,), dtype=np.int64)
            actions = np.concatenate([pad_a, a]).astype(np.int64).reshape(-1,1)
        else:
            dim = a.shape[1]
            pad_a = np.zeros((pad, dim), dtype=np.float32)
            actions = np.vstack([pad_a, a.astype(np.float32)])

        return {
            "states":        torch.from_numpy(states).to(DEVICE),            # [L, S]
            "actions":       torch.from_numpy(actions).to(DEVICE),           # [L, 1] or [L,dim]
            "returns_to_go": torch.from_numpy(returns).to(DEVICE),           # [L,1]
            "timesteps":     torch.from_numpy(timesteps.squeeze(-1)).to(DEVICE),  # [L]
        }


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)


def collect_trajectories(env_name, state_dim, act_dim):
    """Collects offline_steps env steps with a random policy."""
    env = gym.make(env_name)
    trajs = []
    buf = TrajectoryBuffer(max_length, state_dim, act_dim)
    steps = 0
    obs = env.reset()[0]
    while steps < offline_steps:
        action = env.action_space.sample()
        nobs, rew, term, trunc, _ = env.step(action)
        done = term or trunc
        buf.add(obs.astype(np.float32), action, rew)
        obs = nobs if not done else env.reset()[0]
        steps += 1
        if done:
            trajs.append(buf.get_trajectory())
            buf.reset()
    return trajs


def train_offline_dt(env_name):
    set_seed(SEED)
    os.makedirs("checkpoints", exist_ok=True)

    # create env once to infer dims
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    if hasattr(env.action_space, "n"):
        act_dim  = env.action_space.n
        act_type = "discrete"
    else:
        act_dim  = env.action_space.shape[0]
        act_type = "continuous"

    print(f"\nCollecting {offline_steps} steps from {env_name} …")
    trajectories = collect_trajectories(env_name, state_dim, act_dim)
    with open(f"offline_data_{env_name}.pkl","wb") as f:
        pickle.dump(trajectories, f)

    dataset = TrajectoryDataset(trajectories)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    cfg = dt_config.copy()
    cfg.update(state_dim=state_dim, act_dim=act_dim, max_length=max_length)
    model = SNNDecisionTransformer(**cfg).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss() if act_type=="discrete" else nn.MSELoss()

    print(f"Training offline DT on {env_name} ({act_type}) for {dt_epochs} epochs …")
    for epoch in range(dt_epochs):
        total_loss = 0.0
        for batch in loader:
            # B + batch_size
            states = batch["states"]        # [B, L, S]
            actions = batch["actions"]      # [B, L, 1] or [B, L, dim]
            returns = batch["returns_to_go"]  # [B, L, 1]
            times   = batch["timesteps"]    # [B, L]

            if act_type == "discrete":
                a_idx = actions.squeeze(-1).long()        # [B, L]
                a_in  = nn.functional.one_hot(a_idx, act_dim).to(torch.float32)
                _, action_preds, _ = model(states, a_in, None, returns, times)
                logits  = action_preds.reshape(-1, act_dim)
                targets = a_idx.reshape(-1) 
                loss    = loss_fn(logits, targets)
            else:
                # continuous: use raw actions as input & target
                _, action_preds, _ = model(states, actions, None, returns, times)
                loss = loss_fn(action_preds, actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg = total_loss / len(loader)
        simple_logger({"epoch": epoch, "loss": avg}, epoch)
        torch.save({
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict()
        }, f"checkpoints/offline_dt_{env_name}_{epoch}.pt")

    print("✅ Offline DT training complete on", env_name)


if __name__ == "__main__":
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--env", type=str, help="skip menu")
    args,_ = p.parse_known_args()

    if args.env:
        choice = args.env
    else:
        while True:
            print("\nSelect environment:")
            for k,v in ENV_MAP.items():
                print(f"  {k}. {v}")
            print("  6. Exit")
            sel = input("Enter choice: ").strip()
            if sel == "6":
                sys.exit(0)
            if sel in ENV_MAP:
                choice = ENV_MAP[sel]
                break
            print("Invalid—try again.")

    train_offline_dt(choice)
