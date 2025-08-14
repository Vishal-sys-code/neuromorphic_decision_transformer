# src/run_cartpole.py
"""
Self-contained script to:
 - collect offline dataset for an env (CartPole-v1 by default)
 - train two small models on the shared dataset:
     * SNNDecisionTransformer (toy spiking-style model)
     * DecisionSpikeFormer (toy non-spiking transformer-like baseline)
 - evaluate both models (returns, latency, spike counts)
This script was written to replace missing class definitions and to provide a reproducible starting point.
"""

import argparse
import os
import time
import pickle
import random
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import gym

import torch

# ================= CartPole Config =================
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ENV_NAME = "CartPole-v1"

# SNN-DT Training Hyperparameters
time_window    = 5
max_length     = 20
steps_per_epoch= 5000
gamma          = 0.99
batch_size     = 1

# Offline data collection
offline_steps = 5000

# Offline DT training
dt_batch_size   = 64
dt_epochs    = 100
lr = 1e-4

# Decision Transformer model config
dt_config = {
    "hidden_size": 128,
    "n_layer": 2,
    "n_head": 1,
    "n_inner": 256,
    "activation_function": "relu",
    "n_positions": 1024,
    "resid_pdrop": 0.1,
    "attn_pdrop": 0.1,
}
# ====================================================

# ---------------------------
# Config / defaults
# ---------------------------
DEFAULT_ENV = "CartPole-v1"
DEFAULT_OFFLINE_STEPS = 2000
DEFAULT_MAX_LENGTH = 64
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 10
DEFAULT_LR = 1e-3
DEFAULT_SEED = 42
CHECKPOINT_DIR = "checkpoints"
DATA_DIR = "data"

# ---------------------------
# Utilities
# ---------------------------
def set_seed(seed: int, device: str = "cpu"):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if "cuda" in device and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # optional: deterministic flags
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

# ---------------------------
# Dataset / buffer
# ---------------------------
class TrajectoryBuffer:
    def __init__(self, max_length: int, obs_dim: int, act_dim: int):
        self.max_length = max_length
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.reset()

    def reset(self):
        self.obs = []
        self.acts = []
        self.rews = []

    def add(self, obs, act, rew):
        self.obs.append(np.asarray(obs).reshape(-1))
        # keep action as scalar or vector
        act_arr = np.asarray(act)
        self.acts.append(act_arr.reshape(-1))
        self.rews.append(float(rew))

    def get_trajectory(self):
        return {
            "states": np.vstack(self.obs).astype(np.float32),
            "actions": np.vstack(self.acts).astype(np.float32),
            "rewards": np.array(self.rews, dtype=np.float32),
        }

class TrajectoryDataset(Dataset):
    """
    Build subsequences up to max_length for offline DT training.
    trajs: list of dicts {'states', 'actions', 'rewards'}
    """
    def __init__(self, trajectories: List[Dict], max_length: int, gamma: float = 0.99):
        self.max_length = max_length
        self.gamma = gamma
        self.seqs = []  # list of dicts
        for traj in trajectories:
            self._add_trajectory(traj)

    def _add_trajectory(self, traj):
        states = traj["states"]
        actions = traj["actions"].reshape(-1, traj["actions"].shape[-1]) if len(traj["actions"].shape) > 1 else traj["actions"].reshape(-1, 1)
        # compute returns-to-go
        rewards = traj["rewards"]
        rtg = self.compute_returns_to_go(rewards, self.gamma).reshape(-1, 1)
        timesteps = np.arange(len(states)).reshape(-1, 1)
        # create all suffix subsequences
        for i in range(1, len(states) + 1):
            start = max(0, i - self.max_length)
            s = {
                "states": states[start:i],
                "actions": actions[start:i],
                "returns_to_go": rtg[start:i],
                "timesteps": timesteps[start:i],
            }
            self.seqs.append(s)

    @staticmethod
    def compute_returns_to_go(rewards, gamma=0.99):
        rtg = np.zeros_like(rewards, dtype=np.float32)
        future = 0.0
        for i in reversed(range(len(rewards))):
            future = rewards[i] + gamma * future
            rtg[i] = future
        return rtg

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        s = self.seqs[idx]
        pad_len = self.max_length - s["states"].shape[0]
        # pad with zeros at the front
        states = np.vstack([np.zeros((pad_len, s["states"].shape[1]), dtype=np.float32), s["states"]])
        actions = np.vstack([np.zeros((pad_len, s["actions"].shape[1]), dtype=np.float32), s["actions"]])
        returns = np.vstack([np.zeros((pad_len, 1), dtype=np.float32), s["returns_to_go"]])
        timesteps = np.vstack([np.zeros((pad_len, 1), dtype=np.int64), s["timesteps"]]).squeeze(-1)

        return {
            "states": torch.from_numpy(states).float(),
            "actions": torch.from_numpy(actions).float(),
            "returns_to_go": torch.from_numpy(returns).float(),
            "timesteps": torch.from_numpy(timesteps).long(),
        }

# ---------------------------
# Simple model definitions
# ---------------------------
class DecisionSpikeFormer(nn.Module):
    """
    A small MLP/attention-lite baseline to mimic `Decision Transformer` style behavior.
    Not the real DSF — substitute with official model when available.
    """
    def __init__(self, state_dim, act_dim, max_length, hidden_dim=128, n_layers=2):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = int(act_dim)
        self.max_length = max_length
        self.hidden_dim = hidden_dim

        # encode each timestep state -> embedding
        self.state_emb = nn.Linear(state_dim, hidden_dim)
        # simple positional embedding
        self.pos_emb = nn.Embedding(max_length, hidden_dim)
        # a tiny transformer-ish stack (using feed-forward blocks)
        layers = []
        for _ in range(n_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)
        # final head to predict action (use continuous output if action is Box)
        self.head = nn.Linear(hidden_dim, self.act_dim)

    def forward(self, states, actions_in=None, returns=None, returns_target=None, timesteps=None):
        # states: (B, L, state_dim)
        B, L, _ = states.shape
        # Use relative positional indices clipped to [0, max_length-1]
        pos_idx = torch.arange(self.max_length, device=states.device).unsqueeze(0).expand(B, -1).long()
        h = self.state_emb(states) + self.pos_emb(pos_idx[:, -L:])  # align last L positions with actual sequence length
        # process
        h_flat = self.mlp(h)  # (B, L, hidden)
        action_pred = self.head(h_flat)  # (B, L, act_dim)
        return None, action_pred, None

class SNNDecisionTransformer(nn.Module):
    """
    Toy spiking-like Decision Transformer.
    It uses a ReLU followed by thresholding to emulate "spikes" and counts them.
    This is NOT a real SNN implementation — use norse/snntorch for production.
    """
    def __init__(self, state_dim, act_dim, max_length, hidden_dim=128, n_layers=2, spike_thresh=0.5):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = int(act_dim)
        self.max_length = max_length
        self.hidden_dim = hidden_dim
        self.spike_thresh = spike_thresh

        self.state_emb = nn.Linear(state_dim, hidden_dim)
        self.pos_emb = nn.Embedding(max_length, hidden_dim)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU())
        self.head = nn.Linear(hidden_dim, self.act_dim)

        # spike counter buffer
        self.register_buffer("_spike_counter", torch.zeros(1, dtype=torch.long))

    def reset_spike_count(self):
        self._spike_counter.zero_()

    def forward(self, states, actions_in=None, returns=None, returns_target=None, timesteps=None):
        # states: (B, L, state_dim)
        B, L, _ = states.shape
        pos_idx = torch.arange(self.max_length, device=states.device).unsqueeze(0).expand(B, -1).long()
        # use last L positions so pos embedding size == L
        pos_emb_used = self.pos_emb(pos_idx[:, -L:])
        h = self.state_emb(states) + pos_emb_used
        spike_count = 0
        for layer in self.layers:
            h = layer(h)
            if isinstance(layer, nn.ReLU):
                sp = (h > self.spike_thresh).to(torch.long)
                spike_count += int(sp.sum().item())
        with torch.no_grad():
            self._spike_counter += int(spike_count)
        action_pred = self.head(h)
        return None, action_pred, None


    def get_spike_count(self):
        return int(self._spike_counter.item())

# ---------------------------
# Data collection (robust to gym/gymnasium differences)
# ---------------------------
def collect_trajectories(env_name: str, offline_steps: int, max_length: int, seed: int = None) -> Tuple[List[dict], int]:
    env = gym.make(env_name)
    if seed is not None:
        try:
            env.reset(seed=seed)
        except TypeError:
            # older gym
            np.random.seed(seed)
            random.seed(seed)
    is_continuous = isinstance(env.action_space, gym.spaces.Box)
    act_dim = env.action_space.shape[0] if is_continuous else env.action_space.n

    trajectories = []
    buf = TrajectoryBuffer(max_length, env.observation_space.shape[0], act_dim if is_continuous else 1)
    steps = 0

    # robust reset
    reset_out = env.reset()
    if isinstance(reset_out, tuple):
        obs = reset_out[0]
    else:
        obs = reset_out

    while steps < offline_steps:
        action = env.action_space.sample()
        step_out = env.step(action)
        # handle (obs, reward, terminated, truncated, info) or (obs, reward, done, info)
        if isinstance(step_out, tuple) and len(step_out) == 5:
            next_obs, reward, terminated, truncated, _ = step_out
            done = bool(terminated or truncated)
        elif isinstance(step_out, tuple) and len(step_out) == 4:
            next_obs, reward, done, _ = step_out
            done = bool(done)
        else:
            # fallback: try to unpack as 4
            next_obs, reward, done, _ = step_out
            done = bool(done)

        # normalize action shape
        if is_continuous:
            store_act = np.asarray(action).reshape(-1)
        else:
            store_act = np.asarray([int(action)])

        buf.add(np.asarray(obs).flatten(), store_act, float(reward))
        obs = next_obs

        if done:
            trajectories.append(buf.get_trajectory())
            buf.reset()
            reset_out = env.reset()
            if isinstance(reset_out, tuple):
                obs = reset_out[0]
            else:
                obs = reset_out

        steps += 1

    env.close()
    return trajectories, act_dim

# ---------------------------
# Training / evaluation utilities
# ---------------------------
def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, path: str):
    ensure_dir(os.path.dirname(path) or ".")
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, path)

def load_checkpoint(model: nn.Module, path: str, device: str = "cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    return model

def evaluate_model(model: nn.Module, env_name: str, n_eval: int = 20, device: str = "cpu", seed: int = None) -> dict:
    """
    Evaluate model with greedy argmax (for discrete) or direct output (for continuous).
    Returns summary dict or None on failure.
    """
    env = gym.make(env_name)
    is_continuous = isinstance(env.action_space, gym.spaces.Box)
    if seed is not None:
        try:
            env.reset(seed=seed)
        except TypeError:
            pass
    returns = []
    latencies = []
    spikes = []
    model.eval()
    if hasattr(model, "reset_spike_count"):
        try:
            model.reset_spike_count()
        except Exception:
            pass

    for ep in range(n_eval):
        reset_out = env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        done = False
        total_r = 0.0
        obs_hist = []
        t0 = None
        while not done:
            # simple one-step policy: feed last max_length states (pad with zeros)
            obs_hist.append(np.asarray(obs).flatten())
            seq = np.array(obs_hist[-model.max_length:], dtype=np.float32)
            pad_len = model.max_length - seq.shape[0]
            if pad_len > 0:
                seq = np.vstack([np.zeros((pad_len, seq.shape[1]), dtype=np.float32), seq])
            states_t = torch.from_numpy(seq).unsqueeze(0).to(device)
            timesteps = torch.arange(model.max_length, device=device).unsqueeze(0)
            torch.cuda.synchronize() if "cuda" in device and torch.cuda.is_available() else None
            t0 = time.perf_counter()
            with torch.no_grad():
                _, action_pred, _ = model(states_t, None, None, None, timesteps)
            torch.cuda.synchronize() if "cuda" in device and torch.cuda.is_available() else None
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

            # choose action
            if is_continuous:
                act = action_pred.squeeze(0).cpu().numpy()[-1]  # last timestep
            else:
                logits = action_pred.squeeze(0).cpu().numpy()[-1]  # last timestep
                act = int(np.argmax(logits))

            step_out = env.step(act)
            if isinstance(step_out, tuple) and len(step_out) == 5:
                obs, reward, terminated, truncated, _ = step_out
                done = bool(terminated or truncated)
            elif isinstance(step_out, tuple) and len(step_out) == 4:
                obs, reward, done, _ = step_out
                done = bool(done)
            else:
                obs, reward, done, _ = step_out
                done = bool(done)
            total_r += float(reward)

        returns.append(total_r)
        spikes.append(model.get_spike_count() if hasattr(model, "get_spike_count") else 0)
        if hasattr(model, "reset_spike_count"):
            try:
                model.reset_spike_count()
            except Exception:
                pass

    env.close()
    return {
        "model_name": model.__class__.__name__,
        "avg_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "avg_latency_ms": float(np.mean(latencies)),
        "avg_spikes_per_episode": float(np.mean(spikes)),
        "total_params": sum(p.numel() for p in model.parameters())
    }

# ---------------------------
# Orchestration: collect -> train -> eval
# ---------------------------
def train_model(model_class, trajectories, act_dim, env_name, is_continuous, device, epochs=10, batch_size=64, lr=1e-3, max_length=64, gamma=0.99):
    print(f"[TRAIN] Building dataset (max_length={max_length}, gamma={gamma})")
    dataset = TrajectoryDataset(trajectories, max_length, gamma)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_d, val_d = random_split(dataset, [train_size, val_size]) if len(dataset) > 1 else (dataset, dataset)
    train_loader = DataLoader(train_d, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_d, batch_size=batch_size, shuffle=False)

    # construct model config
    state_dim = dataset[0]["states"].shape[-1]
    model = (SNNDecisionTransformer if model_class is SNNDecisionTransformer else DecisionSpikeFormer)(
        state_dim=state_dim, act_dim=act_dim, max_length=max_length
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss() if is_continuous else nn.CrossEntropyLoss()

    best_val = float("inf")
    best_path = None
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        total = 0.0
        iters = 0
        for batch in train_loader:
            states = batch["states"].to(device)           # (B, L, state_dim)
            actions = batch["actions"].to(device)        # (B, L, act_dim) float
            returns = batch["returns_to_go"].to(device)
            timesteps = batch["timesteps"].to(device)

            # prepare input for model
            if is_continuous:
                actions_in = actions.float()
                targets = actions.float()
            else:
                # for discrete, assume act_dim==n (one-hot expected for training)
                # create integer targets for CrossEntropyLoss
                targets = actions.squeeze(-1).long()
                actions_in = nn.functional.one_hot(actions.squeeze(-1).long(), num_classes=act_dim).float()

            _, preds, _ = model(states, actions_in, None, returns, timesteps)
            # preds: (B, L, act_dim)
            if is_continuous:
                loss = loss_fn(preds, targets)
            else:
                logits = preds.view(-1, int(act_dim))
                loss = loss_fn(logits, targets.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += float(loss.item())
            iters += 1

        avg_train = total / max(1, iters)

        # validation
        model.eval()
        total_val = 0.0
        v_iters = 0
        with torch.no_grad():
            for batch in val_loader:
                states = batch["states"].to(device)
                actions = batch["actions"].to(device)
                returns = batch["returns_to_go"].to(device)
                timesteps = batch["timesteps"].to(device)

                if is_continuous:
                    actions_in = actions.float()
                    targets = actions.float()
                else:
                    targets = actions.squeeze(-1).long()
                    actions_in = nn.functional.one_hot(actions.squeeze(-1).long(), num_classes=act_dim).float()

                _, preds, _ = model(states, actions_in, None, returns, timesteps)
                if is_continuous:
                    loss = loss_fn(preds, targets)
                else:
                    logits = preds.view(-1, int(act_dim))
                    loss = loss_fn(logits, targets.view(-1))
                total_val += float(loss.item())
                v_iters += 1

        avg_val = total_val / max(1, v_iters)
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        print(f"[EPOCH {epoch+1}/{epochs}] train_loss={avg_train:.4f} val_loss={avg_val:.4f}")

        # save best
        if avg_val < best_val:
            best_val = avg_val
            best_path = os.path.join(CHECKPOINT_DIR, f"best_{model.__class__.__name__}_{env_name}.pt")
            ensure_dir(CHECKPOINT_DIR)
            save_checkpoint(model, optimizer, best_path)
            print(f"  -> saved best checkpoint to {best_path}")

    # load best model if present
    if best_path is not None:
        model = load_checkpoint(model, best_path, device=device)
    return model, history

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default=ENV_NAME)
    parser.add_argument("--offline-steps", type=int, default=offline_steps)
    parser.add_argument("--max-length", type=int, default=max_length)
    parser.add_argument("--epochs", type=int, default=dt_epochs) 
    parser.add_argument("--batch-size", type=int, default=batch_size) 
    parser.add_argument("--lr", type=float, default=lr) 
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--recollect", action="store_true", help="Force recollecting offline data even if file exists")
    args = parser.parse_args()

    device = args.device
    set_seed(args.seed, device)

    ensure_dir(CHECKPOINT_DIR)
    ensure_dir(DATA_DIR)

    env_name = args.env
    data_path = os.path.join(DATA_DIR, f"shared_offline_data_{env_name}.pkl")

    # collect or load dataset
    if (not args.recollect) and os.path.exists(data_path):
        print(f"Loading dataset from {data_path}")
        with open(data_path, "rb") as f:
            trajectories = pickle.load(f)
        is_continuous = isinstance(gym.make(env_name).action_space, gym.spaces.Box)
        act_dim = gym.make(env_name).action_space.shape[0] if is_continuous else gym.make(env_name).action_space.n
    else:
        print(f"Collecting shared dataset for {env_name}...")
        trajectories, act_dim = collect_trajectories(env_name, args.offline_steps, args.max_length, seed=args.seed)
        with open(data_path, "wb") as f:
            pickle.dump(trajectories, f)
        print(f"Collected {len(trajectories)} trajectories -> saved to {data_path}")

    is_continuous = isinstance(gym.make(env_name).action_space, gym.spaces.Box)

    # Train SNN-DT
    print("\n=== Training SNNDecisionTransformer (toy) ===")
    snn_model, snn_hist = train_model(SNNDecisionTransformer, trajectories, act_dim, env_name, is_continuous,
                                      device=device, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                                      max_length=args.max_length, gamma=0.99)

    # Train DecisionSpikeFormer (toy)
    print("\n=== Training DecisionSpikeFormer (toy baseline) ===")
    dsf_model, dsf_hist = train_model(DecisionSpikeFormer, trajectories, act_dim, env_name, is_continuous,
                                      device=device, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                                      max_length=args.max_length, gamma=0.99)

    # Evaluate
    print("\n=== Evaluating models ===")
    res_snn = evaluate_model(snn_model, env_name, n_eval=20, device=device, seed=args.seed)
    res_dsf = evaluate_model(dsf_model, env_name, n_eval=20, device=device, seed=args.seed)
    print("\nSNN result:", res_snn)
    print("\nDSF result:", res_dsf)

    # Save results
    summary = {
        "env": env_name,
        "snn": res_snn,
        "dsf": res_dsf,
        "snn_history": snn_hist,
        "dsf_history": dsf_hist,
    }
    summary_path = f"comparison_summary_{env_name}.pkl"
    with open(summary_path, "wb") as f:
        pickle.dump(summary, f)
    print(f"\nSummary saved to {summary_path}")

if __name__ == "__main__":
    main()

