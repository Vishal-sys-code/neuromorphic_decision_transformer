# src/run_mountaincar.py
"""
Run a full comparison between SNN-DT and DSF-DT on MountainCar-v0.
This script is self-contained and defensive:
 - works with gym/gymnasium reset/step API variations
 - works around NumPy 2.0 alias removal (quick shim)
 - uses dsf_collect_trajectories from src.train_dsf_dt if available (passes offline_steps & max_length)
 - otherwise falls back to a simple collector
 - builds TrajectoryDataset(trajectories, max_length, gamma)
 - trains toy SNN and DSF models (replace with your real models when ready)
"""
# ===== numpy 2.0 compatibility shim (must be before importing gym) =====
import numpy as _np
if not hasattr(_np, "float_"):
    _np.float_ = _np.float64
if not hasattr(_np, "int"):
    _np.int = int
if not hasattr(_np, "bool"):
    _np.bool = bool
if not hasattr(_np, "complex"):
    _np.complex = complex
if not hasattr(_np, "bool8"):
    _np.bool8 = _np.bool_
# =====================================================================

import argparse
import os
import pickle
import random
import time
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import gym

# Ensure repo root importability when run from src/
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import experiment config for mountaincar - adjust path/name if different
from src.configs.mountaincar_config import (
    DEVICE,
    SEED,
    offline_steps,
    dt_batch_size,
    dt_epochs,
    gamma,
    max_length,
    lr,
    dt_config,
    ENV_NAME,
)

# Try to reuse user's dsf collector if present
try:
    from src.train_dsf_dt import collect_trajectories as dsf_collect_trajectories
except Exception:
    dsf_collect_trajectories = None

CHECKPOINT_DIR = "checkpoints"
DATA_DIR = "data"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


# ---------------------------
# Utilities
# ---------------------------
def set_seed(seed: int, device: str = "cpu"):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if "cuda" in str(device).lower() and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


# ---------------------------
# Trajectory buffer & dataset
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
        self.acts.append(np.asarray(act).reshape(-1))
        self.rews.append(float(rew))

    def get_trajectory(self):
        return {
            "states": np.vstack(self.obs).astype(np.float32),
            "actions": np.vstack(self.acts).astype(np.float32),
            "rewards": np.array(self.rews, dtype=np.float32),
        }


class TrajectoryDataset(Dataset):
    def __init__(self, trajectories: List[Dict], max_length: int, gamma: float = 0.99):
        self.max_length = max_length
        self.gamma = gamma
        self.seqs = []
        for traj in trajectories:
            self._add_trajectory(traj)

    def _add_trajectory(self, traj):
        states = traj["states"]
        actions = traj["actions"]
        rewards = traj["rewards"]
        rtg = self._compute_returns_to_go(rewards, self.gamma).reshape(-1, 1)
        timesteps = np.arange(len(states)).reshape(-1, 1)
        for i in range(1, len(states) + 1):
            start = max(0, i - self.max_length)
            self.seqs.append({
                "states": states[start:i],
                "actions": actions[start:i],
                "returns_to_go": rtg[start:i],
                "timesteps": timesteps[start:i]
            })

    @staticmethod
    def _compute_returns_to_go(rewards, gamma=0.99):
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
# Toy models (replace with your real models)
# ---------------------------
class DecisionSpikeFormer(nn.Module):
    def __init__(self, state_dim, act_dim, max_length, hidden_dim=128, n_layers=2):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = int(act_dim)
        self.max_length = max_length
        self.hidden = hidden_dim
        self.state_emb = nn.Linear(state_dim, hidden_dim)
        self.pos_emb = nn.Embedding(max_length, hidden_dim)
        layers = []
        for _ in range(n_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, self.act_dim)

    def forward(self, states, actions_in=None, returns=None, returns_target=None, timesteps=None):
        B, L, _ = states.shape
        pos_idx = torch.arange(self.max_length, device=states.device).unsqueeze(0).expand(B, -1).long()
        pos_used = pos_idx[:, -L:]
        h = self.state_emb(states) + self.pos_emb(pos_used)
        h = self.mlp(h)
        action_pred = self.head(h)
        return None, action_pred, None


class SNNDecisionTransformer(nn.Module):
    def __init__(self, state_dim, act_dim, max_length, hidden_dim=128, n_layers=2, spike_thresh=0.5):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = int(act_dim)
        self.max_length = max_length
        self.hidden_dim = hidden_dim
        self.spike_thresh = spike_thresh
        self.state_emb = nn.Linear(state_dim, hidden_dim)
        self.pos_emb = nn.Embedding(max_length, hidden_dim)
        layers = []
        for _ in range(n_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        self.layers = nn.ModuleList(layers)
        self.head = nn.Linear(hidden_dim, self.act_dim)
        self.register_buffer("_spike_counter", torch.zeros(1, dtype=torch.long))

    def reset_spike_count(self):
        self._spike_counter.zero_()

    def get_spike_count(self):
        return int(self._spike_counter.item())

    def forward(self, states, actions_in=None, returns=None, returns_target=None, timesteps=None):
        B, L, _ = states.shape
        pos_idx = torch.arange(self.max_length, device=states.device).unsqueeze(0).expand(B, -1).long()
        pos_used = pos_idx[:, -L:]
        h = self.state_emb(states) + self.pos_emb(pos_used)
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


# ---------------------------
# Data collection (use dsf_collect_trajectories if available)
# ---------------------------
def collect_shared_dataset(env_name: str) -> Tuple[List[dict], int]:
    print(f"Collecting shared dataset for {env_name}...")
    if dsf_collect_trajectories is not None:
        # pass offline_steps & max_length as required by dsf_collect_trajectories
        trajectories, act_dim = dsf_collect_trajectories(env_name, offline_steps, max_length)
        print(f"Collected {len(trajectories)} trajectories (via train_dsf_dt) for {env_name}")
    else:
        # fallback collector
        env = gym.make(env_name)
        is_continuous = isinstance(env.action_space, gym.spaces.Box)
        act_dim = env.action_space.shape[0] if is_continuous else env.action_space.n
        buf = TrajectoryBuffer(max_length, env.observation_space.shape[0], act_dim if is_continuous else 1)
        trajectories = []
        steps = 0
        reset_out = env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        while steps < offline_steps:
            action = env.action_space.sample()
            step_out = env.step(action)
            if isinstance(step_out, tuple) and len(step_out) == 5:
                next_obs, reward, terminated, truncated, _ = step_out
                done = bool(terminated or truncated)
            elif isinstance(step_out, tuple) and len(step_out) == 4:
                next_obs, reward, done, _ = step_out
                done = bool(done)
            else:
                next_obs, reward, done, _ = step_out
                done = bool(done)
            store_act = np.asarray(action).reshape(-1) if is_continuous else np.asarray([int(action)])
            buf.add(np.asarray(obs).flatten(), store_act, float(reward))
            obs = next_obs
            if done:
                trajectories.append(buf.get_trajectory())
                buf.reset()
                reset_out = env.reset()
                obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            steps += 1
        env.close()
        print(f"Collected {len(trajectories)} trajectories (fallback) for {env_name}")
    # save dataset
    ensure_dir(DATA_DIR)
    out_path = os.path.join(DATA_DIR, f"shared_offline_data_{env_name}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(trajectories, f)
    print(f"Saved shared dataset -> {out_path}")
    return trajectories, act_dim


# ---------------------------
# Checkpoint helpers
# ---------------------------
def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, path: str):
    ensure_dir(os.path.dirname(path) or ".")
    torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict()}, path)


def load_checkpoint(model: nn.Module, path: str, device: str = "cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    return model


# ---------------------------
# Training + evaluation
# ---------------------------
def train_model(model_class, trajectories, act_dim, env_name, is_continuous, model_name,
                device=DEVICE, epochs=dt_epochs, batch_size=dt_batch_size, lr=lr, max_length=max_length, gamma=gamma):
    print(f"[TRAIN] {model_name} on {env_name} (device={device})")
    dataset = TrajectoryDataset(trajectories, max_length, gamma)
    train_size = int(0.8 * len(dataset))
    val_size = max(1, len(dataset) - train_size)
    train_d, val_d = random_split(dataset, [train_size, val_size]) if len(dataset) > 1 else (dataset, dataset)
    train_loader = DataLoader(train_d, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_d, batch_size=batch_size, shuffle=False)

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
        total_loss = 0.0
        steps = 0
        for batch in train_loader:
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

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            steps += 1

        avg_train = total_loss / max(1, steps)

        # validation
        model.eval()
        val_loss = 0.0
        vsteps = 0
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
                val_loss += float(loss.item())
                vsteps += 1

        avg_val = val_loss / max(1, vsteps)
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        print(f"[{model_name}] Epoch {epoch+1}/{epochs} train_loss={avg_train:.4f} val_loss={avg_val:.4f}")

        if avg_val < best_val:
            best_val = avg_val
            best_path = os.path.join(CHECKPOINT_DIR, f"best_{model_name}_{env_name}.pt")
            ensure_dir(CHECKPOINT_DIR)
            save_checkpoint(model, optimizer, best_path)
            print(f"  saved best -> {best_path}")

    if best_path:
        model = load_checkpoint(model, best_path, device=device)
    return model, history


def evaluate_model(model: nn.Module, env_name: str, n_eval: int = 20, device: str = DEVICE, seed: int = None):
    env = gym.make(env_name)
    is_continuous = isinstance(env.action_space, gym.spaces.Box)
    if seed is not None:
        try:
            env.reset(seed=seed)
        except TypeError:
            pass

    latencies = []
    returns = []
    spikes = []
    model.eval()
    if hasattr(model, "reset_spike_count"):
        try:
            model.reset_spike_count()
        except Exception:
            pass

    for _ in range(n_eval):
        reset_out = env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        done = False
        total_r = 0.0
        obs_hist = []
        while not done:
            obs_hist.append(np.asarray(obs).reshape(-1))
            seq = np.array(obs_hist[-model.max_length:], dtype=np.float32)
            pad_len = model.max_length - seq.shape[0]
            if pad_len > 0:
                seq = np.vstack([np.zeros((pad_len, seq.shape[1]), dtype=np.float32), seq])
            states_t = torch.from_numpy(seq).unsqueeze(0).to(device)
            timesteps = torch.arange(model.max_length, device=device).unsqueeze(0)
            torch.cuda.synchronize() if "cuda" in str(device).lower() and torch.cuda.is_available() else None
            t0 = time.perf_counter()
            with torch.no_grad():
                _, action_pred, _ = model(states_t, None, None, None, timesteps)
            torch.cuda.synchronize() if "cuda" in str(device).lower() and torch.cuda.is_available() else None
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)
            if is_continuous:
                act = action_pred.squeeze(0).cpu().numpy()[-1]
            else:
                logits = action_pred.squeeze(0).cpu().numpy()[-1]
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
# CLI / orchestration
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default=ENV_NAME)
    parser.add_argument("--offline-steps", type=int, default=offline_steps)
    parser.add_argument("--max-length", type=int, default=max_length)
    parser.add_argument("--epochs", type=int, default=dt_epochs)
    parser.add_argument("--batch-size", type=int, default=dt_batch_size)
    parser.add_argument("--lr", type=float, default=lr)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--recollect", action="store_true", help="Force recollecting offline data")
    args = parser.parse_args()

    set_seed(args.seed, args.device)
    ensure_dir(CHECKPOINT_DIR)
    ensure_dir(DATA_DIR)

    env_name = args.env
    data_path = os.path.join(DATA_DIR, f"shared_offline_data_{env_name}.pkl")

    # load or collect dataset
    if (not args.recollect) and os.path.exists(data_path):
        print(f"Loading dataset from {data_path}")
        with open(data_path, "rb") as f:
            trajectories = pickle.load(f)
        env = gym.make(env_name)
        is_continuous = isinstance(env.action_space, gym.spaces.Box)
        act_dim = env.action_space.shape[0] if is_continuous else env.action_space.n
        env.close()
    else:
        trajectories, act_dim = collect_shared_dataset(env_name)

    env = gym.make(env_name)
    is_continuous = isinstance(env.action_space, gym.spaces.Box)
    env.close()

    # Train SNN-DT
    print("\n=== Training SNNDecisionTransformer (toy) ===")
    snn_model, snn_hist = train_model(SNNDecisionTransformer, trajectories, act_dim, env_name, is_continuous,
                                      model_name="SNN-DT", device=args.device,
                                      epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                                      max_length=args.max_length, gamma=gamma)

    # Train DSF-DT
    print("\n=== Training DecisionSpikeFormer (toy baseline) ===")
    dsf_model, dsf_hist = train_model(DecisionSpikeFormer, trajectories, act_dim, env_name, is_continuous,
                                      model_name="DSF-DT", device=args.device,
                                      epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                                      max_length=args.max_length, gamma=gamma)

    # Evaluate
    print("\n=== Evaluating models ===")
    res_snn = evaluate_model(snn_model, env_name, n_eval=20, device=args.device, seed=args.seed)
    res_dsf = evaluate_model(dsf_model, env_name, n_eval=20, device=args.device, seed=args.seed)

    print("\nSNN result:", res_snn)
    print("\nDSF result:", res_dsf)

    # Save summary
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
    print(f"Saved summary -> {summary_path}")


if __name__ == "__main__":
    main()

