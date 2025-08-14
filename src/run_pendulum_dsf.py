# src/run_pendulum_dsf.py
"""
Standalone runner for DecisionSpikeFormer (DSF-DT) on Pendulum-v1 (DSF only).
Defensive: handles numpy 2.0, gym API differences, dataset/model signatures.
"""

# ==== numpy 2.0 compatibility shim (must run BEFORE importing gym) ====
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
# ===================================================================

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
import sys

# Ensure repo root importability when run from src/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Config for pendulum (adjust path if your config lives elsewhere)
from src.configs.pendulum_config import (
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

# Attempt to import DSF helpers/model from your repo
try:
    from src.train_dsf_dt import collect_trajectories as dsf_collect_trajectories
    from src.train_dsf_dt import TrajectoryDataset as DSFTrajectoryDataset
    from src.train_dsf_dt import set_seed as dsf_set_seed
except Exception:
    dsf_collect_trajectories = None
    DSFTrajectoryDataset = None
    dsf_set_seed = None

try:
    from src.models.dsf_dt import DecisionSpikeFormer as RealDecisionSpikeFormer
except Exception:
    RealDecisionSpikeFormer = None

# Toy fallback DSF
class ToyDecisionSpikeFormer(nn.Module):
    def __init__(self, state_dim, act_dim, max_length, hidden_dim=128, n_layers=2):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = int(act_dim)
        self.max_length = max_length
        self.state_emb = nn.Linear(state_dim, hidden_dim)
        self.pos_emb = nn.Embedding(max_length, hidden_dim)
        self.mlp = nn.Sequential(*[nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()) for _ in range(n_layers)])
        self.head = nn.Linear(hidden_dim, self.act_dim)
        self.register_buffer("_spike_counter", torch.zeros(1, dtype=torch.long))

    def reset_spike_count(self):
        self._spike_counter.zero_()

    def get_spike_count(self):
        return int(self._spike_counter.item())

    def forward(self, states, actions_in=None, returns_in=None, returns_target=None, timesteps=None):
        B, L, _ = states.shape
        pos_idx = torch.arange(self.max_length, device=states.device).unsqueeze(0).expand(B, -1).long()
        pos_used = pos_idx[:, -L:]
        h = self.state_emb(states) + self.pos_emb(pos_used)
        h = self.mlp(h)
        with torch.no_grad():
            self._spike_counter += torch.tensor(int((h > 0.5).sum().item()), dtype=torch.long, device=self._spike_counter.device)
        action_pred = self.head(h)
        return None, action_pred, None

# Fallback dataset (continuous-friendly)
class FallbackTrajectoryDataset(Dataset):
    def __init__(self, trajectories: List[Dict], max_length: int, gamma: float = 0.99):
        self.max_length = max_length
        self.gamma = gamma
        self.seqs = []
        for traj in trajectories:
            self._add_trajectory(traj)

    def _compute_rtg(self, rewards):
        rtg = np.zeros_like(rewards, dtype=np.float32)
        future = 0.0
        for i in reversed(range(len(rewards))):
            future = rewards[i] + self.gamma * future
            rtg[i] = future
        return rtg

    def _add_trajectory(self, traj):
        states = traj["states"]
        actions = traj["actions"]
        rewards = traj.get("rewards", np.zeros(len(states), dtype=np.float32))
        rtg = self._compute_rtg(rewards).reshape(-1, 1)
        timesteps = np.arange(len(states)).reshape(-1, 1)
        for i in range(1, len(states) + 1):
            start = max(0, i - self.max_length)
            self.seqs.append({
                "states": states[start:i],
                "actions": actions[start:i],
                "returns_to_go": rtg[start:i],
                "timesteps": timesteps[start:i],
            })

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

# Helpers
CHECKPOINT_DIR = "checkpoints"
DATA_DIR = "data"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

def set_seed_local(seed: int, device: str = "cpu"):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if "cuda" in str(device).lower() and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, path: str):
    torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict()}, path)

def load_checkpoint(model: nn.Module, path: str, device: str = "cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    return model

# Robust gym wrappers
def gym_reset(env):
    out = env.reset()
    return out[0] if isinstance(out, tuple) else out

def gym_step(env, action):
    out = env.step(action)
    if isinstance(out, tuple) and len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
        return obs, reward, done, info
    elif isinstance(out, tuple) and len(out) == 4:
        obs, reward, done, info = out
        done = bool(done)
        return obs, reward, done, info
    else:
        obs, reward, done, info = out
        done = bool(done)
        return obs, reward, done, info

# Collect dataset (prefer repo collector if present)
def collect_shared_dataset(env_name: str, offline_steps: int, max_length_local: int) -> Tuple[List[dict], int]:
    print(f"Collecting shared dataset for {env_name}...")
    if dsf_collect_trajectories is not None:
        trajectories, act_dim = dsf_collect_trajectories(env_name, offline_steps, max_length_local)
        print(f"Collected {len(trajectories)} trajectories (via train_dsf_dt) for {env_name}")
    else:
        env = gym.make(env_name)
        is_continuous = isinstance(env.action_space, gym.spaces.Box)
        act_dim = env.action_space.shape[0] if is_continuous else env.action_space.n
        trajectories = []
        obs = gym_reset(env)
        steps = 0
        cur_obs = []
        cur_act = []
        cur_rew = []
        while steps < offline_steps:
            action = env.action_space.sample()
            next_obs, reward, done, _ = gym_step(env, action)
            cur_obs.append(np.asarray(obs).reshape(-1))
            cur_act.append(np.asarray(action).reshape(-1))
            cur_rew.append(float(reward))
            obs = next_obs
            if done:
                trajectories.append({
                    "states": np.vstack(cur_obs).astype(np.float32),
                    "actions": np.vstack(cur_act).astype(np.float32),
                    "rewards": np.array(cur_rew, dtype=np.float32)
                })
                cur_obs, cur_act, cur_rew = [], [], []
                obs = gym_reset(env)
            steps += 1
        env.close()
        print(f"Collected {len(trajectories)} trajectories (fallback) for {env_name}")

    out_path = os.path.join(DATA_DIR, f"shared_offline_data_{env_name}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(trajectories, f)
    print(f"Saved shared dataset -> {out_path}")
    return trajectories, act_dim

# Training loop for DSF only
def train_dsf(trajectories, act_dim, env_name,
              device: str = DEVICE,
              epochs: int = dt_epochs,
              batch_size: int = dt_batch_size,
              lr_val: float = lr,
              max_length_local: int = max_length,
              gamma_local: float = gamma,
              seed: int = SEED):
    print(f"[TRAIN] DSF-DT on {env_name} (device={device})")

    # set seed: prefer dsf_set_seed if available
    if dsf_set_seed is not None:
        try:
            dsf_set_seed(seed, device)
        except TypeError:
            try:
                dsf_set_seed(seed)
            except Exception:
                set_seed_local(seed, device)
    else:
        set_seed_local(seed, device)

    # choose dataset class defensively
    DatasetClass = DSFTrajectoryDataset if DSFTrajectoryDataset is not None else FallbackTrajectoryDataset
    try:
        dataset = DatasetClass(trajectories, max_length_local, gamma_local)
    except TypeError:
        dataset = DatasetClass(trajectories, max_length_local)

    if len(dataset) == 0:
        raise RuntimeError("Empty dataset — cannot train.")

    train_size = int(0.8 * len(dataset))
    val_size = max(1, len(dataset) - train_size)
    if len(dataset) > 1:
        train_d, val_d = random_split(dataset, [train_size, val_size])
    else:
        train_d, val_d = dataset, dataset
    train_loader = DataLoader(train_d, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_d, batch_size=batch_size, shuffle=False)

    state_dim = dataset[0]["states"].shape[-1]

    # build dt_conf
    dt_conf_local = dt_config.copy() if isinstance(dt_config, dict) else dict(dt_config)
    dt_conf_local.update({
        "state_dim": state_dim,
        "act_dim": int(act_dim),
        "max_length": int(max_length_local),
    })
    dt_conf_local.setdefault("n_positions", dt_conf_local.get("max_length", max_length_local))
    dt_conf_local.setdefault("n_ctx", dt_conf_local["n_positions"])
    dt_conf_local.setdefault("warmup_ratio", 0.1)
    dt_conf_local.setdefault("num_training_steps", epochs * max(1, len(train_loader)))

    # instantiate model (try multiple strategies)
    ModelClass = RealDecisionSpikeFormer if RealDecisionSpikeFormer is not None else ToyDecisionSpikeFormer
    model = None
    try:
        safe_conf = dt_conf_local.copy()
        for _k in ("n_positions", "n_ctx", "n_embd", "vocab_size", "bos_token_id"):
            safe_conf.pop(_k, None)
        model = ModelClass(**safe_conf).to(device)
    except Exception:
        try:
            from transformers import GPT2Config
            cfg = GPT2Config(
                vocab_size=1,
                n_ctx = dt_conf_local.get("n_positions", dt_conf_local.get("max_length")),
                n_layer = dt_conf_local.get("n_layer", 2),
                n_head = dt_conf_local.get("n_head", 1),
                n_embd = dt_conf_local.get("hidden_size", dt_conf_local.get("n_embd", 128)),
            )
            setattr(cfg, "warmup_ratio", dt_conf_local.get("warmup_ratio", 0.1))
            setattr(cfg, "num_training_steps", dt_conf_local.get("num_training_steps"))
            for k in ("hidden_size", "n_inner", "n_layer", "n_head"):
                if k in dt_conf_local:
                    setattr(cfg, k, dt_conf_local[k])
            try:
                model = ModelClass(cfg).to(device)
            except Exception:
                model = ModelClass(
                    dt_conf_local.get("hidden_size", 128),
                    dt_conf_local.get("n_layer", 2),
                    dt_conf_local.get("n_head", 1),
                    dt_conf_local.get("n_inner", dt_conf_local.get("hidden_size", 128) * 4),
                    state_dim,
                    int(act_dim),
                    int(dt_conf_local.get("max_length", max_length_local))
                ).to(device)
        except Exception:
            print("[WARN] Using ToyDecisionSpikeFormer fallback.")
            model = ToyDecisionSpikeFormer(state_dim, act_dim, max_length_local).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_val)
    loss_fn = nn.MSELoss()  # Pendulum is continuous

    best_val_loss = float("inf")
    best_path = None
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        total = 0.0
        steps = 0
        for batch in train_loader:
            states = batch["states"].to(device)            # (B, L, state_dim)
            actions = batch["actions"].to(device)          # (B, L, act_dim)
            returns = batch["returns_to_go"].to(device)    # (B, L, 1)
            timesteps = batch["timesteps"].to(device)

            actions_in = actions.float()
            targets = actions.float()

            # IMPORTANT: pass returns in the 3rd positional argument (the model expects it)
            _, preds, _ = model(states, actions_in, returns, returns, timesteps)

            loss = loss_fn(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += float(loss.item())
            steps += 1

        avg_train = total / max(1, steps)

        # validation
        model.eval()
        val_total = 0.0
        vsteps = 0
        with torch.no_grad():
            for batch in val_loader:
                states = batch["states"].to(device)
                actions = batch["actions"].to(device)
                returns = batch["returns_to_go"].to(device)
                timesteps = batch["timesteps"].to(device)

                actions_in = actions.float()
                targets = actions.float()

                # ensure returns passed correctly here too
                _, preds, _ = model(states, actions_in, returns, returns, timesteps)
                vloss = loss_fn(preds, targets)
                val_total += float(vloss.item())
                vsteps += 1

        avg_val = val_total / max(1, vsteps)
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        print(f"[DSF-DT] Epoch {epoch+1}/{epochs} train_loss={avg_train:.4f} val_loss={avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_path = os.path.join(CHECKPOINT_DIR, f"best_DSF-DT_{env_name}.pt")
            save_checkpoint(model, optimizer, best_path)
            print(f"  saved best -> {best_path}")

    if best_path:
        model = load_checkpoint(model, best_path, device=device)

    return model, history

# Evaluate
def evaluate_model(model: nn.Module, env_name: str, n_eval: int = 10, device: str = DEVICE, seed: int = None):
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
        obs = gym_reset(env)
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
            if "cuda" in str(device).lower() and torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                # We pass returns as None here (no returns at inference), but your real model may accept None.
                # If your real DSF expects a returns tensor, consider passing a zeros tensor shaped (1,L,1).
                # We'll attempt to pass a zeros returns tensor (safer).
                returns_dummy = torch.zeros((1, model.max_length, 1), device=device)
                _, action_pred, _ = model(states_t, None, returns_dummy, None, timesteps)
            if "cuda" in str(device).lower() and torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

            if is_continuous:
                act = action_pred.squeeze(0).cpu().numpy()[-1]
                low = env.action_space.low
                high = env.action_space.high
                act = np.clip(act, low, high)
            else:
                logits = action_pred.squeeze(0).cpu().numpy()[-1]
                act = int(np.argmax(logits))

            obs, reward, done, info = gym_step(env, act)
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
        "avg_latency_ms": float(np.mean(latencies) if len(latencies) else 0.0),
        "avg_spikes_per_episode": float(np.mean(spikes) if len(spikes) else 0.0),
        "total_params": sum(p.numel() for p in model.parameters())
    }

# CLI
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
    parser.add_argument("--load-data", type=str, default=None, help="Path to pre-saved dataset")
    args = parser.parse_args()

    device = args.device
    set_seed_local(args.seed, device)

    env_name = args.env
    data_path = args.load_data or os.path.join(DATA_DIR, f"shared_offline_data_{env_name}.pkl")

    if (not args.recollect) and os.path.exists(data_path):
        print(f"Loaded shared dataset -> {data_path}")
        with open(data_path, "rb") as f:
            trajectories = pickle.load(f)
        env = gym.make(env_name)
        is_continuous = isinstance(env.action_space, gym.spaces.Box)
        act_dim = env.action_space.shape[0] if is_continuous else env.action_space.n
        env.close()
    else:
        trajectories, act_dim = collect_shared_dataset(env_name, args.offline_steps, args.max_length)

    dsf_model, dsf_hist = train_dsf(
        trajectories, act_dim, env_name,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_val=args.lr,
        max_length_local=args.max_length,
        gamma_local=gamma,
        seed=args.seed
    )

    print("\n=== Evaluating DSF model ===")
    res = evaluate_model(dsf_model, env_name, n_eval=10, device=device, seed=args.seed)
    print("DSF result:", res)

    out_summary = {
        "env": env_name,
        "dsf_result": res,
        "dsf_history": dsf_hist,
    }
    summary_path = f"dsf_only_summary_{env_name}.pkl"
    with open(summary_path, "wb") as f:
        pickle.dump(out_summary, f)
    print(f"Saved DSF-only summary -> {summary_path}")

if __name__ == "__main__":
    main()

