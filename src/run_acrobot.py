#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_acobot.py — End-to-end benchmarking script with built-in debugging & experimentation plan

What this script gives you (batteries included):
- ✅ Reproducible seeding & numpy 2.0 compatibility shim for classic Gym
- ✅ Shared offline dataset pipeline (random or expert) + dataset stats JSON
- ✅ Optional expert policy training via stable-baselines3 (if available)
- ✅ Three baselines: SNN-DT (toy), Dense-DT (toy), DSF-DT (toy placeholder)
- ✅ Comparable outputs (discrete heads → raw logits for CrossEntropy)
- ✅ Correct spike counting semantics: spikes-per-decision (final timestep only)
- ✅ Training/validation split, early-stop on val loss, best-checkpoint saving
- ✅ Evaluation loop with latencies, spike counts, returns (multi-seed)
- ✅ Action-mapping debug on identical inputs (sanity check comparability)
- ✅ CSV + JSON + PKL summaries; optional Matplotlib plots
- ✅ Clear warnings about energy estimates (for paper-readiness)

Notes:
- Default env is 'Acrobot-v1'. If you meant to name this file run_acrobot.py, simply rename it.
- The model classes here are *toy placeholders*; replace with your real SNN/DSF/DT implementations to produce publishable results.

Usage examples:
  # Quick smoke test (3 epochs, tiny dataset) on CPU
  python run_acobot.py --quick --seeds 42 --device cpu --n_eval 8

  # Longer run with 100 epochs and plots on GPU
  python run_acobot.py --seeds 42 7 123 --device cuda:0 --epochs 100 --n_eval 20 --save_plots

  # Collect & use expert trajectories (requires stable-baselines3)
  python run_acobot.py --collect_expert --expert_train_steps 50000 --env Acrobot-v1 --seeds 42 --device cuda:0 --epochs 50

  # Use a pre-existing dataset path
  python run_acobot.py --dataset data/shared_offline_data_Acrobot-v1.pkl --seeds 42

"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# ===== numpy 2.0 compatibility shim (place BEFORE importing gym) =====
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
# ====================================================================

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

try:
    import gym  # classic gym
except Exception:
    # Optional fallback to gymnasium if needed
    import gymnasium as gym  # type: ignore

# Optional: stable-baselines3 for expert data collection
try:
    import stable_baselines3 as sb3  # type: ignore
    _SB3_OK = True
except Exception:
    sb3 = None
    _SB3_OK = False

# ---------------------------
# Model output normalizer
# ---------------------------
def extract_logits_from_model_output(out):
    """
    Normalize model(...) output to a single logits Tensor.
    Handles:
      - Tensor
      - (Tensor, aux...)
      - [Tensor, ...]
      - (None, Tensor, ...)  -> returns the first Tensor found
      - dict-like outputs where values may include tensors
    Returns:
      logits Tensor, or raises ValueError if no Tensor found.
    """
    if out is None:
        raise ValueError("Model returned None (no logits).")

    # If single tensor
    if isinstance(out, torch.Tensor):
        return out

    # If tuple/list, search for first tensor
    if isinstance(out, (tuple, list)):
        for item in out:
            if isinstance(item, torch.Tensor):
                return item
            # recursively search nested structures
            if isinstance(item, (tuple, list, dict)):
                try:
                    t = extract_logits_from_model_output(item)
                    if isinstance(t, torch.Tensor):
                        return t
                except ValueError:
                    pass
        raise ValueError(f"No Tensor found in model output tuple/list: {out!r}")

    # If dict, search values
    if isinstance(out, dict):
        for v in out.values():
            if isinstance(v, torch.Tensor):
                return v
            if isinstance(v, (tuple, list, dict)):
                try:
                    t = extract_logits_from_model_output(v)
                    if isinstance(t, torch.Tensor):
                        return t
                except ValueError:
                    pass
        raise ValueError(f"No Tensor found in model output dict: {out!r}")

    raise ValueError(f"Unsupported model output type: {type(out)}")


# ---------------------------
# Utilities
# ---------------------------

def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int, device: str = "cpu"):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if "cuda" in str(device).lower() and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------
# Trajectory buffer & dataset
# ---------------------------

class TrajectoryBuffer:
    def __init__(self, obs_dim: int, act_dim: int):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.reset()

    def reset(self):
        self.obs: List[np.ndarray] = []
        self.acts: List[np.ndarray] = []
        self.rews: List[float] = []

    def add(self, obs, act, rew):
        self.obs.append(np.asarray(obs, dtype=np.float32).reshape(-1))
        self.acts.append(np.asarray(act, dtype=np.int64).reshape(-1))
        self.rews.append(float(rew))

    def get_trajectory(self) -> Dict[str, np.ndarray]:
        return {
            "states": np.vstack(self.obs).astype(np.float32),
            "actions": np.vstack(self.acts).astype(np.int64),
            "rewards": np.array(self.rews, dtype=np.float32),
        }


class TrajectoryDataset(Dataset):
    def __init__(self, trajectories: List[Dict], max_length: int, gamma: float = 0.99):
        self.max_length = max_length
        self.gamma = gamma
        self.seqs: List[Dict[str, np.ndarray]] = []
        for traj in trajectories:
            self._add_trajectory(traj)

    def _add_trajectory(self, traj: Dict[str, np.ndarray]):
        states = traj["states"].astype(np.float32)
        actions = traj["actions"].astype(np.int64)
        rewards = traj["rewards"].astype(np.float32)
        rtg = self.compute_returns_to_go(rewards, self.gamma).reshape(-1, 1)
        timesteps = np.arange(len(states)).reshape(-1, 1).astype(np.int64)

        for i in range(1, len(states) + 1):
            start = max(0, i - self.max_length)
            self.seqs.append({
                "states": states[start:i],
                "actions": actions[start:i],
                "returns_to_go": rtg[start:i],
                "timesteps": timesteps[start:i],
            })

    @staticmethod
    def compute_returns_to_go(rewards: np.ndarray, gamma: float = 0.99) -> np.ndarray:
        rtg = np.zeros_like(rewards, dtype=np.float32)
        future = 0.0
        for i in reversed(range(len(rewards))):
            future = rewards[i] + gamma * future
            rtg[i] = future
        return rtg

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.seqs[idx]
        L = s["states"].shape[0]
        pad = self.max_length - L
        Sdim = s["states"].shape[1]
        states = np.vstack([np.zeros((pad, Sdim), dtype=np.float32), s["states"]])
        actions = np.vstack([np.zeros((pad, 1), dtype=np.int64), s["actions"]])  # discrete → [L,1]
        returns = np.vstack([np.zeros((pad, 1), dtype=np.float32), s["returns_to_go"]])
        timesteps = np.vstack([np.zeros((pad, 1), dtype=np.int64), s["timesteps"]]).squeeze(-1)
        return {
            "states": torch.from_numpy(states).float(),
            "actions": torch.from_numpy(actions).long(),
            "returns_to_go": torch.from_numpy(returns).float(),
            "timesteps": torch.from_numpy(timesteps).long(),
            "length": torch.tensor(L, dtype=torch.long),
        }


# ---------------------------
# Model implementations (toy)
# ---------------------------

class _BaseSeqModel(nn.Module):
    def __init__(self, state_dim: int, act_dim: int, max_length: int, hidden_dim: int = 128, n_layers: int = 2):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = int(act_dim)
        self.max_length = max_length
        self.hidden_dim = hidden_dim
        self.state_emb = nn.Linear(state_dim, hidden_dim)
        self.pos_emb = nn.Embedding(max_length, hidden_dim)
        layers: List[nn.Module] = []
        for _ in range(n_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, self.act_dim)  # logits for discrete

    def forward(self, states: torch.Tensor, timesteps: Optional[torch.Tensor] = None, return_spikes: bool = False):
        """Return logits with shape (B, L, act_dim). `return_spikes` ignored in dense models."""
        B, L, _ = states.shape
        if timesteps is None:
            pos_idx = torch.arange(self.max_length, device=states.device).unsqueeze(0).expand(B, -1).long()
        else:
            pos_idx = timesteps
        pos_used = pos_idx[:, -L:]
        h = self.state_emb(states) + self.pos_emb(pos_used)
        h = self.mlp(h)
        logits = self.head(h)
        if return_spikes:
            return logits, 0  # no spikes in dense
        return logits


# ---------------------------
# Replace toy model implementations with these safe versions
# ---------------------------

class DenseDecisionTransformer(nn.Module):
    """
    Dense decision transformer toy: returns (B, L, act_dim) logits tensor.
    Uses local position ids [0..L-1] clamped to [0..max_length-1].
    """
    def __init__(self, state_dim, act_dim, max_length, hidden_dim=128, n_layers=2):
        super().__init__()
        self.state_dim = int(state_dim)
        self.act_dim = int(act_dim)
        self.max_length = int(max_length)
        self.hidden_dim = hidden_dim

        self.state_emb = nn.Linear(self.state_dim, hidden_dim)
        self.pos_emb = nn.Embedding(self.max_length, hidden_dim)

        layers = []
        for _ in range(n_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, self.act_dim)

    def forward(self, states, timesteps=None):
        # states: (B, L, state_dim)
        B, L, _ = states.shape
        # build local pos ids [0..L-1] and clamp to max_length-1
        pos_ids = torch.arange(L, device=states.device).unsqueeze(0).expand(B, L).long()
        pos_ids = torch.clamp(pos_ids, max=self.max_length - 1)

        h = self.state_emb(states) + self.pos_emb(pos_ids)  # (B, L, hidden)
        h = self.mlp(h)
        logits = self.head(h)  # (B, L, act_dim)
        return logits


class DecisionSpikeFormer(nn.Module):
    """
    Dense-style 'DecisionSpikeFormer' toy baseline — structurally similar to Dense DT,
    kept for parity with earlier code. Returns logits tensor (B, L, act_dim).
    """
    def __init__(self, state_dim, act_dim, max_length, hidden_dim=128, n_layers=2):
        super().__init__()
        self.state_dim = int(state_dim)
        self.act_dim = int(act_dim)
        self.max_length = int(max_length)
        self.hidden_dim = hidden_dim

        self.state_emb = nn.Linear(self.state_dim, hidden_dim)
        self.pos_emb = nn.Embedding(self.max_length, hidden_dim)

        blocks = []
        for _ in range(n_layers):
            blocks += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        self.net = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden_dim, self.act_dim)

    def forward(self, states, timesteps=None):
        B, L, _ = states.shape
        pos_ids = torch.arange(L, device=states.device).unsqueeze(0).expand(B, L).long()
        pos_ids = torch.clamp(pos_ids, max=self.max_length - 1)

        h = self.state_emb(states) + self.pos_emb(pos_ids)
        h = self.net(h)
        logits = self.head(h)
        return logits


class SNNDecisionTransformer(nn.Module):
    """
    Toy SNN-like DT. Counts spikes as a diagnostic (get_spike_count),
    but returns only a logits tensor (B, L, act_dim) for training/eval.
    """
    def __init__(self, state_dim, act_dim, max_length, hidden_dim=128, n_layers=2, spike_thresh=0.5):
        super().__init__()
        self.state_dim = int(state_dim)
        self.act_dim = int(act_dim)
        self.max_length = int(max_length)
        self.hidden_dim = hidden_dim
        self.spike_thresh = float(spike_thresh)

        self.state_emb = nn.Linear(self.state_dim, hidden_dim)
        self.pos_emb = nn.Embedding(self.max_length, hidden_dim)

        # simple linear+relu stack to mimic layers; we will count thresholded activations as spikes
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

    def forward(self, states, timesteps=None):
        B, L, _ = states.shape
        pos_ids = torch.arange(L, device=states.device).unsqueeze(0).expand(B, L).long()
        pos_ids = torch.clamp(pos_ids, max=self.max_length - 1)

        h = self.state_emb(states) + self.pos_emb(pos_ids)

        spike_count = 0
        for layer in self.layers:
            h = layer(h)
            # count spikes after ReLU layers (toy heuristic)
            if isinstance(layer, nn.ReLU):
                sp = (h > self.spike_thresh).to(torch.long)
                spike_count += int(sp.sum().item())

        # update spike counter (non-grad)
        with torch.no_grad():
            self._spike_counter += int(spike_count)

        logits = self.head(h)  # (B, L, act_dim)
        return logits




# ---------------------------
# Data collection
# ---------------------------

def _make_env(env_name: str):
    env = gym.make(env_name)
    return env


def collect_random_dataset(env_name: str, steps: int) -> Tuple[List[Dict], int, bool]:
    env = _make_env(env_name)
    is_continuous = isinstance(env.action_space, gym.spaces.Box)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0] if is_continuous else env.action_space.n

    buf = TrajectoryBuffer(obs_dim, 1 if not is_continuous else act_dim)
    trajectories: List[Dict] = []

    steps_done = 0
    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    while steps_done < steps:
        action = env.action_space.sample()
        step_out = env.step(action)
        if isinstance(step_out, tuple) and len(step_out) == 5:
            next_obs, reward, terminated, truncated, _ = step_out
            done = bool(terminated or truncated)
        else:
            next_obs, reward, done, _ = step_out  # type: ignore
        store_act = np.array([int(action)]) if not is_continuous else np.asarray(action, dtype=np.float32)
        buf.add(obs, store_act, float(reward))
        obs = next_obs
        steps_done += 1
        if done:
            trajectories.append(buf.get_trajectory())
            buf.reset()
            reset_out = env.reset()
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    env.close()
    return trajectories, act_dim, is_continuous


def train_and_collect_expert(env_name: str, train_steps: int, rollout_episodes: int = 100,
                              seed: int = 0) -> Tuple[List[Dict], int, bool]:
    if not _SB3_OK:
        raise RuntimeError("stable-baselines3 is not installed; cannot train expert. pip install stable-baselines3[extra]")

    env = _make_env(env_name)
    is_continuous = isinstance(env.action_space, gym.spaces.Box)
    act_dim = env.action_space.shape[0] if is_continuous else env.action_space.n

    algo = None
    if is_continuous:
        algo = sb3.SAC("MlpPolicy", env, seed=seed, verbose=0)
    else:
        # For discrete envs like Acrobot-v1, DQN works
        algo = sb3.DQN("MlpPolicy", env, seed=seed, verbose=0, learning_starts=1000)

    print(f"[expert] Training expert policy for {train_steps} steps (seed={seed})...")
    algo.learn(total_timesteps=int(train_steps))

    # Rollout to collect trajectories
    trajectories: List[Dict] = []
    for ep in range(rollout_episodes):
        obs, _ = env.reset()
        buf = TrajectoryBuffer(env.observation_space.shape[0], 1 if not is_continuous else act_dim)
        done = False
        while not done:
            action, _ = algo.predict(obs, deterministic=True)
            step_out = env.step(action)
            if isinstance(step_out, tuple) and len(step_out) == 5:
                next_obs, reward, terminated, truncated, _ = step_out
                done = bool(terminated or truncated)
            else:
                next_obs, reward, done, _ = step_out  # type: ignore
            store_act = np.array([int(action)]) if not is_continuous else np.asarray(action, dtype=np.float32)
            buf.add(obs, store_act, float(reward))
            obs = next_obs
        trajectories.append(buf.get_trajectory())
    env.close()
    print(f"[expert] Collected {len(trajectories)} expert trajectories.")
    return trajectories, act_dim, is_continuous


# ---------------------------
# Stats, checkpoints, plotting
# ---------------------------

def compute_dataset_stats(trajectories: List[Dict]) -> Dict[str, float]:
    lengths = [len(t["rewards"]) for t in trajectories]
    returns = [float(np.sum(t["rewards"])) for t in trajectories]
    if len(lengths) == 0:
        return {"num_trajs": 0}
    pct = np.percentile(returns, [0, 25, 50, 75, 100]).tolist()
    return {
        "num_trajs": len(trajectories),
        "avg_len": float(np.mean(lengths)),
        "std_len": float(np.std(lengths)),
        "avg_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "return_percentiles": pct,
    }


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, path: Path):
    ensure_dir(path.parent)
    torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict()}, str(path))


def load_checkpoint(model: nn.Module, path: Path, device: str = "cpu") -> nn.Module:
    ckpt = torch.load(str(path), map_location=device)
    model.load_state_dict(ckpt["model_state"])
    return model


# ---------------------------
# Training & Evaluation
# ---------------------------

def train_model(model_name: str, model_class, dataset: TrajectoryDataset, act_dim: int, is_continuous: bool,
                device: str, epochs: int, batch_size: int, lr: float, max_length: int, outdir: Path,
                seed: int, early_stop_patience: int = 20) -> Tuple[nn.Module, Dict[str, List[float]], Path]:
    print(f"[train] Training {model_name}_seed{seed} (device={device}) - dataset examples: {len(dataset)}")

    train_size = int(0.8 * len(dataset))
    val_size = max(1, len(dataset) - train_size)
    if len(dataset) <= 1:
        train_d, val_d = dataset, dataset
    else:
        train_d, val_d = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_d, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_d, batch_size=batch_size, shuffle=False)

    state_dim = dataset[0]["states"].shape[-1]

    # Build model
    if model_class is SNNDecisionTransformer:
        model = SNNDecisionTransformer(state_dim, act_dim, max_length).to(device)
    else:
        model = model_class(state_dim, act_dim, max_length).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss() if is_continuous else nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_path = outdir / f"best_{model_name}_seed{seed}.pt"
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        steps = 0
        for batch in train_loader:
            states = batch["states"].to(device)
            actions = batch["actions"].to(device)
            timesteps = batch["timesteps"].to(device)

            out = model(states, timesteps)
            try:
                logits = extract_logits_from_model_output(out)
            except ValueError as e:
                # helpful debug
                raise RuntimeError(f"Could not extract logits from model output during training: {e}")

            # Ensure logits is a tensor on same device
            logits = logits.to(device)

            if is_continuous:
                targets = actions.float().to(device)
                # logits shape must match targets
                loss = loss_fn(logits, targets)
            else:
                targets = actions.squeeze(-1).long().to(device)
                # Expected logits shape (B, L, act_dim). If model returned (B, act_dim), expand across L.
                if logits.dim() == 2:
                    logits = logits.unsqueeze(1).expand(-1, states.shape[1], -1)
                elif logits.dim() == 1:
                    # unlikely but handle
                    logits = logits.unsqueeze(0).unsqueeze(1).expand(states.shape[0], states.shape[1], -1)
                loss = loss_fn(logits.reshape(-1, int(act_dim)), targets.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            steps += 1

        avg_train = total_loss / max(1, steps)

        # Validation
        model.eval()
        val_loss = 0.0
        vsteps = 0
        with torch.no_grad():
            for batch in val_loader:
                states = batch["states"].to(device)
                actions = batch["actions"].to(device)
                timesteps = batch["timesteps"].to(device)

                out = model(states, timesteps)
                try:
                    logits = extract_logits_from_model_output(out)
                except ValueError as e:
                    raise RuntimeError(f"Could not extract logits from model output during validation: {e}")
                logits = logits.to(device)

                if is_continuous:
                    targets = actions.float().to(device)
                    loss = loss_fn(logits, targets)
                else:
                    targets = actions.squeeze(-1).long().to(device)
                    if logits.dim() == 2:
                        logits = logits.unsqueeze(1).expand(-1, states.shape[1], -1)
                    loss = loss_fn(logits.reshape(-1, int(act_dim)), targets.reshape(-1))
                val_loss += float(loss.item())
                vsteps += 1

        avg_val = val_loss / max(1, vsteps)
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        print(f"[{model_name}_seed{seed}] Epoch {epoch+1}/{epochs} train_loss={avg_train:.4f} val_loss={avg_val:.4f}")

        # Early stopping on best val
        if avg_val + 1e-8 < best_val:
            best_val = avg_val
            save_checkpoint(model, optimizer, best_path)
            print(f"  saved best -> {best_path}")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= early_stop_patience:
                print(f"  early stopping at epoch {epoch+1} (no val improvement for {early_stop_patience} epochs)")
                break

    # Load best
    model = load_checkpoint(model, best_path, device=device)
    return model, history, best_path


@torch.no_grad()
def evaluate_model(model: nn.Module,
                   env_name: str,
                   n_eval: int = 20,
                   device: str = "cpu",
                   seed: int = None,
                   is_snn: bool = False) -> Dict:
    """
    Robust evaluation wrapper.

    Accepts optional `is_snn` for backward compatibility (ignored;
    spike extraction is automatic). Returns the stats dict:
      { model_name, avg_return, std_return, avg_latency_ms, avg_spikes_per_episode, total_params }
    """
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

    for _ in range(n_eval):
        try:
            reset_out = env.reset()
        except TypeError:
            reset_out = env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        done = False
        total_r = 0.0
        obs_hist = []

        # reset model spike counter if present
        if hasattr(model, "reset_spike_count"):
            try:
                model.reset_spike_count()
            except Exception:
                pass

        while not done:
            obs_hist.append(np.asarray(obs).reshape(-1))
            seq = np.array(obs_hist[-getattr(model, "max_length", 20):], dtype=np.float32)
            pad_len = getattr(model, "max_length", 20) - seq.shape[0]
            if pad_len > 0:
                seq = np.vstack([np.zeros((pad_len, seq.shape[1]), dtype=np.float32), seq])
            states_t = torch.from_numpy(seq).unsqueeze(0).to(device)

            # safe timesteps for models expecting them
            L_used = seq.shape[0]
            timesteps = torch.arange(L_used, device=device).unsqueeze(0).long()

            torch.cuda.synchronize() if "cuda" in str(device).lower() and torch.cuda.is_available() else None
            t0 = time.perf_counter()

            with torch.no_grad():
                out = model(states_t, timesteps)

            torch.cuda.synchronize() if "cuda" in str(device).lower() and torch.cuda.is_available() else None
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

            # Extract logits (first Tensor found)
            try:
                logits = extract_logits_from_model_output(out)
            except Exception:
                if isinstance(out, torch.Tensor):
                    logits = out
                else:
                    raise RuntimeError(f"Could not extract logits from model output: {out!r}")

            # Extract spike info (heuristic)
            spike_count_this = 0.0
            if isinstance(out, (tuple, list)):
                for itm in out:
                    if isinstance(itm, (int, float)):
                        spike_count_this = float(itm)
                        break
                    if isinstance(itm, torch.Tensor) and itm.numel() == 1:
                        spike_count_this = float(itm.cpu().item())
                        break
                    if isinstance(itm, torch.Tensor) and itm.numel() > 1:
                        try:
                            spike_count_this = float(itm.sum().cpu().item())
                            break
                        except Exception:
                            pass
            elif isinstance(out, dict):
                for v in out.values():
                    if isinstance(v, (int, float)):
                        spike_count_this = float(v)
                        break
                    if isinstance(v, torch.Tensor) and v.numel() == 1:
                        spike_count_this = float(v.cpu().item())
                        break
                    if isinstance(v, torch.Tensor) and v.numel() > 1:
                        try:
                            spike_count_this = float(v.sum().cpu().item())
                            break
                        except Exception:
                            pass

            # final fallback to model.get_spike_count()
            if spike_count_this == 0.0 and hasattr(model, "get_spike_count"):
                try:
                    spike_count_this = float(model.get_spike_count())
                except Exception:
                    spike_count_this = 0.0

            # Compute action
            if is_continuous:
                act = logits.squeeze(0).cpu().numpy()[-1]
            else:
                last_logits = logits.squeeze(0).cpu().numpy()[-1]
                act = int(np.argmax(last_logits))

            # Step environment (handle gym versions)
            step_out = env.step(act)
            if isinstance(step_out, tuple) and len(step_out) == 5:
                obs, reward, terminated, truncated, _ = step_out
                done = bool(terminated or truncated)
            elif isinstance(step_out, tuple) and len(step_out) == 4:
                obs, reward, done, _ = step_out
                done = bool(done)
            else:
                obs, reward, done, _ = step_out  # type: ignore
                done = bool(done)

            total_r += float(reward)

        returns.append(total_r)
        spikes.append(spike_count_this)

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



def action_mapping_debug(models: Dict[str, nn.Module], env_name: str, device: str, max_length: int):
    """
    Sanity-check: run the same short observation sequence through all models and print:
      - raw last-step logits / predictions
      - final action chosen (argmax for discrete)
      - any scalar auxiliary (e.g., spikes) if returned by the model

    This function is robust to model outputs that are:
      - Tensor
      - (Tensor,) or (None, Tensor, None)
      - (Tensor, aux)
      - dict-like -> will search values for tensors/scalars
    """
    env = _make_env(env_name)
    try:
        reset_out = env.reset()
    except TypeError:
        reset_out = env.reset()  # fallback if signature differs
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    obs_hist: List[np.ndarray] = []
    for _ in range(10):  # step a few times to create a short history
        obs_hist.append(np.asarray(obs, dtype=np.float32).reshape(-1))
        action = env.action_space.sample()
        step_out = env.step(action)
        if isinstance(step_out, tuple) and len(step_out) == 5:
            obs, reward, terminated, truncated, _ = step_out
            done = bool(terminated or truncated)
        else:
            obs, reward, done, _ = step_out  # type: ignore
        if done:
            break
    env.close()

    # Build a sequence padded to max_length (but use actual seq length for pos ids)
    seq = np.array(obs_hist[-max_length:], dtype=np.float32)
    L = seq.shape[0]
    pad = max_length - L
    if pad > 0:
        seq = np.vstack([np.zeros((pad, seq.shape[1]), dtype=np.float32), seq])
    states_t = torch.from_numpy(seq).unsqueeze(0).to(device)  # (1, max_length, state_dim)

    # Build safe timestep / positional ids for model call (shape (B, L_used))
    timesteps = torch.arange(max_length, device=device).unsqueeze(0).long()  # some models expect this
    # But many of our models build positions from length; still pass timesteps for compatibility.

    print("=== ACTION MAPPING DEBUG ===")
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            out = model(states_t, timesteps)
        # Try to extract the logits tensor
        try:
            logits = extract_logits_from_model_output(out)
        except Exception as e:
            print(f"{name} -> ERROR extracting logits: {e}")
            continue

        # Also attempt to extract a scalar auxiliary value (e.g., spikes) for debug printing
        aux_scalar = None
        if isinstance(out, (tuple, list)):
            for item in out:
                if isinstance(item, (int, float)):
                    aux_scalar = float(item)
                    break
                if isinstance(item, torch.Tensor) and item.numel() == 1:
                    aux_scalar = float(item.cpu().item())
                    break
        elif isinstance(out, dict):
            # search values for a scalar-like entry
            for v in out.values():
                if isinstance(v, (int, float)):
                    aux_scalar = float(v)
                    break
                if isinstance(v, torch.Tensor) and v.numel() == 1:
                    aux_scalar = float(v.cpu().item())
                    break

        # Normalise logits to numpy array for printing
        try:
            l_np = logits.squeeze(0).cpu().numpy()  # shape (L_used, act_dim) or (L_used, ) or (act_dim,)
        except Exception as e:
            print(f"{name} -> ERROR converting logits to numpy: {e}")
            continue

        # If logits is 1D (continuous output for final action) treat accordingly
        if l_np.ndim == 1:
            # Could be (act_dim,) or (L_used,) — try interpret as last-step prediction
            last = l_np
        else:
            # Take final timestep's prediction
            last = l_np[-1]

        # Print
        if isinstance(last, np.ndarray):
            # discrete logits or continuous vector
            if last.size == 0:
                print(f"{name} raw last-step pred: (empty array)")
            else:
                print(f"{name} raw last-step logits/pred: {np.array2string(last, precision=6, suppress_small=True)}")
                # If discrete logits, print chosen action
                if last.ndim == 1 and last.size > 1:
                    try:
                        chosen = int(np.argmax(last))
                        print(f"{name} final action (argmax): {chosen}")
                    except Exception:
                        pass
                else:
                    # scalar or single-value continuous output
                    try:
                        print(f"{name} final action/value: {float(np.asscalar(last))}")  # np.asscalar may be deprecated but fallback below
                    except Exception:
                        try:
                            print(f"{name} final action/value: {float(last)}")
                        except Exception:
                            pass
        else:
            print(f"{name} raw last-step pred: {last}")

        if aux_scalar is not None:
            print(f"{name} aux scalar (e.g., spikes): {aux_scalar}")

    print("=============================")



# ---------------------------
# Plotting
# ---------------------------

def _safe_import_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
        return plt
    except Exception:
        return None


def make_plots(all_histories: Dict[str, Dict[str, List[float]]], results: Dict[str, Dict[str, float]], outdir: Path):
    plt = _safe_import_matplotlib()
    if plt is None:
        print("[plot] matplotlib not available; skipping plots.")
        return

    # Learning curves
    for name, hist in all_histories.items():
        if not hist:
            continue
        plt.figure()
        plt.plot(hist.get("train_loss", []), label="train_loss")
        plt.plot(hist.get("val_loss", []), label="val_loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.title(f"Learning curve — {name}")
        fpath = outdir / f"curve_{name}.png"
        plt.savefig(fpath, bbox_inches="tight")
        plt.close()

    # Bar chart of returns
    if results:
        names = list(results.keys())
        vals = [results[k]["avg_return"] for k in names]
        plt.figure()
        plt.bar(names, vals)
        plt.ylabel("avg return")
        plt.title("Average return (higher is better)")
        fpath = outdir / "bar_avg_return.png"
        plt.savefig(fpath, bbox_inches="tight")
        plt.close()


# ---------------------------
# CLI and orchestration
# ---------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="Acrobot-v1")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seeds", type=int, nargs="+", default=[42])

    # Training loop
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max_length", type=int, default=30)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--n_eval", type=int, default=20)

    # Data
    p.add_argument("--dataset", type=str, default=None, help="Path to existing dataset .pkl; if absent, will collect.")
    p.add_argument("--offline_steps", type=int, default=50_000)
    p.add_argument("--recollect", action="store_true", help="Force recollecting offline data")
    p.add_argument("--collect_expert", action="store_true", help="Train an expert with SB3 and collect trajectories")
    p.add_argument("--expert_train_steps", type=int, default=50_000)
    p.add_argument("--expert_rollout_episodes", type=int, default=200)

    # Models to run
    p.add_argument("--run_snn", action="store_true")
    p.add_argument("--run_dense", action="store_true")
    p.add_argument("--run_dsf", action="store_true")

    # Convenience flags
    p.add_argument("--quick", action="store_true", help="Short run: epochs=3, offline_steps=5k, n_eval=8")
    p.add_argument("--save_plots", action="store_true")

    args = p.parse_args()

    # Quick mode overrides
    if args.quick:
        args.epochs = 3
        args.offline_steps = min(args.offline_steps, 5_000)
        args.n_eval = min(args.n_eval, 8)

    # Default: run all three if none selected
    if not (args.run_snn or args.run_dense or args.run_dsf):
        args.run_snn = True
        args.run_dense = True
        args.run_dsf = True

    # Make run dir
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_out = Path("comparisons") / f"{args.env}_compare_{run_stamp}"
    ensure_dir(base_out)

    # Collect or load dataset
    data_path = Path(args.dataset) if args.dataset else Path("data") / f"shared_offline_data_{args.env}.pkl"
    if args.recollect or (not data_path.exists()):
        if args.collect_expert:
            if not _SB3_OK:
                raise RuntimeError("--collect_expert requires stable-baselines3. Install via: pip install stable-baselines3[extra]")
            trajectories, act_dim, is_cont = train_and_collect_expert(
                env_name=args.env,
                train_steps=args.expert_train_steps,
                rollout_episodes=args.expert_rollout_episodes,
                seed=args.seeds[0],
            )
        else:
            trajectories, act_dim, is_cont = collect_random_dataset(args.env, steps=args.offline_steps)
        ensure_dir(data_path.parent)
        with open(data_path, "wb") as f:
            pickle.dump(trajectories, f)
        print(f"[data] Saved dataset -> {data_path}")
    else:
        with open(data_path, "rb") as f:
            trajectories = pickle.load(f)
        env = _make_env(args.env)
        is_cont = isinstance(env.action_space, gym.spaces.Box)
        act_dim = env.action_space.shape[0] if is_cont else env.action_space.n
        env.close()
        print(f"[data] Loaded dataset from {data_path}")

    # Dataset stats
    stats = compute_dataset_stats(trajectories)
    with open(base_out / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("\n=== DATASET STATS ===")
    print(json.dumps(stats, indent=2))
    print("====================\n")

    # Build dataset object
    dataset = TrajectoryDataset(trajectories, max_length=args.max_length, gamma=args.gamma)

    # Seeding & device
    device = args.device

    # Results containers per seed
    seed_summaries: List[Dict] = []

    for seed in args.seeds:
        set_seed(seed, device)
        # Per-seed output dir
        seed_out = base_out / f"seed{seed}"
        ensure_dir(seed_out)

        histories: Dict[str, Dict[str, List[float]]] = {}
        evals: Dict[str, Dict[str, float]] = {}
        best_paths: Dict[str, str] = {}

        # --- Train selected models ---
        models_trained: Dict[str, nn.Module] = {}
        if args.run_snn:
            snn, hist, ckpt = train_model("SNN-DT", SNNDecisionTransformer, dataset, act_dim, is_cont,
                                           device, args.epochs, args.batch_size, args.lr, args.max_length,
                                           seed_out, seed)
            histories["SNN-DT"] = hist
            best_paths["SNN-DT"] = str(ckpt)
            models_trained["SNN-DT"] = snn
        if args.run_dense:
            dense, hist, ckpt = train_model("Dense-DT", DenseDecisionTransformer, dataset, act_dim, is_cont,
                                             device, args.epochs, args.batch_size, args.lr, args.max_length,
                                             seed_out, seed)
            histories["Dense-DT"] = hist
            best_paths["Dense-DT"] = str(ckpt)
            models_trained["Dense-DT"] = dense
        if args.run_dsf:
            dsf, hist, ckpt = train_model("DSF-DT", DecisionSpikeFormer, dataset, act_dim, is_cont,
                                           device, args.epochs, args.batch_size, args.lr, args.max_length,
                                           seed_out, seed)
            histories["DSF-DT"] = hist
            best_paths["DSF-DT"] = str(ckpt)
            models_trained["DSF-DT"] = dsf

        # --- Action mapping debug ---
        try:
            action_mapping_debug(models_trained, args.env, device, args.max_length)
        except Exception as e:
            print(f"[warn] action_mapping_debug failed: {e}")

        # --- Evaluate ---
        for name, model in models_trained.items():
            is_snn = isinstance(model, SNNDecisionTransformer)
            ev = evaluate_model(model, args.env, n_eval=args.n_eval, device=device, seed=seed, is_snn=is_snn)
            evals[name] = ev

        # Save histories CSV per model
        for name, hist in histories.items():
            if not hist:
                continue
            csv_path = seed_out / f"history_{name}.csv"
            with open(csv_path, "w") as f:
                f.write("epoch,train_loss,val_loss\n")
                for i, (tr, vl) in enumerate(zip(hist.get("train_loss", []), hist.get("val_loss", []))):
                    f.write(f"{i+1},{tr},{vl}\n")

        # Make plots if requested
        if args.save_plots:
            make_plots(histories, evals, seed_out)

        # Per-seed summary
        seed_summary = {
            "seed": seed,
            "env": args.env,
            "dataset_path": str(data_path),
            "dataset_stats": stats,
            "evals": evals,
            "checkpoints": best_paths,
            "hyperparams": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "max_length": args.max_length,
                "gamma": args.gamma,
                "n_eval": args.n_eval,
                "offline_steps": args.offline_steps,
            },
            "energy_note": (
                "Spike counts here are toy proxies (ReLU>thresh activations at final timestep). "
                "Do NOT claim hardware energy savings from these numbers without proper neuromorphic simulation/measurement."
            ),
        }
        with open(seed_out / f"summary_seed{seed}.json", "w") as f:
            json.dump(seed_summary, f, indent=2)
        with open(seed_out / f"summary_seed{seed}.pkl", "wb") as f:
            pickle.dump(seed_summary, f)

        seed_summaries.append(seed_summary)

    # Aggregate across seeds
    agg = {"env": args.env, "seeds": args.seeds, "runs": seed_summaries}
    with open(base_out / "comparison_summary_all_seeds.json", "w") as f:
        json.dump(agg, f, indent=2)
    with open(base_out / "comparison_summary_all_seeds.pkl", "wb") as f:
        pickle.dump(agg, f)

    print(f"\n[main] All seeds completed. Check outputs in: {base_out}\n")
    # Brief stdout summary
    for run in seed_summaries:
        print(f"=== Seed {run['seed']} results ===")
        for name, ev in run["evals"].items():
            print(f"{name}: return {ev['avg_return']:.1f} ± {ev['std_return']:.1f} | avg latency {ev['avg_latency_ms']:.3f} ms | spikes/ep {ev['avg_spikes_per_episode']:.1f}")
        print("===========================\n")


if __name__ == "__main__":
    main()

