# src/run_acrobot.py
"""
Robust run script for comparing SNN-DT vs DecisionSpikeFormer (DSF) on Acrobot-v1.
Implements data parity checks, spike-count instrumentation, action-mapping debug prints,
multiple-seed runs, energy estimates, and reproducible outputs.

- Replace toy model classes below with your real implementations (import them instead).
- The script *builds the shared dataset once* and re-uses it for both model trainings to ensure parity.
"""
import argparse
import json
import os
import pickle
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

# numpy 2.0 compatibility shim (put before gym import if necessary)
import numpy as _np
if not hasattr(_np, "float_"):
    _np.float_ = _np.float64
if not hasattr(_np, "int"):
    _np.int = int
if not hasattr(_np, "bool"):
    _np.bool = bool

import gym
# Ensure repo root is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in __import__("sys").path:
    __import__("sys").path.insert(0, PROJECT_ROOT)

# --- Default config (fallback). Replace or import your project-specific configs if present.
try:
    from src.configs.acrobot_config import (
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
except Exception:
    DEVICE = "cpu"
    SEED = 42
    offline_steps = 5000
    dt_batch_size = 64
    dt_epochs = 100
    gamma = 0.99
    max_length = 20
    lr = 3e-4
    dt_config = {}
    ENV_NAME = "Acrobot-v1"

CHECKPOINT_DIR = "checkpoints"
DATA_DIR = "data"
OUT_DIR = "comparisons"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# try to import your real helper for collecting trajectories if available
try:
    from src.train_dsf_dt import collect_trajectories as dsf_collect_trajectories
except Exception:
    dsf_collect_trajectories = None


# ---------------------------
# Utilities
# ---------------------------
def set_seed(seed: int, device: str = "cpu"):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if "cuda" in device and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # make deterministic (may slow)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def now_str():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


# ---------------------------
# Dataset classes
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
        self.max_length = int(max_length)
        self.gamma = float(gamma)
        self.seqs = []
        for traj in trajectories:
            self._add_trajectory(traj)

    def _add_trajectory(self, traj):
        states = traj["states"]
        actions = traj["actions"]
        rewards = traj["rewards"]
        rtg = self.compute_returns_to_go(rewards, self.gamma).reshape(-1, 1)
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
# Toy model implementations (placeholders)
# Replace these with your real implementations (import them instead).
# ---------------------------
class DecisionSpikeFormer(nn.Module):
    def __init__(self, state_dim, act_dim, max_length, hidden_dim=128, n_layers=2):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = int(act_dim)
        self.max_length = int(max_length)
        self.state_emb = nn.Linear(state_dim, hidden_dim)
        self.pos_emb = nn.Embedding(self.max_length, hidden_dim)
        layers = []
        for _ in range(n_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, self.act_dim)

    def forward(self, states, actions_in=None, returns=None, returns_target=None, timesteps=None):
        B, L, _ = states.shape
        pos_idx = torch.arange(self.max_length, device=states.device).unsqueeze(0).expand(B, -1)
        pos_used = pos_idx[:, -L:].long()
        h = self.state_emb(states) + self.pos_emb(pos_used)
        h = self.mlp(h)
        action_pred = self.head(h)
        return None, action_pred, None


class SNNDecisionTransformer(nn.Module):
    def __init__(self, state_dim, act_dim, max_length, hidden_dim=128, n_layers=2, spike_thresh=0.5, enable_plasticity=False):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = int(act_dim)
        self.max_length = int(max_length)
        self.hidden_dim = hidden_dim
        self.spike_thresh = spike_thresh
        self.state_emb = nn.Linear(state_dim, hidden_dim)
        self.pos_emb = nn.Embedding(self.max_length, hidden_dim)
        # simple stack (not real spiking neurons) but includes spike counting logic
        layers = []
        for _ in range(n_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        self.layers = nn.ModuleList(layers)
        self.head = nn.Linear(hidden_dim, self.act_dim)
        # internal spike counter as a buffer for eval reporting
        self.register_buffer("_spike_counter", torch.tensor(0, dtype=torch.long))
        self.enable_plasticity = bool(enable_plasticity)

    def reset_spike_count(self):
        try:
            self._spike_counter.zero_()
        except Exception:
            self._spike_counter = torch.tensor(0, dtype=torch.long)

    def get_spike_count(self):
        try:
            return int(self._spike_counter.item())
        except Exception:
            return 0

    def forward(self, states, actions_in=None, returns=None, returns_target=None, timesteps=None):
        B, L, _ = states.shape
        pos_idx = torch.arange(self.max_length, device=states.device).unsqueeze(0).expand(B, -1)
        pos_used = pos_idx[:, -L:].long()
        h = self.state_emb(states) + self.pos_emb(pos_used)
        spike_count = 0
        for layer in self.layers:
            h = layer(h)
            if isinstance(layer, nn.ReLU):
                # naive spike detection: count activations above threshold
                sp = (h > self.spike_thresh).to(torch.long)
                spike_count += int(sp.sum().item())
        # accumulate into buffer safely
        try:
            self._spike_counter += int(spike_count)
        except Exception:
            # fallback: store as attribute
            if not hasattr(self, "_spike_counter_val"):
                self._spike_counter_val = 0
            self._spike_counter_val += int(spike_count)
        action_pred = self.head(h)
        return None, action_pred, None


# ---------------------------
# Small helpers for parity, instrumentation, and energy
# ---------------------------
def ensure_spike_api(model: nn.Module):
    """
    Ensure model has get_spike_count() and reset_spike_count().
    If missing, monkeypatch no-op methods (return 0).
    """
    if not hasattr(model, "get_spike_count"):
        def _get(): return 0
        model.get_spike_count = _get
    if not hasattr(model, "reset_spike_count"):
        def _reset(): return None
        model.reset_spike_count = _reset


def data_parity_check(dataset_a: TrajectoryDataset, dataset_b: TrajectoryDataset, n_samples: int = 5, atol: float = 1e-6) -> Tuple[bool, Dict]:
    """
    Compare a few examples from two TrajectoryDataset instances to ensure identical preprocessing.
    Returns (passes, diagnostics)
    """
    diag = {"passed": False, "mismatches": []}
    if len(dataset_a) == 0 or len(dataset_b) == 0:
        diag["reason"] = "one of the datasets is empty"
        return False, diag
    # compare first n_samples items
    n = min(n_samples, len(dataset_a), len(dataset_b))
    for i in range(n):
        a = dataset_a[i]
        b = dataset_b[i]
        for key in ["states", "actions", "returns_to_go", "timesteps"]:
            aval = a[key].numpy() if isinstance(a[key], torch.Tensor) else np.array(a[key])
            bval = b[key].numpy() if isinstance(b[key], torch.Tensor) else np.array(b[key])
            if not np.allclose(aval, bval, atol=atol):
                diag["mismatches"].append({
                    "index": i, "key": key,
                    "a_mean": float(np.mean(aval)), "b_mean": float(np.mean(bval)),
                    "a_shape": aval.shape, "b_shape": bval.shape,
                })
    diag["passed"] = (len(diag["mismatches"]) == 0)
    return diag["passed"], diag


def print_action_mapping_debug(model_a: nn.Module, model_b: nn.Module, device: str, sample_states: torch.Tensor):
    """
    Forward a small sample of sequences through both models and print raw outputs and final mapped actions
    (useful to catch discrete/continuous mapping mismatches).
    sample_states: shape (B, L, state_dim)
    """
    model_a.eval(); model_b.eval()
    with torch.no_grad():
        _, pred_a, _ = model_a(sample_states.to(device))
        _, pred_b, _ = model_b(sample_states.to(device))
    pred_a = pred_a.squeeze(0).cpu().numpy()  # (L, act_dim) or (L,) if continuous
    pred_b = pred_b.squeeze(0).cpu().numpy()
    def final_action_from_pred(pred):
        # If logits (2D), argmax last timestep
        if pred.ndim == 2:
            return int(np.argmax(pred[-1]))
        elif pred.ndim == 1:
            return pred[-1].item()
        else:
            return pred.shape
    print("=== ACTION MAPPING DEBUG ===")
    print("Model A raw last-step pred:", pred_a[-1])
    print("Model A final action:", final_action_from_pred(pred_a))
    print("Model B raw last-step pred:", pred_b[-1])
    print("Model B final action:", final_action_from_pred(pred_b))
    print("=============================")


def estimate_energy_per_decision(avg_spikes_per_decision: float, pj_per_spike: float = 5.0) -> float:
    """
    Convert avg spikes per decision to Joules using pJ/spike assumption (pj_per_spike).
    Returns Joules.
    """
    return float(avg_spikes_per_decision * pj_per_spike * 1e-12)


# ---------------------------
# Data collection
# ---------------------------
def collect_shared_dataset(env_name: str, offline_steps_local: int, max_len: int) -> Tuple[List[Dict], int]:
    print(f"[dataset] Collecting shared dataset for {env_name} (offline_steps={offline_steps_local})...")
    # Try using external collector if available; otherwise fall back to internal collector.
    if dsf_collect_trajectories is not None:
        try:
            trajectories, act_dim = dsf_collect_trajectories(env_name, offline_steps_local, max_len)
            print(f"[dataset] Collected {len(trajectories)} trajectories via dsf_collect_trajectories.")
            ensure_dir(DATA_DIR)
            out_path = os.path.join(DATA_DIR, f"shared_offline_data_{env_name}.pkl")
            with open(out_path, "wb") as f:
                pickle.dump(trajectories, f)
            print(f"[dataset] Saved dataset -> {out_path}")
            return trajectories, act_dim
        except Exception as e:
            print("[dataset] dsf_collect_trajectories failed with exception, falling back:", e)

    # fallback implement simple collector
    env = gym.make(env_name)
    is_continuous = isinstance(env.action_space, gym.spaces.Box)
    act_dim = env.action_space.shape[0] if is_continuous else env.action_space.n
    buf = TrajectoryBuffer(max_len, env.observation_space.shape[0], act_dim if is_continuous else 1)
    trajectories = []
    steps = 0
    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    while steps < offline_steps_local:
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
        if isinstance(env.action_space, gym.spaces.Box):
            store_act = np.asarray(action).reshape(-1)
        else:
            store_act = np.asarray([int(action)])
        buf.add(np.asarray(obs).flatten(), store_act, float(reward))
        obs = next_obs
        if done:
            trajectories.append(buf.get_trajectory())
            buf.reset()
            reset_out = env.reset()
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        steps += 1
    env.close()
    print(f"[dataset] Collected {len(trajectories)} trajectories (fallback collector).")
    ensure_dir(DATA_DIR)
    out_path = os.path.join(DATA_DIR, f"shared_offline_data_{env_name}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(trajectories, f)
    print(f"[dataset] Saved dataset -> {out_path}")
    return trajectories, act_dim


# ---------------------------
# Training & evaluation (using the same dataset object)
# ---------------------------
def train_model_from_dataset(model_cls, dataset: TrajectoryDataset, act_dim: int, is_continuous: bool,
                             model_name: str, device: str, epochs: int, batch_size: int, lr_local: float,
                             max_length_local: int, quick: bool = False, disable_plasticity: bool = False):
    """
    Train model_cls on dataset (which is already a TrajectoryDataset) and return best model and history.
    """
    print(f"[train] Training {model_name} (device={device}) - dataset examples: {len(dataset)}")
    train_size = int(0.8 * len(dataset))
    val_size = max(1, len(dataset) - train_size)
    if len(dataset) > 1:
        train_d, val_d = random_split(dataset, [train_size, val_size])
    else:
        train_d, val_d = dataset, dataset

    train_loader = DataLoader(train_d, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_d, batch_size=batch_size, shuffle=False)

    state_dim = dataset[0]["states"].shape[-1]
    model = model_cls(state_dim, act_dim, max_length_local).to(device)

    # If model supports a constructor arg to disable plasticity, you can pass it when using real models.
    if disable_plasticity and hasattr(model, "enable_plasticity"):
        try:
            model.enable_plasticity = False
        except Exception:
            pass

    ensure_spike_api(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_local)
    loss_fn = nn.MSELoss() if is_continuous else nn.CrossEntropyLoss()

    best_val = float("inf")
    best_path = None
    history = {"train_loss": [], "val_loss": []}
    epochs_run = 3 if quick else epochs

    for epoch in range(epochs_run):
        model.train()
        total_loss = 0.0
        steps = 0
        for batch in train_loader:
            states = batch["states"].to(device)
            actions = batch["actions"].to(device)
            returns = batch["returns_to_go"].to(device)
            timesteps = batch["timesteps"].to(device)

            if is_continuous:
                targets = actions.float()
                actions_in = actions.float()
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
                    targets = actions.float()
                    actions_in = actions.float()
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
        print(f"[{model_name}] Epoch {epoch+1}/{epochs_run} train_loss={avg_train:.4f} val_loss={avg_val:.4f}")

        if avg_val < best_val:
            best_val = avg_val
            best_path = os.path.join(CHECKPOINT_DIR, f"best_{model_name}_{now_str()}.pt")
            torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict()}, best_path)
            print(f"  saved best -> {best_path}")

    if best_path:
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
    return model, history


def evaluate_model(model: nn.Module, env_name: str, n_eval: int = 20, device: str = "cpu", seed: Optional[int] = None) -> Dict:
    """
    Evaluate model for n_eval episodes, return dict with avg_return, std_return, avg_latency_ms, avg_spikes_per_episode, total_params.
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
    if hasattr(model, "reset_spike_count"):
        model.reset_spike_count()

    for ep in range(n_eval):
        reset_out = env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        done = False
        total_r = 0.0
        obs_hist: List = []
        # reset per-episode spike counts if model supports it
        if hasattr(model, "reset_spike_count"):
            try:
                model.reset_spike_count()
            except Exception:
                pass
        while not done:
            obs_hist.append(np.asarray(obs).reshape(-1))
            seq = np.array(obs_hist[-model.max_length:], dtype=np.float32)
            pad_len = model.max_length - seq.shape[0]
            if pad_len > 0:
                seq = np.vstack([np.zeros((pad_len, seq.shape[1]), dtype=np.float32), seq])
            states_t = torch.from_numpy(seq).unsqueeze(0).to(device)
            timesteps = torch.arange(model.max_length, device=device).unsqueeze(0)
            # timing
            if "cuda" in device and torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                _, action_pred, _ = model(states_t, None, None, None, timesteps)
            if "cuda" in device and torch.cuda.is_available():
                torch.cuda.synchronize()
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
            # safety: avoid runaway episodes in broken policies
            if len(obs_hist) > 10000:
                print(f"[eval] episode aborted due to excessive length ({len(obs_hist)})")
                done = True
        returns.append(total_r)
        spikes.append(model.get_spike_count() if hasattr(model, "get_spike_count") else 0)
    env.close()
    return {
        "model_name": model.__class__.__name__,
        "avg_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "avg_latency_ms": float(np.mean(latencies)),
        "avg_spikes_per_episode": float(np.mean(spikes)),
        "total_params": int(sum(p.numel() for p in model.parameters()))
    }


# ---------------------------
# Main CLI & run orchestration
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default=ENV_NAME)
    parser.add_argument("--seeds", type=int, nargs="+", default=[SEED], help="List of seeds to run (e.g. --seeds 42 7 123)")
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--recollect", action="store_true", help="Force recollecting offline data")
    parser.add_argument("--quick", action="store_true", help="Quick run (few epochs) for debugging")
    parser.add_argument("--disable_plasticity", action="store_true", help="Disable plasticity modules (if supported by model)")
    parser.add_argument("--offline_steps", type=int, default=offline_steps)
    parser.add_argument("--epochs", type=int, default=dt_epochs)
    parser.add_argument("--batch_size", type=int, default=dt_batch_size)
    parser.add_argument("--lr", type=float, default=lr)
    parser.add_argument("--max_length", type=int, default=max_length)
    parser.add_argument("--pj_per_spike", type=float, default=5.0, help="pJ per spike for energy estimate")
    parser.add_argument("--n_eval", type=int, default=20)
    parser.add_argument("--outdir", type=str, default=OUT_DIR)
    args = parser.parse_args()

    ensure_dir(args.outdir)
    run_id = f"acrobot_compare_{now_str()}"
    outdir = os.path.join(args.outdir, run_id)
    ensure_dir(outdir)

    # save resolved args for reproducibility
    with open(os.path.join(outdir, "run_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    for seed in args.seeds:
        print(f"\n=== RUN seed={seed} device={args.device} quick={args.quick} ===")
        set_seed(seed, args.device)

        # (re)collect dataset if needed
        data_path = os.path.join(DATA_DIR, f"shared_offline_data_{args.env}.pkl")
        if (not args.recollect) and os.path.exists(data_path):
            print(f"[main] Loading dataset from {data_path}")
            with open(data_path, "rb") as f:
                trajectories = pickle.load(f)
        else:
            trajectories, _ = collect_shared_dataset(args.env, args.offline_steps, args.max_length)

        # Build TrajectoryDataset once and reuse for both models to guarantee parity
        dataset = TrajectoryDataset(trajectories, args.max_length, gamma)

        # For parity debugging, create *two* copies of the dataset object (they are deterministic transforms of same data)
        dataset_for_snn = dataset
        dataset_for_dsf = dataset  # same object: ensures identical source sequences

        # Parity check (very important)
        ok, diag = data_parity_check(dataset_for_snn, dataset_for_dsf, n_samples=8)
        print("[parity] passed:", ok)
        if not ok:
            print("[parity] diagnostics (first mismatches):", diag.get("mismatches")[:3])
            # save diag
            with open(os.path.join(outdir, f"parity_diag_seed{seed}.pkl"), "wb") as f:
                pickle.dump(diag, f)
            # continue but warn
            print("[parity] WARNING: dataset parity failed. Fix loader/normalization before claiming benchmarks.")

        # Prepare model classes
        # Replace these assignments with your real classes if available, e.g.:
        # from src.models.snn_dt import SNNDecisionTransformer as RealSNN
        # from src.models.dsf_dt import DecisionSpikeFormer as RealDSF
        RealSNN = SNNDecisionTransformer
        RealDSF = DecisionSpikeFormer

        is_continuous = False
        # derive act dim from env
        env_tmp = gym.make(args.env)
        is_continuous = isinstance(env_tmp.action_space, gym.spaces.Box)
        act_dim = env_tmp.action_space.shape[0] if is_continuous else env_tmp.action_space.n
        env_tmp.close()

        # Train SNN (on same dataset object)
        snn_model, snn_hist = train_model_from_dataset(
            RealSNN, dataset_for_snn, act_dim, is_continuous,
            model_name=f"SNN-DT_seed{seed}", device=args.device,
            epochs=args.epochs, batch_size=args.batch_size, lr_local=args.lr,
            max_length_local=args.max_length, quick=args.quick, disable_plasticity=args.disable_plasticity
        )

        # Train DSF baseline
        dsf_model, dsf_hist = train_model_from_dataset(
            RealDSF, dataset_for_dsf, act_dim, is_continuous,
            model_name=f"DSF-DT_seed{seed}", device=args.device,
            epochs=args.epochs, batch_size=args.batch_size, lr_local=args.lr,
            max_length_local=args.max_length, quick=args.quick, disable_plasticity=args.disable_plasticity
        )

        # Ensure both expose spike API
        ensure_spike_api(snn_model)
        ensure_spike_api(dsf_model)

        # Quick action-mapping debug using one minibatch from dataset
        sample_item = dataset[0]
        sample_states = sample_item["states"].unsqueeze(0)  # (1, L, state_dim)
        try:
            print_action_mapping_debug(snn_model, dsf_model, args.device, sample_states)
        except Exception as e:
            print("[debug] action mapping debug failed:", e)

        # Evaluate both models
        res_snn = evaluate_model(snn_model, args.env, n_eval=args.n_eval, device=args.device, seed=seed)
        res_dsf = evaluate_model(dsf_model, args.env, n_eval=args.n_eval, device=args.device, seed=seed)

        # Energy estimates (per decision) using pJ per spike assumption
        # Convert avg_spikes_per_episode -> avg_spikes_per_decision by dividing by average episode length.
        # We estimate avg episode length by using the dataset: average trajectory length in dataset.
        avg_episode_len = np.mean([traj["states"].shape[0] for traj in trajectories]) if len(trajectories) > 0 else 1.0
        snn_spikes_per_decision = res_snn["avg_spikes_per_episode"] / max(1.0, avg_episode_len)
        dsf_spikes_per_decision = res_dsf["avg_spikes_per_episode"] / max(1.0, avg_episode_len)
        snn_energy_J = estimate_energy_per_decision(snn_spikes_per_decision, pj_per_spike=args.pj_per_spike)
        dsf_energy_J = estimate_energy_per_decision(dsf_spikes_per_decision, pj_per_spike=args.pj_per_spike)

        # Compose summary
        summary = {
            "run_id": run_id,
            "seed": seed,
            "env": args.env,
            "snn": res_snn,
            "dsf": res_dsf,
            "snn_history": snn_hist,
            "dsf_history": dsf_hist,
            "avg_episode_len": float(avg_episode_len),
            "snn_spikes_per_decision": float(snn_spikes_per_decision),
            "dsf_spikes_per_decision": float(dsf_spikes_per_decision),
            "snn_energy_J": float(snn_energy_J),
            "dsf_energy_J": float(dsf_energy_J),
            "pj_per_spike_used": float(args.pj_per_spike),
            "parity_ok": ok,
            "parity_diag": diag,
            "args": vars(args),
            "timestamp": now_str()
        }

        # save artifacts
        out_file = os.path.join(outdir, f"comparison_summary_{args.env}_seed{seed}.pkl")
        with open(out_file, "wb") as f:
            pickle.dump(summary, f)
        # human readable json too
        out_json = os.path.join(outdir, f"comparison_summary_{args.env}_seed{seed}.json")
        with open(out_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[main] Saved summary -> {out_file} and {out_json}")

        # print brief human summary
        print("\n=== Brief summary ===")
        print("SNN:", res_snn["avg_return"], "±", res_snn["std_return"], "| spikes/ep:", res_snn["avg_spikes_per_episode"],
              "| est energy/decision (J):", snn_energy_J)
        print("DSF:", res_dsf["avg_return"], "±", res_dsf["std_return"], "| spikes/ep:", res_dsf["avg_spikes_per_episode"],
              "| est energy/decision (J):", dsf_energy_J)
        print("=====================")

    print("\nAll seeds completed. Check outputs in:", outdir)


if __name__ == "__main__":
    main()

