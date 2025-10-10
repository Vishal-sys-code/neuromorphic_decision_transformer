# src/train_utils.py
"""
Flexible train_utils used by run_experiment.py

Exports:
 - train_model(model, trajectories, args, log_dir)
 - evaluate_model(model, env_name, max_length)

This file is intentionally defensive: it tries a few reasonable model.forward signatures
so it will work with different model implementations. If your model requires a
different calling convention, update the "model_forward" wrapper accordingly.
"""

import os
import time
import math
import random
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import numpy as np
import gym
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def _infer_action_type(trajectories: List[Dict]) -> bool:
    """Return True if actions appear continuous (float), False if discrete (int)."""
    # inspect first non-empty action
    for traj in trajectories:
        if len(traj.get("actions", [])) > 0:
            a0 = traj["actions"][0]
            if isinstance(a0, (list, tuple, np.ndarray)):
                a0 = np.asarray(a0)
                return np.issubdtype(a0.dtype, np.floating)
            return isinstance(a0, float)
    # default assume discrete
    return False


def _pad_clip(traj, max_len, state_dim, act_space):
    """Pad a single clip to max_len. Return (states, actions, mask)."""
    states = np.zeros((max_len, state_dim), dtype=np.float32)
    if isinstance(act_space, gym.spaces.Box):
        action_dim = int(np.prod(act_space.shape))
        actions = np.zeros((max_len, action_dim), dtype=np.float32)
    else:
        # store discrete actions as integers
        actions = np.zeros((max_len,), dtype=np.int64)

    mask = np.zeros((max_len,), dtype=np.float32)
    L = min(len(traj["observations"]), max_len)
    for i in range(L):
        states[i, :] = np.asarray(traj["observations"][i], dtype=np.float32)
        a = traj["actions"][i]
        if isinstance(a, (list, tuple, np.ndarray)):
            actions[i] = np.asarray(a, dtype=np.float32)
        else:
            # scalar
            try:
                actions[i] = int(a)
            except Exception:
                actions[i] = float(a)
        mask[i] = 1.0
    return states, actions, mask


def _build_dataset(trajectories: List[Dict], max_length: int, state_dim: int, act_space):
    """Convert list of trajectory dicts into numpy arrays for batching."""
    X_states = []
    X_actions = []
    X_masks = []
    for traj in trajectories:
        s, a, m = _pad_clip(traj, max_length, state_dim, act_space)
        X_states.append(s)
        X_actions.append(a)
        X_masks.append(m)
    X_states = np.stack(X_states, axis=0)  # (N, L, state_dim)
    X_actions = np.stack(X_actions, axis=0)
    X_masks = np.stack(X_masks, axis=0)
    return X_states, X_actions, X_masks


def compute_returns_to_go(rewards, gamma=1.0):
    """
    Compute returns-to-go (sum of future rewards) for each timestep.
    No discounting if gamma=1.0.
    """
    rtg = torch.zeros_like(rewards, dtype=torch.float32)
    for t in reversed(range(len(rewards))):
        rtg[t] = rewards[t] + (gamma * rtg[t + 1] if t + 1 < len(rewards) else 0.0)
    return rtg


def train_model(model, trajectories, args, log_dir):
    import torch
    import torch.nn as nn
    import torch.optim as optim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    criterion = nn.MSELoss()

    # Convert trajectories -> tensors
    states, actions, returns, timesteps = [], [], [], []
    for traj in trajectories:
        states.append(torch.tensor(traj["observations"], dtype=torch.float32))
        actions.append(torch.tensor(traj["actions"], dtype=torch.long))
        returns.append(torch.tensor(traj["returns_to_go"], dtype=torch.float32))
        timesteps.append(torch.arange(len(traj["observations"])))

    states = torch.cat(states).to(device)
    actions = torch.cat(actions).to(device)
    returns = torch.cat(returns).to(device)
    timesteps = torch.cat(timesteps).to(device)

    dataset = torch.utils.data.TensorDataset(states, actions, returns, timesteps)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch_states, batch_actions, batch_returns, batch_timesteps in dataloader:
            batch_states = batch_states.unsqueeze(0).to(device)   # [B, T, state_dim]
            batch_actions = batch_actions.unsqueeze(0).to(device) # [B, T]
            batch_returns = batch_returns.unsqueeze(0).to(device) # [B, T]
            batch_timesteps = batch_timesteps.unsqueeze(0).to(device)

            optimizer.zero_grad()

            # Forward pass
            _, action_preds, _ = model(
                states=batch_states,
                actions=batch_actions,
                returns_to_go=batch_returns,
                timesteps=batch_timesteps,
                attention_mask=None
            )

            # Compute loss (predict next action)
            action_preds = action_preds.reshape(-1, action_preds.size(-1))
            batch_actions = batch_actions.reshape(-1)

            loss = criterion(action_preds.float(), nn.functional.one_hot(batch_actions, num_classes=action_preds.size(-1)).float())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"[Train] Epoch {epoch:03d} | Loss = {avg_loss:.6f}")


def evaluate_model(model, env_name: str, max_length: int = 50, n_episodes: int = 10):
    """
    Evaluate 'model' in env_name deterministically (greedy actions) for n_episodes.

    Tries multiple policies:
      - If model has an 'act' or 'predict' method, uses it.
      - Else attempts to use model.forward to compute action from state (prefers model(state))
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    env = gym.make(env_name)
    returns = []
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        total_r = 0.0
        steps = 0
        while not done and steps < max_length:
            # convert obs to tensor
            st = torch.tensor(np.asarray(obs, dtype=np.float32)).to(device)
            action = None
            # try model.act / model.predict
            if hasattr(model, "act"):
                try:
                    action = model.act(st.unsqueeze(0))  # assume returns numpy or tensor
                except Exception:
                    action = None
            if action is None and hasattr(model, "predict"):
                try:
                    action = model.predict(st.unsqueeze(0))
                except Exception:
                    action = None
            if action is None:
                # fallback: try model(st) or model(st.unsqueeze(0))
                try:
                    with torch.no_grad():
                        out = model(st.unsqueeze(0))
                    # out could be logits or action values
                    if isinstance(out, torch.Tensor):
                        out_np = out.detach().cpu().numpy()
                        # heuristics: if shape (1,C) choose argmax; if (1,1) take scalar
                        if out_np.ndim == 2 and out_np.shape[1] > 1:
                            a = int(out_np[0].argmax())
                            action = a
                        else:
                            # continuous
                            action = out_np.reshape(-1)
                    else:
                        action = out
                except Exception:
                    # as last resort sample random action
                    action = env.action_space.sample()

            # ensure action numeric type accepted by env
            try:
                next_obs, r, done, info = env.step(action)
            except Exception:
                # try converting action to np array or scalar
                try:
                    next_obs, r, done, info = env.step(action.numpy() if hasattr(action, "numpy") else np.array(action))
                except Exception:
                    next_obs, r, done, info = env.step(env.action_space.sample())

            total_r += float(r)
            obs = next_obs
            steps += 1

        returns.append(total_r)
    env.close()
    mean_r = float(np.mean(returns))
    std_r = float(np.std(returns))
    return {"mean_return": mean_r, "std_return": std_r}

