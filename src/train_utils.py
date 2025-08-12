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


def _compute_return_to_go(rewards_array: np.ndarray, mask: np.ndarray):
    # rewards_array shape (B, L)
    B, L = rewards_array.shape
    rtg = np.zeros_like(rewards_array, dtype=np.float32)
    for b in range(B):
        cum = 0.0
        for t in reversed(range(L)):
            if mask[b, t] > 0:
                cum = cum + rewards_array[b, t]
                rtg[b, t] = cum
            else:
                rtg[b, t] = 0.0
        # optional normalization could be added here
    return rtg


def train_model(model, trajectories: List[Dict], args, log_dir: str):
    """
    Train 'model' on the given trajectories.

    Args:
      model: torch.nn.Module
      trajectories: list of dicts with keys 'observations','actions','rewards','dones'
      args: namespace with hyperparams (batch_size, max_iters, learning_rate, max_length, etc.)
      log_dir: directory to write checkpoints and logs
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    # Infer dims
    # assume trajectories not empty
    state_dim = int(np.asarray(trajectories[0]["observations"][0]).shape[0])
    # infer action space from first traj first action
    first_action = trajectories[0]["actions"][0]
    if isinstance(first_action, (list, tuple, np.ndarray)):
        # continuous
        act_dim = int(np.asarray(first_action).shape[0])
        is_continuous = True
    elif isinstance(first_action, float):
        act_dim = 1
        is_continuous = True
    else:
        # discrete scalar
        is_continuous = False
        # try to infer number of classes from actions present
        all_actions = set()
        for traj in trajectories:
            for a in traj["actions"]:
                all_actions.add(int(a))
        act_dim = int(max(all_actions) + 1) if len(all_actions) > 0 else 2

    # Build arrays
    max_length = int(getattr(args, "max_length", 50))
    X_states, X_actions, X_masks = _build_dataset(trajectories, max_length, state_dim, 
                                                  gym.spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,)) if is_continuous else gym.spaces.Discrete(act_dim))  # type: ignore

    # try to build rewards matrix for RTG if available
    B, L = X_states.shape[0], X_states.shape[1]
    rewards = np.zeros((B, L), dtype=np.float32)
    for i, traj in enumerate(trajectories):
        for t in range(min(len(traj["rewards"]), L)):
            rewards[i, t] = float(traj["rewards"][t])
    returns_to_go = _compute_return_to_go(rewards, X_masks)

    # Convert to torch tensors
    states_t = torch.tensor(X_states, dtype=torch.float32, device=device)  # (B, L, state_dim)
    masks_t = torch.tensor(X_masks, dtype=torch.float32, device=device)
    actions_t = torch.tensor(X_actions, dtype=(torch.float32 if is_continuous else torch.long), device=device)
    rtg_t = torch.tensor(returns_to_go, dtype=torch.float32, device=device)

    # Simple optimizer
    optim = torch.optim.AdamW(model.parameters(), lr=float(getattr(args, "learning_rate", 1e-4)))
    if is_continuous:
        criterion = nn.MSELoss(reduction="none")
    else:
        criterion = nn.CrossEntropyLoss(reduction="none")

    batch_size = int(getattr(args, "batch_size", 64))
    n_epochs = int(getattr(args, "max_iters", 10))
    n_samples = states_t.shape[0]
    steps_per_epoch = max(1, math.ceil(n_samples / batch_size))

    losses = []
    for epoch in range(n_epochs):
        perm = np.random.permutation(n_samples)
        epoch_loss = 0.0
        model.train()
        t0 = time.time()
        for step in range(steps_per_epoch):
            idx = perm[step * batch_size:(step + 1) * batch_size]
            batch_states = states_t[idx]   # (B0, L, state_dim)
            batch_masks = masks_t[idx]
            batch_actions = actions_t[idx]
            batch_rtg = rtg_t[idx]

            optim.zero_grad()

            # Try a sequence of likely model forward signatures:
            # 1) model(batch_states, batch_rtg)  -> returns predicted actions shape (B0, L, act_dim)
            # 2) model(batch_states) -> predicted actions
            # 3) model(batch_states[:,0,:]) -> per-step
            pred = None
            error_msgs = []
            try:
                pred = model(batch_states, batch_rtg)  # common DT interface
            except Exception as e1:
                error_msgs.append(str(e1))
                try:
                    pred = model(batch_states)
                except Exception as e2:
                    error_msgs.append(str(e2))
                    try:
                        # try flattening time dimension in case model expects (B*L, state_dim)
                        B0, L0, Sdim = batch_states.shape
                        flat = batch_states.reshape(B0 * L0, Sdim)
                        pred = model(flat)
                        # reshape back if needed
                        if pred is not None and pred.dim() == 2:
                            # assume (B0*L0, act_dim)
                            pred = pred.reshape(B0, L0, -1)
                    except Exception as e3:
                        error_msgs.append(str(e3))
                        # No valid forward signature found; raise a clear error with messages
                        raise RuntimeError("Model.forward failed for all tried signatures. Errors:\n" + "\n".join(error_msgs))

            # Now compute loss depending on action type
            if pred is None:
                raise RuntimeError("Model produced no prediction (pred is None)")

            # ensure pred shape (B0, L, act_dim)
            if pred.dim() == 2:
                # (B0, act_dim) or (B0*L, act_dim) — try to align
                B0 = batch_states.shape[0]
                L0 = batch_states.shape[1]
                if pred.shape[0] == B0 * L0:
                    pred = pred.reshape(B0, L0, -1)
                elif pred.shape[0] == B0:
                    # expand to L dimension
                    pred = pred.unsqueeze(1).expand(-1, L0, -1)
                else:
                    # ambiguous
                    pass

            # Masked loss accumulation
            if is_continuous:
                # pred and batch_actions should have same shape
                # If actions were stored as shape (B, L) for scalar continuous, expand dims
                if batch_actions.dim() == 2 and pred.size(-1) == 1:
                    target = batch_actions.unsqueeze(-1).float()
                else:
                    target = batch_actions.float()
                per_elem = criterion(pred, target)  # (B0, L, act_dim)
                # reduce over action dim
                per_timestep = per_elem.mean(dim=-1) if per_elem.dim() == 3 else per_elem
                masked = per_timestep * batch_masks
                loss = masked.sum() / (batch_masks.sum() + 1e-8)
            else:
                # discrete: pred logits shape (B,L,num_classes), target shape (B,L)
                # reshape to (B*L, C) and (B*L,)
                B0, L0, C = pred.shape
                logits = pred.reshape(B0 * L0, C)
                targets = batch_actions.reshape(B0 * L0)
                per = criterion(logits, targets)  # (B*L,)
                per = per.reshape(B0, L0)
                masked = per * batch_masks
                loss = masked.sum() / (batch_masks.sum() + 1e-8)

            loss.backward()
            optim.step()

            epoch_loss += float(loss.item())

        epoch_loss = epoch_loss / float(steps_per_epoch)
        losses.append(epoch_loss)
        t1 = time.time()
        print(f"[Train] epoch={epoch} avg_loss={epoch_loss:.6f} time={(t1-t0):.2f}s")

        # checkpoint each epoch
        ckpt_file = os.path.join(log_dir, f"checkpoint_epoch{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optim.state_dict(),
            "loss": epoch_loss,
        }, ckpt_file)
    print("[Train] done. Last checkpoint saved to", ckpt_file)
    return losses


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
