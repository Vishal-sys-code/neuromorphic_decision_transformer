"""
Author: Vishal Pandey
Email: pandeyvishal.mlprof@gmail.com
"""
import os
import random
import src.patch_numpy_bool
import gym
import numpy as np
import torch
import torch.nn as nn

# ensure top-level src and external in PYTHONPATH
import src.setup_paths

# Re-add the missing alias so gym.env_checker can see it
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

from src.config import (
    DEVICE, SEED,
    steps_per_epoch, epochs, gamma,
    max_length, time_window,
    lr,
    dt_config,
)
from src.utils.trajectory_buffer import TrajectoryBuffer
from src.utils.helpers import compute_returns_to_go, simple_logger, save_checkpoint
from src.models.snn_dt_patch import SNNDecisionTransformer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)

def train_snn_dt(env_name="CartPole-v1"):
    # 1. setup
    set_seed(SEED)
    os.makedirs("checkpoints", exist_ok=True)
    env = gym.make(env_name)
    # override dims
    dt_conf = dt_config.copy()
    dt_conf.update(
        state_dim=env.observation_space.shape[0],
        act_dim=env.action_space.n,
        max_length=max_length,
        time_window=time_window,
    )
    model = SNNDecisionTransformer(**dt_conf).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # 2. epochs
    for epoch in range(epochs):
        buf = TrajectoryBuffer(max_length, dt_conf['state_dim'], dt_conf['act_dim'])
        obs, _ = env.reset()
        ep_ret = 0

        # 2a. collect steps
        for t in range(steps_per_epoch):
            # build history so far
            traj = buf.get_trajectory()
            states_np = traj["states"]
            actions_np= traj["actions"]
            rewards_np= traj["rewards"]

            # pad or crop history to max_length
            L = len(states_np)
            start = max(0, L - max_length)
            s_hist = states_np[start:]
            a_hist = actions_np[start:].reshape(-1,1)
            r_hist = rewards_np[start:]
            # compute returns-to-go
            rtg = compute_returns_to_go(r_hist, gamma=gamma)

            # timesteps array
            max_ep_len = dt_conf.get('max_ep_len', 4096)
            timesteps = np.arange(start, start + len(s_hist))
            timesteps = np.clip(timesteps, 0, max_ep_len - 1)

            # convert to tensors
            states_t = torch.tensor(s_hist, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            actions_t= torch.tensor(a_hist, dtype=torch.long,   device=DEVICE).unsqueeze(0)
            # convert actions to one-hot encoding
            actions_t_onehot = torch.nn.functional.one_hot(actions_t.squeeze(-1), num_classes=dt_conf['act_dim']).to(torch.float32).unsqueeze(0)
            rtg_t    = torch.tensor(rtg.reshape(-1,1), dtype=torch.float32, device=DEVICE).unsqueeze(0)
            tim_t    = torch.tensor(timesteps, dtype=torch.long, device=DEVICE).unsqueeze(0)

            # sample action via SNN-DT policy (returns a vector of logits)
            with torch.no_grad():
                action_logits = model.get_action(
                    states_t, actions_t_onehot, None, rtg_t, tim_t
                )  # shape [act_dim]
            action = int(torch.argmax(action_logits).item())

            # step env
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            obs_arr = np.array(obs, dtype=np.float32)
            buf.add(obs_arr, action, reward)
            obs = next_obs if not done else env.reset()[0]
            ep_ret += reward

        # 2b. on-policy update at epoch end
        traj = buf.get_trajectory()
        S = len(traj["states"])
        # prepare full batch
        states_b = torch.tensor(traj["states"], dtype=torch.float32, device=DEVICE).unsqueeze(0)
        actions_idx = torch.tensor(traj["actions"], dtype=torch.long, device=DEVICE)
        actions_b   = torch.nn.functional.one_hot(actions_idx, num_classes=dt_conf['act_dim'])\
                .to(torch.float32).unsqueeze(0)
        rtg       = compute_returns_to_go(traj["rewards"], gamma=gamma)
        rtg_b    = torch.tensor(rtg.reshape(-1,1), dtype=torch.float32, device=DEVICE).unsqueeze(0)
        max_ep_len = dt_conf.get("max_ep_len", 4096)
        timesteps = torch.arange(S, device=DEVICE)
        timesteps = timesteps.clamp(max= max_ep_len - 1)
        tim_b = timesteps.unsqueeze(0)

        # forward pass
        model.train()
        optimizer.zero_grad()
        _, action_preds, _ = model(states_b, actions_b, None, rtg_b, tim_b)
        # action_preds: [1, S, act_dim]
        logits  = action_preds.view(-1, model.act_dim)    # [S, act_dim]
        targets = actions_idx                   # [S]
        loss = loss_fn(logits, targets)

        # backward & step
        loss.backward()
        optimizer.step()

        # logging & checkpoint
        simple_logger(
            {"epoch": epoch, "return": ep_ret, "loss": loss.item()}, epoch
        )
        save_checkpoint(model, optimizer, f"checkpoints/snn_dt_{env_name}_{epoch}.pt")

    print("✅ SNN-DT training complete.")

if __name__ == "__main__":
    train_snn_dt()