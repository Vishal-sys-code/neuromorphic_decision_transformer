"""
Author: Vishal Pandey
Email: pandeyvishal.mlprof@gmail.com

REINFORCE‐style policy-gradient on Spiking Decision Transformer
"""
import os
os.environ["GYM_DISABLE_ENV_CHECKER"] = "true"
import random
import numpy as np

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

import src.patch_numpy_bool  # must before gym import
import gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

# ensure top-level src and external in PYTHONPATH
import src.setup_paths

from src.config import (
    DEVICE, SEED,
    steps_per_epoch, epochs, gamma,
    max_length, time_window,
    lr, dt_config,
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
    # Setup
    set_seed(SEED)
    os.makedirs("checkpoints", exist_ok=True)
    env = gym.make(env_name)

    # Build model
    dt_conf = dt_config.copy()
    dt_conf.update(
        state_dim=env.observation_space.shape[0],
        act_dim=env.action_space.n,
        max_length=max_length,
        time_window=time_window,
    )
    model = SNNDecisionTransformer(**dt_conf).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        # Storage for PG
        buf = TrajectoryBuffer(max_length, dt_conf['state_dim'], dt_conf['act_dim'])
        log_probs = []
        rewards   = []
        ep_returns = []
        ep_ret = 0

        obs = env.reset()[0]

        # Collect steps
        for t in range(steps_per_epoch):
            # 1) Build history window
            traj = buf.get_trajectory()
            states_np  = traj["states"]
            actions_np = traj["actions"]
            rewards_np = traj["rewards"]

            L = len(states_np)
            start = max(0, L - max_length)
            s_hist = states_np[start:]
            a_hist = actions_np[start:].reshape(-1,1)
            r_hist = rewards_np[start:]

            # returns-to-go for conditioning (optional in PG)
            rtg = compute_returns_to_go(r_hist, gamma=gamma)

            # timesteps array (clamped)
            max_ep_len = dt_conf.get("max_ep_len", 4096)
            timesteps = np.arange(start, start + len(s_hist))
            timesteps = np.clip(timesteps, 0, max_ep_len - 1)

            # to tensors
            states_t = torch.tensor(s_hist, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            # one-hot previous actions (for autoregression)
            if len(a_hist)>0:
                actions_t_onehot = torch.nn.functional.one_hot(
                    torch.tensor(a_hist, dtype=torch.long, device=DEVICE).squeeze(-1),
                    num_classes=dt_conf['act_dim']
                ).to(torch.float32).unsqueeze(0)
            else:
                # no history: all zeros
                actions_t_onehot = torch.zeros((1, 0, dt_conf['act_dim']), device=DEVICE)
            rtg_t   = torch.tensor(rtg.reshape(-1,1), dtype=torch.float32, device=DEVICE).unsqueeze(0)
            tim_t   = torch.tensor(timesteps, dtype=torch.long, device=DEVICE).unsqueeze(0)

            # 2) Infer action logits (we need gradients for log_prob later)
            action_logits = model.get_action(
                states_t, actions_t_onehot, None, rtg_t, tim_t
            )  # shape [act_dim]

            # 3) Sample action
            probs = torch.softmax(action_logits, dim=-1)
            dist  = Categorical(probs)
            action = dist.sample().item()
            log_probs.append(dist.log_prob(torch.tensor(action, device=DEVICE)))

            # 4) Step env
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            buf.add(np.array(obs, dtype=np.float32), action, reward)
            obs = next_obs if not done else env.reset()[0]
            ep_ret += reward
            rewards.append(reward)

            if done:
                ep_returns.append(ep_ret)
                ep_ret = 0

        # End of epoch: compute policy gradient update
        # 1) Compute discounted returns G
        G = compute_returns_to_go(np.array(rewards), gamma=gamma)
        G = torch.tensor(G, dtype=torch.float32, device=DEVICE)

        # 2) Stack log_probs
        log_probs_t = torch.stack(log_probs)  # [N]

        # 3) Optionally normalize returns to reduce variance
        G = (G - G.mean()) / (G.std(unbiased=False) + 1e-8)

        # 4) Policy loss
        policy_loss = -(log_probs_t * G).mean()

        # 5) Backprop & update
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        # Logging & checkpointing
        avg_ret = float(np.mean(ep_returns)) if ep_returns else 0.0
        simple_logger({
            "epoch": epoch,
            "avg_ep_return": avg_ret,
            "policy_loss": policy_loss.item()
        }, epoch)

        save_checkpoint(model, optimizer, f"checkpoints/snn_dt_pg_{env_name}_{epoch}.pt")

    print("✅ SNN-DT policy-gradient training complete.")


if __name__ == "__main__":
    train_snn_dt()