# src/evaluate_multi_env.py
"""
Multi-Environment Evaluation for (Spiking) Decision Transformer
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict

import numpy as _np
if not hasattr(_np, "bool8"):
    _np.bool8 = _np.bool_

os.environ["GYM_DISABLE_ENV_CHECKER"] = "true"

# ensure our packages are on PYTHONPATH
import src.setup_paths

from src.config import DEVICE, dt_config, dt_epochs, gamma, max_length
from src.utils.helpers import compute_returns_to_go
from src.models.snn_dt_patch import SNNDecisionTransformer

def load_model_for_env(env_name, checkpoint_epoch=None):
    # Build model config
    env = gym.make(env_name)
    cfg = dt_config.copy()
    cfg.update(
        state_dim=env.observation_space.shape[0],
        act_dim=env.action_space.n,
        max_length=max_length,
        time_window=cfg.get("time_window", 5),
    )
    model = SNNDecisionTransformer(**cfg).to(DEVICE)
    # Default to last epoch if not specified
    ep = checkpoint_epoch if checkpoint_epoch is not None else dt_epochs - 1
    ckpt = torch.load(f"checkpoints/offline_dt_{env_name}_{ep}.pt", map_location=DEVICE)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model

def evaluate_model(model, env_name, num_episodes=20):
    env = gym.make(env_name)
    stats = defaultdict(list)

    for _ in range(num_episodes):
        # reset
        obs = env.reset()[0]
        done = False
        total_return = 0
        total_length = 0

        # history buffers
        states, actions, rewards = [], [], []

        while not done:
            # record state
            states.append(np.array(obs, dtype=np.float32))

            # prepare sequence slices
            L = len(states)
            start = max(0, L - max_length)
            s_seq = states[start:]
            a_seq = actions[start:]
            r_seq = rewards[start:]

            # convert s_seq to tensor once
            s_arr = np.stack(s_seq, axis=0)  # shape [seq_len, state_dim]
            s_t   = torch.from_numpy(s_arr).to(DEVICE).unsqueeze(0)  # [1, seq_len, state_dim]

            # prepare action history one-hot
            if a_seq:
                a_idx = torch.tensor(a_seq, dtype=torch.long, device=DEVICE)
                a_t   = torch.nn.functional.one_hot(a_idx, num_classes=model.act_dim)\
                            .to(torch.float32).unsqueeze(0)  # [1, seq_len, act_dim]
            else:
                a_t = torch.zeros((1, 0, model.act_dim), device=DEVICE)

            # returns-to-go conditioning
            rtg = compute_returns_to_go(np.array(r_seq), gamma=gamma)
            rtg_t = torch.from_numpy(rtg.reshape(-1,1)).to(DEVICE).unsqueeze(0)  # [1, seq_len, 1]

            # timesteps clamped to embedding size
            max_ep_len = model.embed_timestep.num_embeddings
            tim_idx = np.arange(start, start + len(s_seq))
            tim_idx = np.clip(tim_idx, 0, max_ep_len - 1)
            tim_t   = torch.from_numpy(tim_idx).to(DEVICE).unsqueeze(0)  # [1, seq_len]

            # get action logits
            with torch.no_grad():
                logits = model.get_action(s_t, a_t, None, rtg_t, tim_t)
            action = int(torch.argmax(logits, dim=-1).item())

            # step environment
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # update buffers
            actions.append(action)
            rewards.append(reward)
            obs = next_obs
            total_return += reward
            total_length += 1

        stats["returns"].append(total_return)
        stats["lengths"].append(total_length)

    avg_ret = float(torch.tensor(stats["returns"], dtype=torch.float32).mean().item())
    avg_len = float(torch.tensor(stats["lengths"], dtype=torch.float32).mean().item())
    return avg_ret, avg_len

def main():
    envs = ["CartPole-v1", "MountainCar-v0", "LunarLander-v2"]
    results = []

    for env_name in envs:
        print(f"Evaluating {env_name} …")
        model = load_model_for_env(env_name)
        avg_ret, avg_len = evaluate_model(model, env_name, num_episodes=20)
        results.append((env_name, avg_ret, avg_len))

    # print table
    print("\nEnvironment      AvgReturn    AvgLength")
    print("---------------------------------------")
    for env_name, avg_ret, avg_len in results:
        print(f"{env_name:15s}  {avg_ret:10.2f}    {avg_len:8.1f}")

    # plot
    names = [r[0] for r in results]
    returns = [r[1] for r in results]
    plt.figure()
    plt.bar(names, returns)
    plt.ylabel("Average Return")
    plt.title("Offline DT – Average Return by Environment")
    plt.tight_layout()
    os.makedirs("visualisation_images", exist_ok=True)
    plt.savefig("visualisation_images/multi_env_returns.png")
    print("\nSaved plot to visualisation_images/multi_env_returns.png")

if __name__ == "__main__":
    main()