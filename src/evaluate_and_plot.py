# src/evaluate_and_plot.py
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Patch numpy for Gym
import numpy as np
if not hasattr(np, "bool8"): np.bool8 = np.bool_
if not hasattr(np, "float_"): np.float_ = np.float64
os.environ["GYM_DISABLE_ENV_CHECKER"] = "true"

import gym
import torch
import pandas as pd
import matplotlib.pyplot as plt

from src.config import DEVICE, dt_config, dt_epochs, gamma, max_length
from src.models.snn_dt_patch import SNNDecisionTransformer
from src.utils.helpers import compute_returns_to_go

def load_model(env_name):
    cfg = dt_config.copy()
    env = gym.make(env_name)
    cfg.update(
        state_dim=env.observation_space.shape[0],
        act_dim=(env.action_space.n
                  if hasattr(env.action_space, "n")
                  else env.action_space.shape[0]),
        max_length=max_length
    )
    model = SNNDecisionTransformer(**cfg).to(DEVICE)
    ckpt = torch.load(f"checkpoints/offline_dt_{env_name}_{dt_epochs-1}.pt",
                      map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model

def evaluate(env_name, model, episodes=20):
    env = gym.make(env_name)
    discrete = hasattr(env.action_space, "n")
    stats = {"returns": [], "lengths": []}

    for _ in range(episodes):
        obs = env.reset()[0]
        done, ep_ret, ep_len = False, 0, 0
        states, actions, rewards = [], [], []

        while not done:
            states.append(obs.astype(np.float32))

            # build batch array once
            start = max(0, len(states) - max_length)
            s_arr = np.stack(states[start:], axis=0)       # [seq_len, state_dim]
            s_t   = torch.from_numpy(s_arr).to(DEVICE).unsqueeze(0)  # [1, seq_len, state_dim]

            if discrete:
                if actions:
                    a_idx = np.array(actions[start:], dtype=np.int64)
                    a_in  = torch.nn.functional.one_hot(
                        torch.from_numpy(a_idx).to(DEVICE),
                        num_classes=model.act_dim
                    ).unsqueeze(0).to(torch.float32)      # [1, seq_len, act_dim]
                else:
                    a_in = torch.zeros((1, 0, model.act_dim), device=DEVICE)
            else:
                # continuous: we pass previous raw actions as floats
                if actions:
                    a_arr = np.array(actions[start:], dtype=np.float32).reshape(-1,1)
                    a_in  = torch.from_numpy(a_arr).to(DEVICE).unsqueeze(0)  # [1, seq_len, 1]
                else:
                    a_in = torch.zeros((1, 0, 1), device=DEVICE)

            # returns-to-go
            r_arr = np.array(rewards[start:], dtype=np.float32).reshape(-1,1)
            rtg = compute_returns_to_go(r_arr.flatten(), gamma=gamma).reshape(-1,1)
            rtg_t = torch.from_numpy(rtg).to(DEVICE).unsqueeze(0)        # [1, seq_len, 1]

            # timesteps
            tim_idx = np.arange(start, start + s_arr.shape[0])
            tim_idx = np.clip(tim_idx, 0, max_length - 1)
            tim_t   = torch.from_numpy(tim_idx).to(DEVICE).unsqueeze(0)  # [1, seq_len]

            # get action logits or values
            with torch.no_grad():
                logits = model.get_action(s_t, a_in, None, rtg_t, tim_t)

            if discrete:
                action = int(torch.argmax(logits, dim=-1).item())
            else:
                # continuous: logits is a tensor [1, act_dim] so extract scalar
                action = float(logits.item())

            # wrap continuous action as array of shape (1,)
            env_action = action if discrete else np.array([action], dtype=np.float32)
            obs, reward, term, trunc, _ = env.step(env_action)
            done = term or trunc

            actions.append(action)
            rewards.append(reward)
            ep_ret += reward
            ep_len += 1

        stats["returns"].append(ep_ret)
        stats["lengths"].append(ep_len)

    return np.mean(stats["returns"]), np.mean(stats["lengths"])

if __name__ == "__main__":
    envs = ["CartPole-v1", "MountainCar-v0", "Acrobot-v1", "Pendulum-v1"]
    records = []

    for e in envs:
        print(f"Evaluating {e}")
        model = load_model(e)
        avg_ret, avg_len = evaluate(e, model, episodes=20)
        records.append({"Environment": e, "AvgReturn": avg_ret, "AvgLength": avg_len})

    df = pd.DataFrame(records)
    print(df)

    # Plot average returns
    plt.figure(figsize=(6,4))
    plt.bar(df["Environment"], df["AvgReturn"])
    plt.ylabel("Average Return")
    plt.title("Offline DT Evaluation")
    plt.tight_layout()
    plt.savefig("multi_env_returns.png")
    print("Saved multi_env_returns.png")