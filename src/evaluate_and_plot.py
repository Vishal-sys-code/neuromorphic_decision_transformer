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
        act_dim=(env.action_space.n if hasattr(env.action_space,"n")
                 else env.action_space.shape[0]),
        max_length=max_length
    )
    model = SNNDecisionTransformer(**cfg).to(DEVICE)
    ckpt = torch.load(f"checkpoints/offline_dt_{env_name}_{dt_epochs-1}.pt", map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model

def evaluate(env_name, model, episodes=20):
    env = gym.make(env_name)
    rets, lengths = [], []
    for _ in range(episodes):
        obs = env.reset()[0]
        done, ep_ret, ep_len = False, 0, 0
        states, actions, rewards = [], [], []
        while not done and ep_len <  max_length:
            states.append(obs.astype(np.float32))
            # prepare DT inputs just like training…
            start = max(0, len(states)-max_length)
            s = torch.tensor(states[start:], device=DEVICE).unsqueeze(0)
            if actions:
                a = torch.nn.functional.one_hot(
                    torch.tensor(actions[start:], device=DEVICE),
                    num_classes=model.act_dim
                ).unsqueeze(0).float()
            else:
                a = torch.zeros((1,0,model.act_dim), device=DEVICE)
            rtg = compute_returns_to_go(np.array(rewards[start:]), gamma=gamma)
            rtg_t = torch.tensor(rtg.reshape(-1,1), device=DEVICE).unsqueeze(0)
            tim = torch.arange(start, start+len(states[start:]), device=DEVICE).clamp(max=max_length-1).unsqueeze(0)

            with torch.no_grad():
                logits = model.get_action(s, a, None, rtg_t, tim)
            action = int(logits.argmax().item())
            obs, r, term, trunc, _ = env.step(action)
            done = term or trunc
            actions.append(action); rewards.append(r)
            ep_ret += r; ep_len += 1

        rets.append(ep_ret); lengths.append(ep_len)
    return np.mean(rets), np.mean(lengths)

if __name__=="__main__":
    envs = ["CartPole-v1","MountainCar-v0","Acrobot-v1","Pendulum-v1"]
    rows = []
    for e in envs:
        print("Evaluating", e)
        m = load_model(e)
        r, l = evaluate(e, m, episodes=20)
        rows.append({"Environment": e, "AvgReturn": r, "AvgLength": l})

    df = pd.DataFrame(rows)
    print(df)

    # Bar chart of returns
    plt.figure(figsize=(6,4))
    plt.bar(df["Environment"], df["AvgReturn"])
    plt.ylabel("Avg Return"); plt.title("Agent Performance by Env")
    plt.tight_layout()
    plt.savefig("multi_env_returns.png")
    print("Saved multi_env_returns.png")

    # If you logged losses to disk, you can load and plot them:
    # for e in envs:
    #   losses = np.loadtxt(f"logs/offline_loss_{e}.txt")
    #   plt.plot(range(len(losses)), losses, label=e)
    # plt.legend(); plt.xlabel("Epoch"); plt.ylabel("Loss")
    # plt.savefig("offline_loss_curves.png")