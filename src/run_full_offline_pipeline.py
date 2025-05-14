import os, sys, pickle, random
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 1) Patch NumPy
import numpy as np
if not hasattr(np, "bool8"):  np.bool8 = np.bool_
if not hasattr(np, "float_"): np.float_ = np.float64
os.environ["GYM_DISABLE_ENV_CHECKER"] = "true"

import gym
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# Project modules
import src.setup_paths
from src.config import (
    DEVICE, SEED,
    offline_steps, batch_size, dt_epochs,
    gamma, max_length, lr, dt_config
)
from src.utils.trajectory_buffer import TrajectoryBuffer
from src.utils.helpers import compute_returns_to_go, simple_logger
from src.models.snn_dt_patch import SNNDecisionTransformer

# Environments to process
ENVS = ["CartPole-v1", "MountainCar-v0", "Acrobot-v1", "Pendulum-v1"]
# Expert demo files, if you have them:
DEMO_FILES = { env: f"expert_{env}.pkl" for env in ENVS }

class TrajectoryDataset(Dataset):
    def __init__(self, trajs):
        self.seqs = []
        for traj in trajs:
            states  = traj["states"]
            actions = np.array(traj["actions"])
            returns = compute_returns_to_go(traj["rewards"], gamma).reshape(-1,1)
            timesteps = np.arange(len(states)).reshape(-1,1)
            T = len(states)
            for i in range(1, T+1):
                start = max(0, i - max_length)
                self.seqs.append({
                    "states":    states[start:i],
                    "actions":   actions[start:i],
                    "returns":   returns[start:i],
                    "timesteps": timesteps[start:i],
                })
    def __len__(self): return len(self.seqs)
    def __getitem__(self, idx):
        s = self.seqs[idx]; L = len(s["states"]); pad = max_length - L
        pad_s = np.zeros((pad, s["states"].shape[1]), dtype=np.float32)
        pad_r = np.zeros((pad,1), dtype=np.float32)
        pad_t = np.zeros((pad,1), dtype=np.int64)

        states    = np.vstack([pad_s, s["states"]]).astype(np.float32)
        returns   = np.vstack([pad_r, s["returns"].astype(np.float32)])
        timesteps = np.vstack([pad_t, s["timesteps"]]).astype(np.int64)

        a = s["actions"]
        if a.ndim==1:  # discrete
            pad_a = np.zeros((pad,), dtype=np.int64)
            actions = np.concatenate([pad_a, a]).reshape(-1,1).astype(np.int64)
        else:          # continuous
            pad_a = np.zeros((pad, a.shape[1]), dtype=np.float32)
            actions = np.vstack([pad_a, a.astype(np.float32)])

        return {
            "states":        torch.from_numpy(states).to(DEVICE),
            "actions":       torch.from_numpy(actions).to(DEVICE),
            "returns_to_go": torch.from_numpy(returns).to(DEVICE),
            "timesteps":     torch.from_numpy(timesteps.squeeze(-1)).to(DEVICE),
        }

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if DEVICE.startswith("cuda"): torch.cuda.manual_seed_all(seed)

def collect_random(env_name, state_dim, act_dim):
    env = gym.make(env_name)
    buf = TrajectoryBuffer(max_length, state_dim, act_dim)
    trajs, steps = [], 0
    obs = env.reset()[0]
    while steps < offline_steps:
        action = env.action_space.sample()
        nxt, r, term, trunc, _ = env.step(action)
        buf.add(obs.astype(np.float32), action, r)
        obs = nxt if not (term or trunc) else env.reset()[0]
        steps += 1
        if term or trunc:
            trajs.append(buf.get_trajectory()); buf.reset()
    return trajs

def train_and_eval(env_name):
    print(f"\n=== ENV {env_name} ===")
    set_seed(SEED)
    # 1) infer dims & type
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    if hasattr(env.action_space, "n"):
        act_dim, act_type = env.action_space.n, "discrete"
    else:
        act_dim, act_type = env.action_space.shape[0], "continuous"

    # 2) load expert demos if exist
    trajs = []
    demo_f = DEMO_FILES[env_name]
    if os.path.isfile(demo_f):
        with open(demo_f,"rb") as f: trajs = pickle.load(f)
        print(f"  → Loaded {len(trajs)} expert demos")
    
    # 3) collect the rest random
    print(f"  • Collecting {offline_steps} random steps …", end="")
    r_trajs = collect_random(env_name, state_dim, act_dim)
    print(" done")
    trajs += r_trajs

    # 4) dataset & loader
    dataset = TrajectoryDataset(trajs)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # 5) build model
    cfg = dt_config.copy()
    cfg.update(state_dim=state_dim, act_dim=act_dim, max_length=max_length)
    model = SNNDecisionTransformer(**cfg).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss() if act_type=="discrete" else nn.MSELoss()

    # 6) offline training
    losses = []
    print(f"  • Training offline for {dt_epochs} epochs …")
    for ep in range(dt_epochs):
        tot = 0.0
        for batch in loader:
            states  = batch["states"]       # [B,L,S]
            acts    = batch["actions"]      # [B,L,1] or [B,L,dim]
            rtg     = batch["returns_to_go"]# [B,L,1]
            times   = batch["timesteps"]    # [B,L]

            if act_type=="discrete":
                a_idx = acts.squeeze(-1).long()
                a_in  = nn.functional.one_hot(a_idx, act_dim).to(torch.float32)
                _, pred, _ = model(states, a_in, None, rtg, times)
                logits = pred.view(-1, act_dim)
                tgt    = a_idx.view(-1)
                loss   = loss_fn(logits, tgt)
            else:
                _, pred, _ = model(states, acts, None, rtg, times)
                loss = loss_fn(pred, acts)

            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        avg = tot/len(loader)
        simple_logger({"epoch": ep, "loss": avg}, ep)
        losses.append(avg)
        torch.save({
            "model_state": model.state_dict(),
            "optim_state": opt.state_dict()
        }, f"checkpoints/offline_dt_{env_name}_{ep}.pt")
    print("  • Done. Final loss:", losses[-1])

    # 7) evaluate online
    print("  • Evaluating online …")
    rets, lens = [], []
    for _ in range(20):
        obs = env.reset()[0]; done, ret, length = False, 0, 0
        hist_s, hist_a, hist_r = [], [], []
        while not done:
            hist_s.append(obs.astype(np.float32))
            # build input
            start = max(0, len(hist_s)-max_length)
            s_arr = np.stack(hist_s[start:],axis=0)
            s_t   = torch.from_numpy(s_arr).to(DEVICE).unsqueeze(0)
            if act_type=="discrete":
                if hist_a:
                    a_arr = np.array(hist_a[start:],dtype=np.int64)
                    a_in  = nn.functional.one_hot(
                        torch.from_numpy(a_arr).to(DEVICE),
                        num_classes=act_dim).unsqueeze(0).to(torch.float32)
                else:
                    a_in = torch.zeros((1,0,act_dim),device=DEVICE)
            else:
                if hist_a:
                    a_arr = np.array(hist_a[start:],dtype=np.float32).reshape(-1,act_dim)
                    a_in  = torch.from_numpy(a_arr).to(DEVICE).unsqueeze(0)
                else:
                    a_in = torch.zeros((1,0,act_dim),device=DEVICE)
            rtg = compute_returns_to_go(np.array(hist_r[start:],dtype=np.float32),gamma).reshape(-1,1)
            rtg_t = torch.from_numpy(rtg).to(DEVICE).unsqueeze(0)
            tim = np.clip(
                np.arange(start, start+s_arr.shape[0]), 0, max_length-1)
            tim_t = torch.from_numpy(tim).to(DEVICE).unsqueeze(0)

            with torch.no_grad():
                out = model.get_action(s_t, a_in, None, rtg_t, tim_t)
            if act_type=="discrete":
                action = int(out.argmax().item())
            else:
                action = float(out.item())
            step_a = action if act_type=="discrete" else np.array([action],dtype=np.float32)
            obs, r, term, trunc, _ = env.step(step_a)
            done = term or trunc
            hist_a.append(action); hist_r.append(r)
            ret += r; length += 1
        rets.append(ret); lens.append(length)
    avg_ret = sum(rets)/len(rets)
    avg_len = sum(lens)/len(lens)
    print(f"  → AvgReturn={avg_ret:.1f}, AvgLen={avg_len:.1f}")

    return losses, avg_ret

if __name__=="__main__":
    ALL_LOSSES, RESULTS = {}, {}
    for e in ENVS:
        losses, avg_ret = train_and_eval(e)
        ALL_LOSSES[e] = losses
        RESULTS[e]     = avg_ret

    # Plot loss curves
    plt.figure(figsize=(6,4))
    for e, ls in ALL_LOSSES.items():
        plt.plot(ls, label=e)
    plt.xlabel("Epoch"); plt.ylabel("Offline Loss")
    plt.legend(); plt.tight_layout()
    plt.savefig("offline_loss_curves.png")
    print("Saved offline_loss_curves.png")

    # Plot avg returns
    plt.figure(figsize=(6,4))
    plt.bar(RESULTS.keys(), RESULTS.values())
    plt.ylabel("AvgReturn"); plt.title("Offline DT Performance")
    plt.tight_layout()
    plt.savefig("visualisation_images/multi_env_returns.png")
    print("Saved multi_env_returns.png")