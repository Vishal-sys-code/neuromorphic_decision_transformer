"""
Author: Vishal Pandey
Email: pandeyvishal.mlprof@gmail.com
"""
# ────────────────────────────────────────────────────────────────────────────────
# 1) Fix up PYTHONPATH so we can import decision_transformer from external/
# ────────────────────────────────────────────────────────────────────────────────
import os
import sys
import random
import torch
import gym
import numpy as np

# Monkey patch np.bool8 for gym compatibility
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

# project_root = SpikingMindRL/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# Path A: if you cloned the official repo directly under external/decision-transformer/
dt_repo_root = os.path.join(project_root, "external", "decision-transformer")
if os.path.isdir(os.path.join(dt_repo_root, "decision_transformer")):
    # e.g. external/decision-transformer/decision_transformer/__init__.py
    sys.path.insert(0, dt_repo_root)
else:
    # Path B: if you have it under external/decision-transformer/gym/decision_transformer
    alt = os.path.join(dt_repo_root, "gym")
    if os.path.isdir(os.path.join(alt, "decision_transformer")):
        sys.path.insert(0, alt)
    else:
        raise ImportError(
            f"Cannot find decision_transformer package under:\n"
            f"  {dt_repo_root}/decision_transformer or\n"
            f"  {dt_repo_root}/gym/decision_transformer"
        )

# ────────────────────────────────────────────────────────────────────────────────
# 2) Local imports
# ────────────────────────────────────────────────────────────────────────────────

from config import ENVIRONMENTS, DEVICE, epochs, steps_per_epoch, lr, dt_config
from utils.trajectory_buffer import TrajectoryBuffer
from utils.helpers import compute_returns_to_go, simple_logger, save_checkpoint
from external.decision_transformer.gym.decision_transformer.models.decision_transformer import DecisionTransformer

# ────────────────────────────────────────────────────────────────────────────────
# 3) Training code
# ────────────────────────────────────────────────────────────────────────────────

def train_cartpole():
    os.makedirs("checkpoints", exist_ok=True)

    env = gym.make("CartPole-v1")
    # override DT config for this env
    dt_conf = dt_config.copy()
    dt_conf.update(
        state_dim=env.observation_space.shape[0],
        act_dim=env.action_space.n,
    )

    model = DecisionTransformer(**dt_conf).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        buf = TrajectoryBuffer(max_len=steps_per_epoch, state_dim=dt_conf["state_dim"], action_dim=dt_conf["act_dim"])
        obs, _ = env.reset()
        total_ret = 0
        for t in range(steps_per_epoch):
            # random policy placeholder
            action = env.action_space.sample()
            next_obs, r, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buf.add(obs, action, r)
            obs = next_obs if not done else env.reset()[0]
            total_ret += r

        traj = buf.get_trajectory()
        rtg = compute_returns_to_go(traj["rewards"])
        simple_logger({"epoch_ret": total_ret, "mean_rtg": np.mean(rtg)}, epoch)

    save_checkpoint(model, opt, "checkpoints/dt_cartpole_baseline.pt")
    print("✅ Baseline training complete.")

if __name__ == "__main__":
    # for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    train_cartpole()