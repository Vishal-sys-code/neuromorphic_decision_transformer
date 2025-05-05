import os
import sys
import torch
import gym
import numpy as np

# Add decision transformer to path
dt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "external/decision-transformer/gym/decision_transformer")
if dt_path not in sys.path:
    sys.path.append(dt_path)

from config import ENVIRONMENTS, DEVICE, epochs, steps_per_epoch, lr, dt_config
from utils.trajectory_buffer import TrajectoryBuffer
from utils.helpers import compute_returns_to_go, simple_logger, save_checkpoint

from models.decision_transformer import DecisionTransformer

def train_cartpole():
    os.makedirs("checkpoints", exist_ok=True)
    env = gym.make("CartPole-v1")
    # override DT config for CartPole
    dt_conf = dt_config.copy()
    dt_conf.update(
        state_dim=env.observation_space.shape[0],
        act_dim=env.action_space.n,
    )

    model = DecisionTransformer(**dt_conf).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        buf = TrajectoryBuffer()
        obs, total_ret = env.reset(), 0
        for t in range(steps_per_epoch):
            # random policy placeholder
            action = env.action_space.sample()
            next_obs, r, done, _ = env.step(action)
            buf.add(obs, action, r)
            obs = next_obs if not done else env.reset()
            total_ret += r

        traj = buf.get_trajectory()
        rtg = compute_returns_to_go(traj["rewards"])
        simple_logger({"epoch_ret": total_ret, "mean_rtg": np.mean(rtg)}, epoch)

    # save final checkpoint
    save_checkpoint(model, opt, "checkpoints/dt_cartpole_baseline.pt")
    print("Baseline training complete.")

if __name__ == "__main__":
    train_cartpole()