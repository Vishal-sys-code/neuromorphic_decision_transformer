import random
import numpy as np
import torch
import gym
from config import ENVIRONMENTS, SEED

# setup seeds for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

def run_random(env_name, episodes=5):
    try:
        env = gym.make(env_name)
        returns = []
        for _ in range(episodes):
            obs, done, ep_ret = env.reset(), False, 0
            while not done:
                action = env.action_space.sample()
                obs, r, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                ep_ret += r
            returns.append(ep_ret)
        print(f"{env_name:15s} random avg return: {np.mean(returns):.2f}")
    except gym.error.DependencyNotInstalled as e:
        print(f"{env_name:15s} skipped - {str(e)}")
    except Exception as e:
        print(f"{env_name:15s} failed - {str(e)}")

if __name__ == "__main__":
    for e in ENVIRONMENTS:
        run_random(e)