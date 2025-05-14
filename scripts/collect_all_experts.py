# 1) Patch NumPy for Gym compatibility
import numpy as np
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
if not hasattr(np, "float_"):
    np.float_ = np.float64

# 2) Imports
import gym
import pickle
from stable_baselines3 import PPO
import os

import warnings
warnings.filterwarnings("ignore", category=UserWarning,
    message="You provided an OpenAI Gym environment.*")

# Number of expert episodes per env
NUM_EPISODES = 200
# Number of PPO timesteps for the non‐CartPole envs
PPO_TIMESTEPS = 50_000

ENVIRONMENTS = [
    "CartPole-v1",
    "MountainCar-v0",
    "Acrobot-v1",
    "Pendulum-v1",
]

def collect_cartpole(env_name, episodes):
    env = gym.make(env_name)
    demos = []
    for _ in range(episodes):
        obs, _ = env.reset()
        traj = {"states": [], "actions": [], "rewards": []}
        done = False
        while not done:
            # simple heuristic: push in direction of velocity
            action = 0 if obs[1] < 0 else 1
            nxt, r, term, trunc, _ = env.step(action)
            traj["states"].append(obs.astype(np.float32))
            traj["actions"].append(action)
            traj["rewards"].append(r)
            obs, done = nxt, term or trunc
        demos.append(traj)
    return demos

def collect_with_ppo(env_name, episodes, timesteps):
    env = gym.make(env_name)
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=timesteps)

    demos = []
    for _ in range(episodes):
        obs, _ = env.reset()
        traj = {"states": [], "actions": [], "rewards": []}
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            nxt, r, term, trunc, _ = env.step(action)
            traj["states"].append(np.array(obs, dtype=np.float32))
            traj["actions"].append(action)
            traj["rewards"].append(r)
            obs, done = nxt, term or trunc
        demos.append(traj)
    return demos

if __name__ == "__main__":
    os.makedirs("demos", exist_ok=True)
    for env_name in ENVIRONMENTS:
        print(f"\nCollecting {NUM_EPISODES} expert episodes for {env_name} …")
        if env_name == "CartPole-v1":
            demos = collect_cartpole(env_name, NUM_EPISODES)
        else:
            demos = collect_with_ppo(env_name, NUM_EPISODES, PPO_TIMESTEPS)

        path = os.path.join("demos", f"expert_{env_name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(demos, f)
        print(f"→ Saved {len(demos)} demos to {path}")