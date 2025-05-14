# 1) Patch NumPy for Gym’s env_checker and classic_control
import numpy as np
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
if not hasattr(np, "float_"):
    np.float_ = np.float64

# 2) Imports
import os
import gym
import pickle
from stable_baselines3 import PPO

# 3) Settings
ENVS      = ["CartPole-v1", "MountainCar-v0", "Acrobot-v1", "Pendulum-v1"]
TIMESTEPS = 50_000  # PPO timesteps per environment
EPISODES  = 100     # number of roll‑outs to save per environment
OUT_DIR   = "demos_sb3"

os.makedirs(OUT_DIR, exist_ok=True)

# 4) Loop through each environment
for env_name in ENVS:
    print(f"\n=== {env_name} ===")
    # 4a) Create and train PPO
    env = gym.make(env_name)
    model = PPO("MlpPolicy", env, verbose=0)
    print(f" Training PPO for {TIMESTEPS} timesteps…", end="", flush=True)
    model.learn(total_timesteps=TIMESTEPS)
    print(" done.")

    # 4b) Roll out expert demos
    demos = []
    for ep in range(EPISODES):
        obs, _ = env.reset()
        traj = {"states": [], "actions": [], "rewards": []}
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            nxt, r, term, trunc, _ = env.step(action)
            traj["states"].append(np.array(obs, dtype=np.float32))
            traj["actions"].append(action)
            traj["rewards"].append(r)
            obs, done = nxt, (term or trunc)
        demos.append(traj)
        if (ep+1) % 20 == 0:
            print(f"  Collected {ep+1}/{EPISODES} demos…", end="\r", flush=True)

    # 4c) Save to disk
    out_path = os.path.join(OUT_DIR, f"expert_{env_name}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(demos, f)
    print(f" Saved {len(demos)} expert demos to {out_path}")