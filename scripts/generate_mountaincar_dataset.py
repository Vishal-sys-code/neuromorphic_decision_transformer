# Patches NumPy for Gym’s env_checker and classic_control, which is required by Stable Baselines 3.
import numpy as np
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
if not hasattr(np, "float_"):
    np.float_ = np.float64

import os
import pickle
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# --- Configuration ---
ENV_NAME = "MountainCar-v0"
OUT_DIR = "data"
OUT_FILENAME = os.path.join(OUT_DIR, "mountaincar_v0_mixed.pkl")

# Dataset composition
N_EXPERT_TRAJS = 50
N_MEDIUM_TRAJS = 25
N_RANDOM_TRAJS = 25
TOTAL_TRAJS = N_EXPERT_TRAJS + N_MEDIUM_TRAJS + N_RANDOM_TRAJS

# PPO training steps for different agent qualities
EXPERT_TIMESTEPS = 100_000
MEDIUM_TIMESTEPS = 25_000

# --- Helper Function to Collect Trajectories ---
def collect_trajectories(model, env, n_trajs, is_random=False):
    """Collects trajectories from a given model and environment."""
    trajectories = []
    for i in range(n_trajs):
        obs, _ = env.reset()
        done = False
        states, actions, rewards, returns_to_go = [], [], [], []
        
        while not done:
            if is_random:
                action = env.action_space.sample()
            else:
                action, _ = model.predict(obs, deterministic=True)

            next_obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc

            states.append(obs)
            actions.append(action)
            rewards.append(reward)
            
            obs = next_obs
        
        # Calculate returns-to-go
        discounted_returns = np.zeros_like(rewards)
        cumulative_return = 0
        for t in reversed(range(len(rewards))):
            cumulative_return = rewards[t] + 0.99 * cumulative_return
            discounted_returns[t] = cumulative_return
            
        trajectories.append({
            "observations": np.array(states, dtype=np.float32),
            "actions": np.array(actions, dtype=np.int64),
            "rewards": np.array(rewards, dtype=np.float32),
            "returns_to_go": np.array(discounted_returns, dtype=np.float32),
        })
    return trajectories

# --- Main Script Logic ---
if __name__ == "__main__":
    print(f"Generating dataset for {ENV_NAME}...")
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # 1. Create the environment
    env = gym.make(ENV_NAME)

    # 2. Train the expert model
    print(f"Training expert PPO model for {EXPERT_TIMESTEPS} timesteps...")
    expert_model = PPO("MlpPolicy", env, verbose=0)
    expert_model.learn(total_timesteps=EXPERT_TIMESTEPS)
    print("Expert model training complete.")

    # 3. Train the medium model
    print(f"Training medium PPO model for {MEDIUM_TIMESTEPS} timesteps...")
    medium_model = PPO("MlpPolicy", env, verbose=0)
    medium_model.learn(total_timesteps=MEDIUM_TIMESTEPS)
    print("Medium model training complete.")

    # 4. Collect trajectories
    print(f"Collecting {N_EXPERT_TRAJS} expert trajectories...")
    expert_trajs = collect_trajectories(expert_model, env, N_EXPERT_TRAJS)

    print(f"Collecting {N_MEDIUM_TRAJS} medium trajectories...")
    medium_trajs = collect_trajectories(medium_model, env, N_MEDIUM_TRAJS)

    print(f"Collecting {N_RANDOM_TRAJS} random trajectories...")
    random_trajs = collect_trajectories(None, env, N_RANDOM_TRAJS, is_random=True)

    # 5. Combine and save the dataset
    all_trajectories = expert_trajs + medium_trajs + random_trajs
    
    print(f"\nTotal trajectories collected: {len(all_trajectories)}")
    print(f"Saving dataset to {OUT_FILENAME}...")
    
    with open(OUT_FILENAME, "wb") as f:
        pickle.dump(all_trajectories, f)
        
    print("Dataset generation complete.")
    env.close()