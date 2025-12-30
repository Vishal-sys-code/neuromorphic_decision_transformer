import gymnasium as gym
import numpy as np
import pickle
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# --- Configuration ---
ENVS = ["CartPole-v1", "Acrobot-v1", "Pendulum-v1", "MountainCar-v0"]
NUM_TRAJECTORIES_PER_POLICY = 100
SAVE_DIR = Path(__file__).parent.parent / "datasets/raw"

# --- Helper Functions ---
def train_policy(env_name, training_timesteps):
    """Trains a PPO policy for a given number of timesteps."""
    env = DummyVecEnv([lambda: gym.make(env_name)])
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=training_timesteps)
    return model

def collect_trajectories(policy, env_name, num_trajectories):
    """Collects trajectories using a given policy."""
    trajectories = []
    env = gym.make(env_name)
    for _ in range(num_trajectories):
        obs, _ = env.reset()
        done = False
        observations, actions, rewards = [], [], []
        while not done:
            if policy == 'random':
                action = env.action_space.sample()
            else:
                action, _ = policy.predict(obs, deterministic=True)
            
            observations.append(obs)
            actions.append(action)
            
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            rewards.append(reward)
            
        trajectories.append({
            "observations": np.array(observations),
            "actions": np.array(actions),
            "rewards": np.array(rewards),
            "episode_return": np.sum(rewards),
        })
    env.close()
    return trajectories

# --- Main Script ---
def main():
    for env_name in ENVS:
        print(f"--- Generating trajectories for {env_name} ---")
        env_save_dir = SAVE_DIR / env_name
        env_save_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Random Policy
        print("Collecting random trajectories...")
        random_trajs = collect_trajectories('random', env_name, NUM_TRAJECTORIES_PER_POLICY)
        
        # 2. Medium Policy
        print("Training medium policy...")
        medium_timesteps = 10000 if env_name != "Pendulum-v1" else 50000
        medium_policy = train_policy(env_name, medium_timesteps)
        print("Collecting medium trajectories...")
        medium_trajs = collect_trajectories(medium_policy, env_name, NUM_TRAJECTORIES_PER_POLICY)
        
        # 3. Expert Policy
        print("Training expert policy...")
        expert_timesteps = 50000 if env_name != "Pendulum-v1" else 200000
        expert_policy = train_policy(env_name, expert_timesteps)
        print("Collecting expert trajectories...")
        expert_trajs = collect_trajectories(expert_policy, env_name, NUM_TRAJECTORIES_PER_POLICY)
        
        # Save all trajectories
        all_trajs = random_trajs + medium_trajs + expert_trajs
        for i, traj in enumerate(all_trajs):
            with open(env_save_dir / f"traj_{i:04d}.pkl", "wb") as f:
                pickle.dump(traj, f)
                
        print(f"--- Saved {len(all_trajs)} raw trajectories for {env_name} ---")

if __name__ == "__main__":
    main()