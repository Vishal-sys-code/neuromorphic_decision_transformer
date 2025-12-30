import numpy as np
import pickle
from pathlib import Path
import gym
import json

# --- Configuration ---
ENVS = ["CartPole-v1", "Acrobot-v1", "Pendulum-v1", "MountainCar-v0"]
RAW_DIR = Path(__file__).parent.parent / "datasets/raw"
PROCESSED_DIR = Path(__file__).parent.parent / "datasets/processed"
CLIP_LEN = 20

# Trajectory indices based on A1_generate_trajectories.py (100 per policy)
# 0-99: Random, 100-199: Medium, 200-299: Expert
NUM_PER_POLICY = 100

def load_trajectories(env_name):
    """Loads all raw pickle files for an environment."""
    env_dir = RAW_DIR / env_name
    trajs = []
    # We expect files named traj_0000.pkl to traj_0299.pkl
    # Load in order to preserve policy types
    pkl_files = sorted(list(env_dir.glob("traj_*.pkl")))
    
    if not pkl_files:
        print(f"[{env_name}] No trajectories found in {env_dir}")
        return []

    for pkl_path in pkl_files:
        with open(pkl_path, "rb") as f:
            trajs.append(pickle.load(f))
    
    return trajs

def process_dataset(trajectories, env_name, clip_len=20):
    """
    Processes a list of trajectories into the Sequence Dataset format.
    Adapted from snn-dt/scripts/make_dataset.py
    """
    
    states, actions, returns_to_go, timesteps, masks = [], [], [], [], []
    
    # Get dimensions from first trajectory
    if not trajectories:
        return None
        
    state_dim = trajectories[0]["observations"].shape[1]
    
    # Handle action dimension
    act_sample = trajectories[0]["actions"]
    if act_sample.ndim == 1:
        action_dim = 1
    else:
        action_dim = act_sample.shape[1]

    for traj in trajectories:
        rewards = traj["rewards"]
        traj_len = len(rewards)
        
        # Calculate Returns-to-Go
        traj_returns = np.zeros(traj_len)
        running_return = 0
        for t in reversed(range(traj_len)):
            running_return += rewards[t]
            traj_returns[t] = running_return
            
        # Create Clips (Padding)
        num_clips = (traj_len + clip_len - 1) // clip_len
        
        for i in range(num_clips):
            start = i * clip_len
            end = (i + 1) * clip_len
            
            actual_len = min(clip_len, traj_len - start)
            
            # Init buffers
            clip_states = np.zeros((clip_len, state_dim))
            clip_actions = np.zeros((clip_len, action_dim))
            clip_rtg = np.zeros((clip_len, 1))
            clip_timesteps = np.zeros(clip_len, dtype=int)
            clip_mask = np.zeros(clip_len)
            
            # Fill buffers
            clip_states[:actual_len] = traj["observations"][start:end]
            
            # Handle action shaping
            current_actions = traj["actions"][start:end]
            if action_dim == 1 and current_actions.ndim == 1:
                current_actions = current_actions.reshape(-1, 1)
            clip_actions[:actual_len] = current_actions
            
            clip_rtg[:actual_len] = traj_returns[start:end].reshape(-1, 1)
            clip_timesteps[:actual_len] = np.arange(start, start + actual_len)
            clip_mask[:actual_len] = 1 # 1 means valid data
            
            states.append(clip_states)
            actions.append(clip_actions)
            returns_to_go.append(clip_rtg)
            timesteps.append(clip_timesteps)
            masks.append(clip_mask)
            
    return {
        "states": np.array(states, dtype=np.float32),
        "actions": np.array(actions, dtype=np.float32),
        "returns_to_go": np.array(returns_to_go, dtype=np.float32),
        "timesteps": np.array(timesteps, dtype=np.int32),
        "mask": np.array(masks, dtype=np.float32),
    }

def save_npz(data, env_name, filename):
    out_dir = PROCESSED_DIR / env_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    
    # Get lightweight env metadata
    # Note: We avoid creating the env to speed up if possible, but we need max_steps
    # Using defaults/heuristics or minimal instantiation
    try:
        env = gym.make(env_name)
        max_timesteps = int(env._max_episode_steps) if hasattr(env, '_max_episode_steps') else 1000
        env.close()
    except:
        max_timesteps = 1000
        
    metadata = {
        "env": env_name,
        "clip_len": CLIP_LEN,
        "max_timesteps": max_timesteps
    }
    
    np.savez(
        out_path,
        states=data["states"],
        actions=data["actions"],
        returns_to_go=data["returns_to_go"],
        timesteps=data["timesteps"],
        mask=data["mask"],
        metadata=json.dumps(metadata)
    )
    print(f"Saved {out_path} with {len(data['states'])} clips.")

def main():
    for env_name in ENVS:
        print(f"--- Processing {env_name} ---")
        
        all_trajs = load_trajectories(env_name)
        if not all_trajs:
            continue
            
        print(f"Loaded {len(all_trajs)} trajectories.")
        
        # --- Stratified Dataset (Balanced) ---
        # Uses all trajectories (Random: 100, Medium: 100, Expert: 100)
        stratified_data = process_dataset(all_trajs, env_name, CLIP_LEN)
        if stratified_data:
            save_npz(stratified_data, env_name, "stratified_dataset.npz")
            
        # --- Random-Heavy Dataset ---
        # Uses Random (100) + subset of Medium (10) + subset of Expert (10)
        # Random are 0-99
        # Medium are 100-199
        # Expert are 200-299
        
        random_trajs = all_trajs[:100]
        medium_subset = all_trajs[100:110]
        expert_subset = all_trajs[200:210]
        
        heavy_trajs = random_trajs + medium_subset + expert_subset
        
        heavy_data = process_dataset(heavy_trajs, env_name, CLIP_LEN)
        if heavy_data:
            save_npz(heavy_data, env_name, "random_heavy_dataset.npz")

    print("\nProcessing Complete.")

if __name__ == "__main__":
    main()