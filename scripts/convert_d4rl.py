import os
import argparse
import h5py
import numpy as np
import json
import pandas as pd
from tqdm import tqdm

def get_keys(h5file):
    keys = []
    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)
    h5file.visititems(visitor)
    return keys

def load_dataset_from_h5(h5_path):
    dataset = {}
    with h5py.File(h5_path, 'r') as f:
        # Standard D4RL keys
        dataset['observations'] = f['observations'][:]
        dataset['actions'] = f['actions'][:]
        dataset['rewards'] = f['rewards'][:]
        dataset['terminals'] = f['terminals'][:]
        
        if 'timeouts' in f:
            dataset['timeouts'] = f['timeouts'][:]
        else:
            dataset['timeouts'] = np.zeros_like(dataset['terminals'])
            
    return dataset

def segment_trajectories(dataset):
    # Split into individual trajectories
    trajectories = []
    
    observations = dataset['observations']
    actions = dataset['actions']
    rewards = dataset['rewards']
    terminals = dataset['terminals']
    timeouts = dataset['timeouts']
    
    N = observations.shape[0]
    start = 0
    
    for i in range(N):
        done_bool = bool(terminals[i])
        final_timestep = bool(timeouts[i])
        
        if done_bool or final_timestep:
            end = i + 1
            traj = {
                'states': observations[start:end],
                'actions': actions[start:end],
                'rewards': rewards[start:end],
                'dones': terminals[start:end]
            }
            trajectories.append(traj)
            start = end
            
    # Handle last trajectory if not terminated
    if start < N:
        traj = {
            'states': observations[start:],
            'actions': actions[start:],
            'rewards': rewards[start:],
            'dones': terminals[start:]
        }
        trajectories.append(traj)
        
    return trajectories

def compute_returns_to_go(trajectories):
    for traj in trajectories:
        rewards = traj['rewards']
        rtg = np.zeros_like(rewards)
        running_return = 0
        for t in reversed(range(len(rewards))):
            running_return += rewards[t]
            rtg[t] = running_return
        traj['returns_to_go'] = rtg
    return trajectories

def clip_and_pad_trajectories(trajectories, clip_len, state_dim, act_dim):
    states, actions, returns_to_go, timesteps, masks, rewards_list, dones_list = [], [], [], [], [], [], []
    
    for traj in trajectories:
        traj_len = len(traj['states'])
        num_clips = (traj_len + clip_len - 1) // clip_len
        
        for i in range(num_clips):
            start = i * clip_len
            end = min((i + 1) * clip_len, traj_len)
            actual_len = end - start
            
            # Init buffers with zeros (padding)
            clip_states = np.zeros((clip_len, state_dim), dtype=np.float32)
            clip_actions = np.zeros((clip_len, act_dim), dtype=np.float32)
            clip_rtg = np.zeros((clip_len, 1), dtype=np.float32)
            clip_timesteps = np.zeros((clip_len), dtype=np.int64)
            clip_mask = np.zeros((clip_len), dtype=np.float32)
            clip_rewards = np.zeros((clip_len, 1), dtype=np.float32)
            clip_dones = np.zeros((clip_len, 1), dtype=np.float32)
            
            # Fill data
            clip_states[:actual_len] = traj['states'][start:end]
            clip_actions[:actual_len] = traj['actions'][start:end].reshape(-1, act_dim)
            clip_rtg[:actual_len] = traj['returns_to_go'][start:end].reshape(-1, 1)
            clip_timesteps[:actual_len] = np.arange(start, end)
            clip_mask[:actual_len] = 1.0
            clip_rewards[:actual_len] = traj['rewards'][start:end].reshape(-1, 1)
            clip_dones[:actual_len] = traj['dones'][start:end].reshape(-1, 1)
            
            states.append(clip_states)
            actions.append(clip_actions)
            returns_to_go.append(clip_rtg)
            timesteps.append(clip_timesteps)
            masks.append(clip_mask)
            rewards_list.append(clip_rewards)
            dones_list.append(clip_dones)
            
    return {
        "states": np.array(states),
        "actions": np.array(actions),
        "returns_to_go": np.array(returns_to_go),
        "timesteps": np.array(timesteps),
        "mask": np.array(masks),
        "rewards": np.array(rewards_list),
        "dones": np.array(dones_list)
    }

def main(args):
    raw_dir = args.raw_dir
    out_base_dir = args.out_dir
    
    os.makedirs(out_base_dir, exist_ok=True)
    manifest_path = os.path.join(out_base_dir, "manifest.csv")
    manifest = []
    
    # List all hdf5 files
    files = [f for f in os.listdir(raw_dir) if f.endswith('.hdf5')]
    
    for filename in tqdm(files, desc="Converting datasets"):
        # Infer env name from filename logic in download_d4rl.py
        # filename: {env}_{dataset_suffix}-v2.hdf5
        # env_name: {env}-{dataset}-v2
        # e.g. hopper_medium-v2.hdf5 -> hopper-medium-v2
        
        # Reverse mapping:
        name_stem = filename.replace('.hdf5', '')
        # Special case handling if needed, but generic replace should work if strict adherence
        # hopper_medium_expert-v2 -> hopper-medium-expert-v2
        # But wait, hopper_medium-v2 -> hopper-medium-v2 works.
        # But hopper_medium_expert-v2 needs underscores to become hyphens.
        env_name = name_stem.replace('_', '-')
        # Fix 'v2' back to '-v2' if it got messed up? No, 'v2' doesn't have underscore.
        
        # Wait, filename: `hopper_medium_expert-v2.hdf5`.
        # `replace('_', '-')` -> `hopper-medium-expert-v2`. Correct.
        
        h5_path = os.path.join(raw_dir, filename)
        out_dir = os.path.join(out_base_dir, env_name)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "dataset_v1.npz")
        
        if os.path.exists(out_path) and not args.force:
            print(f"Skipping {env_name}, already converted.")
            continue
            
        # Load and Process
        try:
            raw_data = load_dataset_from_h5(h5_path)
            trajectories = segment_trajectories(raw_data)
            trajectories = compute_returns_to_go(trajectories)
            
            state_dim = raw_data['observations'].shape[1]
            act_dim = raw_data['actions'].shape[1]
            max_timesteps = 1000 # Standard MuJoCo limit
            
            processed_data = clip_and_pad_trajectories(trajectories, args.clip_len, state_dim, act_dim)
            
            metadata = {
                "env": env_name,
                "state_dim": int(state_dim),
                "act_dim": int(act_dim),
                "max_timesteps": int(max_timesteps),
                "clip_len": int(args.clip_len)
            }
            
            np.savez_compressed(
                out_path,
                **processed_data,
                metadata=json.dumps(metadata)
            )
            
            manifest.append({
                "env": env_name,
                "raw_file": filename,
                "output_file": out_path,
                "status": "success",
                "trajectories": len(trajectories),
                "clips": processed_data['states'].shape[0]
            })
            
        except Exception as e:
            print(f"Failed to convert {filename}: {e}")
            manifest.append({
                "env": env_name,
                "raw_file": filename,
                "output_file": "",
                "status": f"failed: {str(e)}",
                "trajectories": 0,
                "clips": 0
            })
            
    # Save manifest
    if os.path.exists(manifest_path):
        old_df = pd.read_csv(manifest_path)
        new_df = pd.DataFrame(manifest)
        combined_df = pd.concat([old_df, new_df]).drop_duplicates(subset=['env'], keep='last')
        combined_df.to_csv(manifest_path, index=False)
    else:
        pd.DataFrame(manifest).to_csv(manifest_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=str, default="data/d4rl_raw", help="Directory containing HDF5 files.")
    parser.add_argument("--out-dir", type=str, default="data/d4rl", help="Output directory for NPZ files.")
    parser.add_argument("--clip-len", type=int, default=20, help="Sequence length for clipping.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files.")
    args = parser.parse_args()
    
    main(args)
