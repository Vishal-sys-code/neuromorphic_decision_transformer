
import numpy as np
import shutil
import json
from pathlib import Path
import datetime

# --- Configuration ---
SOURCE_DIR = Path("d:/Github/neuromorphic_decision_transformer/ablation_studies/datasets/processed")
DEST_DIR = Path("d:/Github/neuromorphic_decision_transformer/ablation_studies/datasets/frozen_v1")
ENVS = ["CartPole-v1", "Acrobot-v1", "Pendulum-v1", "MountainCar-v0"]

def freeze_dataset(env_name):
    print(f"--- Freezing {env_name} ---")
    
    # Paths
    src_env_dir = SOURCE_DIR / env_name
    dest_env_dir = DEST_DIR / env_name
    
    if not src_env_dir.exists():
        print(f"Skipping {env_name}: Source directory not found.")
        return

    dest_env_dir.mkdir(parents=True, exist_ok=True)
    
    # Datasets to freeze
    datasets = {
        "stratified_dataset.npz": "v1-stratified",
        "random_heavy_dataset.npz": "v1-random-heavy"
    }
    
    normalization_info = {}

    for filename, version_id in datasets.items():
        src_file = src_env_dir / filename
        dest_file = dest_env_dir / filename
        
        if not src_file.exists():
            print(f"Warning: {filename} not found in {src_env_dir}")
            continue
            
        # 1. Load data to compute stats
        print(f"Loading {filename}...")
        data = np.load(src_file)
        
        # Copy data to memory to modify/save
        data_dict = dict(data)
        
        # Compute Stats if not present (simple calculation over all states)
        states = data_dict['states'] # Shape: (N, L, D)
        # Flatten for mean/std calculation
        flat_states = states.reshape(-1, states.shape[-1])
        # Mask handling (only compute on valid timesteps)
        mask = data_dict['mask'].flatten()
        valid_flat_states = flat_states[mask > 0.5]
        
        state_mean = np.mean(valid_flat_states, axis=0)
        state_std = np.std(valid_flat_states, axis=0) + 1e-6 # Avoid div by zero
        
        # Add to dictionary
        data_dict['state_mean'] = state_mean
        data_dict['state_std'] = state_std
        data_dict['version_id'] = version_id
        
        # 2. Save modified NPZ to frozen dir
        print(f"Saving frozen copy to {dest_file}...")
        np.savez(dest_file, **data_dict)
        
        # Store for metadata
        normalization_info[version_id] = {
            "state_mean": state_mean.tolist(),
            "state_std": state_std.tolist()
        }

    # 3. Save Metadata
    metadata = {
        "frozen_at": datetime.datetime.now().isoformat(),
        "source_dir": str(src_env_dir),
        "normalization": normalization_info,
        "datasets": datasets
    }
    
    with open(dest_env_dir / "dataset_info.json", "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Frozen {env_name} successfully.")

def main():
    if DEST_DIR.exists():
         print(f"Warning: Destination directory {DEST_DIR} already exists.")
         # Using existing could be dangerous if we want to be strictly immutable, 
         # but for this script we will allow overwriting to fix issues.
    
    for env in ENVS:
        freeze_dataset(env)
        
    print("\nAll datasets frozen.")

if __name__ == "__main__":
    main()
