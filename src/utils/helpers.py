"""
Author: Vishal Pandey
Email: pandeyvishal.mlprof@gmail.com
"""
import numpy as np
import torch
import os
import glob


def compute_returns_to_go(rewards, gamma=0.99):
    rtg = np.zeros_like(rewards, dtype=float)
    running = 0.0
    for i in reversed(range(len(rewards))):
        running = rewards[i] + gamma * running
        rtg[i] = running
    return rtg


def simple_logger(log_dict, step):
    entries = []
    for k, v in log_dict.items():
        if isinstance(v, float): entries.append(f"{k}={v:.3f}")
        else: entries.append(f"{k}={v}")
    print(f"[Step {step}] " + ", ".join(entries))


def save_checkpoint(model, optimizer, path):
    torch.save({
        'model_state': model.state_dict(),
        'optim_state': optimizer.state_dict(),
    }, path)

def get_latest_checkpoint(env_name, model_name, checkpoint_dir="checkpoints"):
    """
    Retrieves the path of the latest checkpoint file for a given model and environment.

    Args:
        env_name (str): The name of the environment (e.g., 'CartPole-v1').
        model_name (str): The name of the model (e.g., 'snn-dt').
        checkpoint_dir (str): The directory where checkpoints are stored.

    Returns:
        str or None: The path to the latest checkpoint file, or None if no checkpoint is found.
    """
    # Mapping from model_name to the file pattern prefix
    model_pattern_map = {
        "snn-dt": "offline_dt",
        "dsf-dt": "dsf_dt"  # Assuming 'dsf-dt' maps to 'dsf_dt' filenames
    }

    pattern_prefix = model_pattern_map.get(model_name)
    if not pattern_prefix:
        print(f"Unknown model name: {model_name}")
        return None

    # Construct the full search pattern
    pattern = os.path.join(checkpoint_dir, f"{pattern_prefix}_{env_name}_*.pt")
    
    # Find all matching checkpoint files
    checkpoint_files = glob.glob(pattern)

    if not checkpoint_files:
        return None

    # Extract epoch numbers and find the latest file
    try:
        latest_file = max(checkpoint_files, key=lambda f: int(os.path.basename(f).split('_')[-1].split('.')[0]))
    except (ValueError, IndexError):
        # Handle cases where parsing the epoch number fails
        return None
        
    return latest_file
