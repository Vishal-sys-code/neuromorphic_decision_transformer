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

import gym

def evaluate_model(model, env_name, max_episodes, max_length, state_dim, act_dim, device, gamma):
    env = gym.make(env_name)
    returns = []

    for _ in range(max_episodes):
        obs = env.reset()[0]
        states = []
        actions = []
        rewards = []
        returns_to_go = []
        timesteps = []
        
        episode_return = 0
        t = 0
        done = False

        while not done:
            states.append(obs)
            timesteps.append(t)

            # Prepare inputs for the model
            current_states = torch.tensor(states, dtype=torch.float32).reshape(1, -1, state_dim).to(device)
            current_actions = torch.tensor(actions, dtype=torch.long).reshape(1, -1, 1).to(device)
            current_returns_to_go = torch.tensor(returns_to_go, dtype=torch.float32).reshape(1, -1, 1).to(device)
            current_timesteps = torch.tensor(timesteps, dtype=torch.long).reshape(1, -1).to(device)

            # Get action from the model
            action_preds = model.get_action(
                current_states,
                current_actions,
                current_returns_to_go,
                current_timesteps,
            )
            action = action_preds.argmax(dim=-1).item()

            # Step the environment
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            actions.append(action)
            rewards.append(reward)
            episode_return += reward

            # Update returns_to_go (simple approach for evaluation)
            returns_to_go = [episode_return] + [0] * (len(states) - 1) # This is a simplified RTG for evaluation

            t += 1
            if t >= max_length: # Limit episode length for evaluation if needed
                break
        returns.append(episode_return)
    env.close()
    return np.mean(returns)

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
        "dsf-dt": "offline_dsf"  # Corrected to match the actual filename prefix
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
