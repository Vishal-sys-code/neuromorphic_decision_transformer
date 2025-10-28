import argparse
import os
from pathlib import Path
import sys
import gymnasium as gym
import numpy as np
import yaml

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

def generate_random_trajectories(env_name, num_trajectories=1000, max_steps=1000):
    env = gym.make(env_name)
    trajectories = []
    
    for _ in range(num_trajectories):
        states, actions, rewards = [], [], []
        state, _ = env.reset()
        done = False
        timestep = 0
        
        while not done and timestep < max_steps:
            states.append(state)
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            actions.append(action)
            rewards.append(reward)
            
            state = next_state
            done = terminated or truncated
            timestep += 1
            
        # Add final state
        states.append(state)
        
        trajectories.append({
            'states': np.array(states),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'length': len(actions)
        })
    
    env.close()
    return trajectories

def process_trajectories(trajectories, max_timesteps=1000):
    # Find the maximum trajectory length
    max_len = min(max([t['length'] for t in trajectories]), max_timesteps)
    
    # Initialize arrays
    num_trajectories = len(trajectories)
    state_dim = trajectories[0]['states'][0].shape[0]
    act_dim = 1  # For discrete actions
    
    states = np.zeros((num_trajectories, max_len + 1, state_dim))
    actions = np.zeros((num_trajectories, max_len, act_dim))
    returns_to_go = np.zeros((num_trajectories, max_len + 1, 1))
    timesteps = np.zeros((num_trajectories, max_len + 1))
    mask = np.zeros((num_trajectories, max_len + 1))
    
    # Fill arrays
    for i, traj in enumerate(trajectories):
        length = min(traj['length'], max_len)
        
        # States include the final state
        states[i, :length + 1] = traj['states'][:length + 1]
        actions[i, :length] = traj['actions'][:length, None]  # Add dimension for scalar actions
        
        # Calculate returns to go
        returns = np.cumsum(traj['rewards'][:length][::-1])[::-1]
        returns_to_go[i, :length] = returns[:, None]
        
        # Timesteps and mask
        timesteps[i, :length + 1] = np.arange(length + 1)
        mask[i, :length + 1] = 1
    
    # Create metadata
    metadata = {
        'state_dim': state_dim,
        'act_dim': 2,  # CartPole has 2 actions
        'max_timesteps': max_timesteps
    }
    
    return {
        'states': states,
        'actions': actions,
        'returns_to_go': returns_to_go,
        'timesteps': timesteps,
        'mask': mask,
        'metadata': metadata
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, required=True, help='Gym environment name')
    parser.add_argument('--num-trajectories', type=int, default=1000, help='Number of trajectories to collect')
    parser.add_argument('--max-steps', type=int, default=1000, help='Maximum steps per trajectory')
    args = parser.parse_args()
    
    # Create data directory
    data_dir = project_root / 'data' / args.env
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {args.num_trajectories} trajectories for {args.env}...")
    trajectories = generate_random_trajectories(args.env, args.num_trajectories, args.max_steps)
    print("Processing trajectories...")
    dataset = process_trajectories(trajectories)
    
    # Save dataset
    output_path = data_dir / 'dataset.npz'
    np.savez_compressed(str(output_path), **dataset)
    print(f"Dataset saved to {output_path}")

if __name__ == '__main__':
    main()