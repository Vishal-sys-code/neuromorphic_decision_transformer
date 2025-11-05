import argparse
import os
from pathlib import Path
import sys
import numpy as np
import yaml

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

def generate_random_trajectories(env_name, num_trajectories=1000, max_steps=1000):
    import gymnasium as gym
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
    max_len = min(max([t['length'] for t in trajectories if t['length'] > 0], default=0), max_timesteps)
    
    if not trajectories or max_len == 0:
        # Handle case with no trajectories or all empty trajectories
        return {
            'states': np.array([]), 'actions': np.array([]), 'returns_to_go': np.array([]),
            'timesteps': np.array([]), 'mask': np.array([]),
            'metadata': {'state_dim': 0, 'act_dim': 0, 'max_timesteps': max_timesteps}
        }

    # Initialize arrays
    num_trajectories = len(trajectories)
    state_dim = trajectories[0]['states'][0].shape[0]
    
    # Determine action dimension and type from data
    first_traj_actions = trajectories[0]['actions']
    is_discrete = first_traj_actions.ndim == 1

    if is_discrete:
        all_actions = np.concatenate([t['actions'] for t in trajectories if t['length'] > 0])
        act_dim = int(all_actions.max()) + 1 if all_actions.size > 0 else 1
        action_shape = 1
    else:
        act_dim = first_traj_actions.shape[1]
        action_shape = act_dim

    states = np.zeros((num_trajectories, max_len + 1, state_dim))
    actions = np.zeros((num_trajectories, max_len, action_shape))
    returns_to_go = np.zeros((num_trajectories, max_len + 1, 1))
    timesteps = np.zeros((num_trajectories, max_len + 1))
    mask = np.zeros((num_trajectories, max_len + 1))
    
    # Fill arrays
    for i, traj in enumerate(trajectories):
        length = min(traj['length'], max_len)
        if length == 0:
            continue
        
        # States include the final state
        states[i, :length + 1] = traj['states'][:length + 1]
        
        if is_discrete:
            actions[i, :length] = traj['actions'][:length, None]
        else:
            actions[i, :length] = traj['actions'][:length]
        
        # Calculate returns to go
        returns = np.cumsum(traj['rewards'][:length][::-1])[::-1]
        returns_to_go[i, :length] = returns[:, None]
        
        # Timesteps and mask
        timesteps[i, :length + 1] = np.arange(length + 1)
        mask[i, :length + 1] = 1
    
    # Create metadata
    metadata = {
        'state_dim': state_dim,
        'act_dim': act_dim,
        'max_timesteps': max_timesteps,
        'is_discrete': is_discrete
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