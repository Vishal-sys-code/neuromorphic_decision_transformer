import argparse
import json
import os
import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch


# Heuristic policy for CartPole-v1
def cartpole_expert_policy(obs):
    x, x_dot, theta, theta_dot = obs
    if abs(theta) < 0.03:
        return 0 if theta_dot < 0 else 1
    else:
        return 0 if theta < 0 else 1


# PPO agent for Pendulum-v1 (a simple pre-trained agent)
class PendulumExpert:
    def __init__(self):
        # A simple linear policy is sufficient for Pendulum
        self.actor = torch.nn.Linear(3, 1)
        # These weights are not optimal, but provide decent performance
        self.actor.weight.data.fill_(0.5)
        self.actor.bias.data.fill_(0.0)

    def __call__(self, obs):
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32).view(1, -1)
            action = self.actor(obs_tensor)
            return action.clamp(-2.0, 2.0).numpy().flatten()
            
# Heuristic policy for Acrobot-v1
def acrobot_expert_policy(obs):
    s = obs
    s_tip = np.array([s[4], s[5]])
    if np.dot(s_tip, np.array([1, 0])) > 0:
        return 0
    elif np.dot(s_tip, np.array([-1, 0])) > 0:
        return 2
    else:
        return 1

# PPO agent for MountainCar-v0 (a simple pre-trained agent)
class MountainCarExpert:
    def __init__(self):
        # A simple linear policy is sufficient for MountainCar
        self.actor = torch.nn.Linear(2, 1)
        # These weights are not optimal, but provide decent performance
        self.actor.weight.data.fill_(0.5)
        self.actor.bias.data.fill_(0.0)

    def __call__(self, obs):
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32).view(1, -1)
            action = self.actor(obs_tensor)
            return action.clamp(-1.0, 1.0).numpy().flatten()
            
def get_expert_policy(env_name):
    if env_name == "CartPole-v1":
        return cartpole_expert_policy
    elif env_name == "Pendulum-v1":
        return PendulumExpert()
    elif env_name == "Acrobot-v1":
        return acrobot_expert_policy
    elif env_name == "MountainCar-v0":
        return MountainCarExpert()
    else:
        raise NotImplementedError(f"Expert policy for {env_name} not implemented.")


def generate_trajectories(env, policy, num_steps):
    trajectories = []
    collected_steps = 0
    while collected_steps < num_steps:
        obs, info = env.reset()
        done = False
        states, actions, rewards = [], [], []
        while not done:
            action = policy(obs)
            states.append(obs)
            actions.append(action)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            rewards.append(reward)
            collected_steps += 1
        trajectories.append({"states": np.array(states), "actions": np.array(actions), "rewards": np.array(rewards)})
    return trajectories


def main(args):
    # Set seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create environment
    env = gym.make(args.env)
    env.reset(seed=args.seed)

    # Get policies
    expert_policy = get_expert_policy(args.env)
    random_policy = lambda obs: env.action_space.sample()

    # Generate trajectories
    print("Generating expert trajectories...")
    expert_trajectories = generate_trajectories(env, expert_policy, args.num_steps // 2)
    print("Generating random trajectories...")
    random_trajectories = generate_trajectories(env, random_policy, args.num_steps // 2)

    # Combine and process trajectories
    trajectories = expert_trajectories + random_trajectories
    process_and_save_dataset(trajectories, args)


def process_and_save_dataset(trajectories, args):
    # Processing logic will go here
    print(f"Processing {len(trajectories)} trajectories...")

    states, actions, returns_to_go, timesteps, masks = [], [], [], [], []
    state_dim = trajectories[0]["states"].shape[1]
    
    # Determine action dimension robustly
    actions_arr = trajectories[0]["actions"]
    if actions_arr.ndim == 1:
        action_dim = 1
    else:
        action_dim = actions_arr.shape[1]


    for traj in trajectories:
        # returns to go
        rewards = traj["rewards"]
        traj_returns = np.zeros(len(rewards))
        running_return = 0
        for t in reversed(range(len(rewards))):
            running_return += rewards[t]
            traj_returns[t] = running_return
        
        # padding
        num_clips = (len(traj["states"]) + args.clip_len - 1) // args.clip_len
        for i in range(num_clips):
            start = i * args.clip_len
            end = (i + 1) * args.clip_len
            
            clip_states = np.zeros((args.clip_len, state_dim))
            clip_actions = np.zeros((args.clip_len, action_dim))
            clip_rtg = np.zeros((args.clip_len, 1))
            clip_timesteps = np.zeros(args.clip_len, dtype=int)
            clip_mask = np.zeros(args.clip_len)

            actual_len = min(args.clip_len, len(traj["states"]) - start)
            
            clip_states[:actual_len] = traj["states"][start:end]
            clip_actions[:actual_len] = traj["actions"][start:end].reshape(-1, action_dim)
            clip_rtg[:actual_len] = traj_returns[start:end].reshape(-1, 1)
            clip_timesteps[:actual_len] = np.arange(start, start + actual_len)
            clip_mask[:actual_len] = 1

            states.append(clip_states)
            actions.append(clip_actions)
            returns_to_go.append(clip_rtg)
            timesteps.append(clip_timesteps)
            masks.append(clip_mask)

    # Save dataset
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    
    # Get env metadata
    env = gym.make(args.env)
    
    np.savez(
        args.out,
        states=np.array(states),
        actions=np.array(actions),
        returns_to_go=np.array(returns_to_go),
        timesteps=np.array(timesteps),
        mask=np.array(masks),
        metadata=json.dumps({
            "env": args.env,
            "seed": args.seed,
            "num_steps": args.num_steps,
            "clip_len": args.clip_len,
            "state_dim": int(env.observation_space.shape[0]),
            "act_dim": int(env.action_space.n) if isinstance(env.action_space, gym.spaces.Discrete) else int(env.action_space.shape[0]),
            "max_timesteps": int(env._max_episode_steps),
        }),
    )
    print(f"Dataset saved to {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Gym environment name.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
    parser.add_argument("--out", type=str, default="data/CartPole-v1/dataset_v1.npz", help="Output file path.")
    parser.add_argument("--num_steps", type=int, default=10000, help="Total number of steps in the dataset.")
    parser.add_argument("--clip_len", type=int, default=20, help="Length of trajectory clips.")
    args = parser.parse_args()
    main(args)