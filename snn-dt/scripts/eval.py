import argparse
import json
import time

import gymnasium as gym
import numpy as np
import torch
import yaml

from src.utils.config import AttrDict
from src.utils.models import get_model


def evaluate_policy(model, env, cfg, episodes):
    model.eval()
    returns = []
    latencies = []
    
    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n if isinstance(env.action_space, gym.spaces.Discrete) else env.action_space.shape[0]

    for _ in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        
        states = np.zeros((1, cfg.model.seq_len, state_dim))
        actions = np.zeros((1, cfg.model.seq_len, 1))
        returns_to_go = np.zeros((1, cfg.model.seq_len, 1))
        timesteps = np.zeros((1, cfg.model.seq_len), dtype=int)
        
        states[0, 0] = obs
        timesteps[0, 0] = 0
        
        for t in range(env._max_episode_steps):
            start_time = time.perf_counter()
            if cfg.model.name in ["iql", "cql"]:
                action = model.predict_action(obs)
            else:
                action = model.predict_action(
                    states,
                    actions,
                    returns_to_go,
                    timesteps,
                )
            latencies.append(time.perf_counter() - start_time)

            if isinstance(env.action_space, gym.spaces.Discrete):
                action = np.argmax(action)
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

            # Shift context window
            actions[0, :-1] = actions[0, 1:]
            states[0, :-1] = states[0, 1:]
            returns_to_go[0, :-1] = returns_to_go[0, 1:]
            timesteps[0, :-1] = timesteps[0, 1:]
            
            # Add new values
            actions[0, -1] = action
            states[0, -1] = obs
            returns_to_go[0, -1] = returns_to_go[0, -2] - reward
            timesteps[0, -1] = t + 1
            
            if done:
                break
        returns.append(total_reward)

    return {
        "return_mean": np.mean(returns),
        "return_std": np.std(returns),
        "latency_mean_ms": np.mean(latencies) * 1000,
        "returns": returns,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument("--env", type=str, required=True, help="Gym environment name.")
    parser.add_argument("--episodes", type=int, default=50, help="Number of evaluation episodes.")
    parser.add_argument("--config", type=str, default="experiments/configs/default.yaml", help="Path to config file.")
    args = parser.parse_args()

    # Load config and model
    with open(args.config, 'r') as f:
        cfg = AttrDict(yaml.safe_load(f))
    
    model = get_model(cfg)
    model.load_state_dict(torch.load(args.ckpt))
    
    # Create environment
    env = gym.make(args.env)

    # Evaluate
    results = evaluate_policy(model, env, cfg, args.episodes)
    
    # Save results
    save_path = f"{args.ckpt.rsplit('.', 1)[0]}_eval.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Evaluation results for {args.ckpt}:")
    print(f"  Mean return: {results['return_mean']:.2f} +/- {results['return_std']:.2f}")
    print(f"Results saved to {save_path}")


if __name__ == "__main__":
    main()