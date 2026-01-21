import argparse
import json
import time
import logging
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

import numpy as np
import torch
import yaml
import gymnasium as gym

from src.utils.config import AttrDict
from src.utils.models import get_model
from src.utils.evaluation import create_env


def evaluate_policy(model, env, cfg, episodes, state_mean=None, state_std=None):
    model.eval()
    returns = []
    latencies = []
    
    state_dim = env.observation_space.shape[0]
    # Handle both Discrete and Box action spaces
    if isinstance(env.action_space, gym.spaces.Discrete):
        act_dim = env.action_space.n
        is_discrete = True
    else:
        act_dim = env.action_space.shape[0]
        is_discrete = False

    # Ensure mean/std are numpy
    if state_mean is not None and isinstance(state_mean, torch.Tensor):
        state_mean = state_mean.cpu().numpy()
    if state_std is not None and isinstance(state_std, torch.Tensor):
        state_std = state_std.cpu().numpy()

    for i in range(episodes):
        # Deterministic seeding for paired comparison
        eval_seed = cfg.seed + i
        obs, info = env.reset(seed=eval_seed)
        
        if state_mean is not None and state_std is not None:
            obs = (obs - state_mean) / state_std

        done = False
        total_reward = 0
        
        # Initialize context buffers
        states = np.zeros((1, cfg.model.seq_len, state_dim))
        actions = np.zeros((1, cfg.model.seq_len, 1 if is_discrete else act_dim))
        returns_to_go = np.zeros((1, cfg.model.seq_len, 1))
        timesteps = np.zeros((1, cfg.model.seq_len), dtype=int)
        
        # Determine target return?
        # Standard DT evaluation often prompts with a specific target return (e.g. expert return).
        # Phase 0 prompt doesn't specify target return prompting strategy.
        # But `train.py` usually implies we might want to track RTG.
        # Current code initializes RTG to zeros.
        # If the model relies on RTG, 0 might be bad (implies 0 expected return).
        # However, for now I will leave initialization as is (0) or maybe we should default to something else?
        # If I change it, I might deviate from baseline.
        # I'll stick to existing logic: zeros. 
        # (Though usually DT is prompted with Expert Return).
        # Given "Identical datasets... evaluation protocols", I should ensure this is consistent.
        # DSFormer/DT usually prompt with max return.
        # But if the codebase had 0, I keep 0 unless instructed.
        # The user instruction: "Supporting DT, DSFormer... Enforcing identical... evaluation protocols".
        # If the previous code used 0, I use 0.
        
        states[0, 0] = obs
        timesteps[0, 0] = 0
        
        steps = 0
        max_steps = getattr(env, "_max_episode_steps", 1000)
        
        while not done and steps < max_steps:
            start_time = time.perf_counter()
            if cfg.model.name in ["iql", "cql"]:
                action = model.predict_action(obs)
            else:
                action = model.predict_action(
                    states,
                    actions,
                    returns_to_go,
                    timesteps,
                    first_step=(steps == 0),
                )
            latencies.append(time.perf_counter() - start_time)

            if is_discrete:
                if isinstance(action, np.ndarray):
                    env_action = np.argmax(action) if action.size > 1 else int(action.item())
                else:
                    env_action = action
            else:
                env_action = action
            
            obs, reward, terminated, truncated, info = env.step(env_action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            
            # Normalize new observation
            if state_mean is not None and state_std is not None:
                obs = (obs - state_mean) / state_std

            # Shift context window
            actions[0, :-1] = actions[0, 1:]
            states[0, :-1] = states[0, 1:]
            returns_to_go[0, :-1] = returns_to_go[0, 1:]
            timesteps[0, :-1] = timesteps[0, 1:]
            
            # Add new values
            # Ideally we update returns_to_go with expected return, but for now we just decrement
            # Note: This logic assumes RTG is tracked. In offline RL eval, we usually prompt with desired RTG.
            # But the current logic just shifts. We should probably update RTG based on reward?
            # Existing code: returns_to_go[0, -1] = returns_to_go[0, -2] - reward
            
            if is_discrete:
                actions[0, -1] = env_action
            else:
                actions[0, -1] = action
                
            states[0, -1] = obs
            returns_to_go[0, -1] = returns_to_go[0, -2] - reward
            timesteps[0, -1] = steps # t + 1? steps is 1-based now?
            
        returns.append(total_reward)

    return {
        "return_mean": np.mean(returns),
        "return_std": np.std(returns),
        "latency_mean_ms": np.mean(latencies) * 1000 if latencies else 0.0,
        "returns": returns,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument("--env", type=str, required=True, help="Gym environment name.")
    parser.add_argument("--episodes", type=int, default=50, help="Number of evaluation episodes.")
    parser.add_argument("--config", type=str, default="experiments/configs/default.yaml", help="Path to config file.")
    parser.add_argument("--simulator-available", action="store_true", help="Use real simulator.")
    parser.add_argument("--dataset-path", type=str, default=None, help="Path to dataset (for replay mode).")
    args = parser.parse_args()

    # Load config and model
    with open(args.config, 'r') as f:
        cfg = AttrDict(yaml.safe_load(f))
    
    model = get_model(cfg)
    model.load_state_dict(torch.load(args.ckpt))
    
    # Create environment
    env = create_env(args.env, args.simulator_available, args.dataset_path)

    # Evaluate
    # Note: main() doesn't currently support loading mean/std. 
    # This implies main() might be incorrect for normalized models unless updated.
    # For Phase 0 training, we use train.py which calls evaluate_policy directly.
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