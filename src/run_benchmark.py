import os
import sys
import argparse
import time
import pandas as pd
import torch
import gym
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.snn_dt_patch import SNNDecisionTransformer as SNNDT
from src.models.dsf_dt import DecisionSpikeFormer
from src.utils.helpers import get_latest_checkpoint

def evaluate_model(model, env_name, model_name, seed, device):
    """
    Evaluate a trained model.
    """
    print(f"Evaluating model {model_name} on {env_name} with seed {seed}")

    env = gym.make(env_name)
    model.eval()
    
    num_eval_episodes = 10
    episode_returns = []
    inference_latencies = []
    total_spikes = 0

    for _ in range(num_eval_episodes):
        state, _ = env.reset()
        done = False
        episode_return = 0
        
        if hasattr(model, 'reset_total_spike_count'):
            model.reset_total_spike_count()

        actions = torch.zeros((1, 0, model.act_dim), device=device, dtype=torch.float32)
        returns_to_go = torch.zeros((1, 0, 1), device=device, dtype=torch.float32)
        timesteps = torch.zeros((1, 0), dtype=torch.long, device=device)


        while not done:
            start_time = time.time()
            
            current_state_tensor = torch.from_numpy(state).to(device).reshape(1, 1, model.state_dim).float()
            
            action_tensor = model.get_action(
                current_state_tensor,
                actions,
                returns_to_go,
                timesteps
            )
            
            inference_latencies.append(time.time() - start_time)
            
            if isinstance(action_tensor, torch.Tensor):
                action = action_tensor.detach().cpu().numpy()
            else:
                action = action_tensor # if it's already a numpy array, for example

            if isinstance(env.action_space, gym.spaces.Discrete):
                action_to_step = np.argmax(action)
            else:
                action_to_step = action

            state, reward, terminated, truncated, _ = env.step(action_to_step)
            done = terminated or truncated
            episode_return += reward

        episode_returns.append(episode_return)
        
        if hasattr(model, 'get_total_spike_count'):
            total_spikes += model.get_total_spike_count()

    avg_return = np.mean(episode_returns)
    std_return = np.std(episode_returns)
    avg_latency = np.mean(inference_latencies) * 1000
    avg_spikes = total_spikes / num_eval_episodes if num_eval_episodes > 0 else 0
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    results = {
        "model": model_name,
        "env": env_name,
        "seed": seed,
        "avg_return": avg_return,
        "std_return": std_return,
        "avg_latency_ms": avg_latency,
        "avg_spikes_per_episode": avg_spikes,
        "total_params": total_params,
    }
    
    return results


def main():
    from src.run_experiment import load_config
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["snn-dt", "dsf-dt"], help="Model to train and evaluate.")
    parser.add_argument("--env", type=str, required=True, help="Gym environment name.")
    parser.add_argument("--seed", type=int, help="Random seed.")
    args = parser.parse_args()

    config = load_config(args.env)
    
    seed = args.seed if args.seed is not None else config.SEED
    
    env = gym.make(args.env)
    is_continuous = isinstance(env.action_space, gym.spaces.Box)
    act_dim = env.action_space.shape[0] if is_continuous else env.action_space.n
    state_dim = env.observation_space.shape[0]
    env.close()

    model_class, model_name_str = (SNNDT, "snn-dt") if args.model == "snn-dt" else (DecisionSpikeFormer, "dsf-dt")
    
    dt_conf = config.dt_config.copy()
    dt_conf.update({
        'state_dim': state_dim,
        'act_dim': act_dim,
        'max_length': config.max_length
    })

    model = model_class(**dt_conf).to(config.DEVICE)
    
    checkpoint_path = get_latest_checkpoint(f"checkpoints", f"offline_{model_name_str}_{args.env}")
    if checkpoint_path:
        print(f"Loading model from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path)['model_state'])
    else:
        print("No checkpoint found. Evaluating a randomly initialized model.")

    print(f"--- Evaluating {model_name_str} on {args.env} with seed {seed} ---")
    eval_results = evaluate_model(model, args.env, model_name_str, seed, config.DEVICE)

    if eval_results:
        results_df = pd.DataFrame([eval_results])
        results_file = "benchmark_results.csv"
        if os.path.exists(results_file):
            results_df.to_csv(results_file, mode='a', header=False, index=False)
        else:
            results_df.to_csv(results_file, index=False)
        print(f"Results logged to {results_file}")

if __name__ == "__main__":
    main()