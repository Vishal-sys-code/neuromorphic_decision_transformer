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

from src.config import DEVICE, SEED, dt_config, max_length
from src.train_snn_dt import train_offline_dt as train_snn_dt
from src.train_dsf_dt import train_offline_dsf as train_dsf
from src.models.snn_dt_patch import SNNDecisionTransformer as SNNDT
from src.models.dsf_dt import DecisionSpikeFormer
from src.utils.helpers import get_latest_checkpoint

def evaluate_model(model_class, model_config, env_name, model_name, seed):
    """
    Evaluate a trained model.
    """
    print(f"Evaluating model {model_name} on {env_name} with seed {seed}")

    env = gym.make(env_name)
    # For DecisionSpikeFormer, add env-specific dims to config
    if model_name == 'dsf-dt':
        model_config.update({
            "state_dim": env.observation_space.shape[0],
            "act_dim": env.action_space.n if isinstance(env.action_space, gym.spaces.Discrete) else env.action_space.shape[0],
            "max_length": max_length
        })
    
    # Add dummy training steps and warmup_ratio if not present, for model constructors that need them
    if 'num_training_steps' not in model_config:
        model_config['num_training_steps'] = 10000 # dummy value for eval
    if 'warmup_ratio' not in model_config:
        model_config['warmup_ratio'] = 0.1 # dummy value for eval

    # Filter model_config to only include parameters expected by the model's constructor
    import inspect
    sig = inspect.signature(model_class.__init__)
    allowed_args = set(sig.parameters.keys())
    
    filtered_config = {k: v for k, v in model_config.items() if k in allowed_args}

    model = model_class(**filtered_config).to(DEVICE)

    # For models that need state_dim and act_dim as attributes post-initialization
    if not hasattr(model, 'state_dim'):
        model.state_dim = env.observation_space.shape[0]
    if not hasattr(model, 'act_dim'):
        model.act_dim = env.action_space.n if isinstance(env.action_space, gym.spaces.Discrete) else env.action_space.shape[0]
    
    checkpoint_path = get_latest_checkpoint(env_name, model_name)
    if not checkpoint_path:
        print(f"No checkpoint found for {model_name} on {env_name}. Skipping evaluation.")
        # As a fallback for testing, let's train if no checkpoint is found
        print("Training the model as no checkpoint was found...")
        if model_name == 'snn-dt':
            train_snn_dt(env_name)
        else:
            train_dsf(env_name)
        checkpoint_path = get_latest_checkpoint(env_name, model_name)
        if not checkpoint_path:
            print("Still no checkpoint after training. Aborting evaluation.")
            return None


    try:
        state_dict = torch.load(checkpoint_path, map_location=DEVICE)
        if 'model_state' in state_dict:
            model.load_state_dict(state_dict['model_state'])
        else:
            model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading checkpoint for {model_name} on {env_name}: {e}")
        return None

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

        actions = torch.zeros((1, 0, model.act_dim), device=DEVICE, dtype=torch.float32)
        returns_to_go = torch.zeros((1, 0, 1), device=DEVICE, dtype=torch.float32)
        timesteps = torch.zeros((1, 0), dtype=torch.long, device=DEVICE)


        while not done:
            start_time = time.time()
            
            current_state_tensor = torch.from_numpy(state).to(DEVICE).reshape(1, 1, model.state_dim).float()
            
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["snn-dt", "dsf-dt"], help="Model to train and evaluate.")
    parser.add_argument("--env", type=str, required=True, help="Gym environment name.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed.")
    args = parser.parse_args()

    model_class, train_fn, model_name_str = (SNNDT, train_snn_dt, "snn-dt") if args.model == "snn-dt" else (DecisionSpikeFormer, train_dsf, "dsf-dt")
    
    print(f"--- Evaluating {model_name_str} on {args.env} with seed {args.seed} ---")
    eval_results = evaluate_model(model_class, dt_config.copy(), args.env, model_name_str, args.seed)

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
