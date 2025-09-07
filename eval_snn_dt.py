"""
Author: Vishal Pandey
Email: pandeyvishal.mlprof@gmail.com

Offline evaluation script for Spiking Neural Network Decision Transformer (SNN-DT).

This script evaluates a pre-trained SNN-DT model on a specified Gym environment.
It measures performance metrics such as average return and standard deviation,
as well as efficiency metrics like spike counts, latency, and estimated energy consumption.

The evaluation follows the standard Decision Transformer inference methodology, where actions
are generated autoregressively based on a history of states, actions, and a target return.
"""

import argparse
import time
import torch
import gym
import numpy as np

from src.models.snn_dt_gpt2_attention import SNNDecisionTransformer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(checkpoint_path, env, device=DEVICE):
    """
    Load a pre-trained SNN-DT model from a checkpoint.

    Args:
        checkpoint_path (str): Path to the model checkpoint.
        env (gym.Env): The Gym environment.
        device (torch.device): The device to load the model on.

    Returns:
        SNNDecisionTransformer: The loaded model.
    """
    # Model parameters should be saved in the checkpoint, but we can infer them
    # from the environment if they are not.
    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n # Assuming discrete action space for now

    # These are typical hyperparameters for Decision Transformer.
    # They should ideally be saved in the checkpoint file.
    # If not, we use some reasonable defaults here.
    max_ep_len = 1000
    hidden_size = 128
    n_layer = 3
    n_head = 1
    n_inner = 4 * hidden_size
    activation_function = "relu"
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    model = SNNDecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        max_length=20, # K in the paper
        max_ep_len=max_ep_len,
        hidden_size=hidden_size,
        n_layer=n_layer,
        n_head=n_head,
        n_inner=n_inner,
        activation_function=activation_function,
        resid_pdrop=resid_pdrop,
        attn_pdrop=attn_pdrop,
        action_tanh=False, # For discrete action space
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded model from {checkpoint_path}")
    return model

def evaluate(
    model,
    env,
    target_return,
    episodes=50,
    max_ep_len=1000,
    per_spike_energy_J=4.6e-12, # Energy per spike in Joules (e.g., 4.6 pJ for 45nm process)
    context_len=20,
):
    """
    Evaluate the SNN-DT model.

    Args:
        model (SNNDecisionTransformer): The model to evaluate.
        env (gym.Env): The Gym environment.
        target_return (float): The target return for the Decision Transformer.
        episodes (int): The number of episodes to evaluate for.
        max_ep_len (int): The maximum length of an episode.
        per_spike_energy_J (float): The energy consumption per spike in Joules.
        context_len (int): The context length (K) for the Decision Transformer.

    Returns:
        dict: A dictionary containing the evaluation results.
    """
    returns, all_spikes, all_latencies = [], [], []

    for ep in range(episodes):
        state = env.reset()
        done = False
        ep_return = 0
        ep_spikes = 0
        ep_steps = 0
        ep_latency = 0

        # The history of states, actions, rewards, and returns-to-go
        states = torch.from_numpy(state).reshape(1, 1, model.state_dim).to(device=DEVICE, dtype=torch.float32)
        actions = torch.zeros((1, 0, model.act_dim), device=DEVICE, dtype=torch.float32)
        rewards = torch.zeros(1, 0, device=DEVICE, dtype=torch.float32)
        timesteps = torch.tensor([0], device=DEVICE, dtype=torch.long).reshape(1, 1)
        
        # The target return for the current episode
        target_return_tensor = torch.tensor([target_return], device=DEVICE, dtype=torch.float32).reshape(1, 1, 1)
        
        sim_states = []

        while not done and ep_steps < max_ep_len:
            
            # Reset spike count before each forward pass
            model.reset_spike_count()

            # Autoregressively generate the next action
            start_time = time.time()
            with torch.no_grad():
                action = model.get_action(
                    states,
                    actions,
                    rewards,
                    target_return_tensor,
                    timesteps,
                )
            latency = (time.time() - start_time) * 1000  # in ms
            
            # Get the spike count for the last forward pass
            spikes = model.get_spike_count()

            action_np = action.detach().cpu().numpy()
            state, reward, done, _ = env.step(np.argmax(action_np))


            # Update history
            states = torch.cat([states, torch.from_numpy(state).reshape(1, 1, model.state_dim).to(device=DEVICE, dtype=torch.float32)], dim=1)
            actions = torch.cat([actions, action.reshape(1, 1, model.act_dim).to(device=DEVICE, dtype=torch.float32)], dim=1)
            rewards = torch.cat([rewards, torch.tensor([reward], device=DEVICE).reshape(1, 1)], dim=1)
            timesteps = torch.cat([timesteps, torch.tensor([ep_steps + 1], device=DEVICE, dtype=torch.long).reshape(1, 1)], dim=1)
            target_return_tensor = torch.cat([target_return_tensor, torch.tensor([target_return - ep_return], device=DEVICE, dtype=torch.float32).reshape(1, 1, 1)], dim=1)

            # Truncate history to context length
            states = states[:, -context_len:]
            actions = actions[:, -context_len:]
            rewards = rewards[:, -context_len:]
            timesteps = timesteps[:, -context_len:]
            target_return_tensor = target_return_tensor[:, -context_len:]


            ep_return += reward
            ep_spikes += spikes
            ep_steps += 1
            ep_latency += latency

        returns.append(ep_return)
        all_spikes.append(ep_spikes / ep_steps if ep_steps > 0 else 0)
        all_latencies.append(ep_latency / ep_steps if ep_steps > 0 else 0)
        
        print(f"Episode {ep+1}/{episodes}: Return={ep_return:.2f}, Avg Spikes={(ep_spikes/ep_steps if ep_steps > 0 else 0):.2f}, Avg Latency={(ep_latency/ep_steps if ep_steps > 0 else 0):.2f}ms")


    env.close()

    # Aggregate stats
    result = {
        "avg_return": np.mean(returns),
        "std_return": np.std(returns),
        "avg_spikes_per_step": np.mean(all_spikes),
        "avg_latency_ms": np.mean(all_latencies),
        "estimated_energy_mJ_per_step": (np.mean(all_spikes) * per_spike_energy_J) * 1000,
    }
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Gym environment name")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes for evaluation")
    parser.add_argument("--target_return", type=float, required=True, help="Target return for the Decision Transformer")
    parser.add_argument("--context_len", type=int, default=20, help="Context length (K) for the Decision Transformer")
    parser.add_argument("--max_ep_len", type=int, default=1000, help="Maximum episode length")
    parser.add_argument("--per_spike_energy", type=float, default=4.6e-12, help="Energy per spike in Joules")
    
    args = parser.parse_args()

    env = gym.make(args.env)
    model = load_model(args.checkpoint_path, env, device=DEVICE)
    
    results = evaluate(
        model,
        env,
        args.target_return,
        episodes=args.episodes,
        max_ep_len=args.max_ep_len,
        per_spike_energy_J=args.per_spike_energy,
        context_len=args.context_len,
    )

    print("" + "="*30)
    print("SNN-DT Evaluation Results")
    print("="*30)
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
    print("="*30)