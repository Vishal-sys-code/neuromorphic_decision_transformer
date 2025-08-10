"""
Run a full comparison between SNN-DT and DSF-DT on multiple environments.
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import gym
import pickle
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import (
    DEVICE, SEED,
    offline_steps,
    batch_size,
    dt_epochs,
    gamma,
    max_length,
    lr,
    dt_config,
)
from src.utils.trajectory_buffer import TrajectoryBuffer
from src.utils.helpers import compute_returns_to_go, get_latest_checkpoint
from src.models.snn_dt_patch import SNNDecisionTransformer
from src.models.dsf_dt import DecisionSpikeFormer
from src.train_snn_dt import TrajectoryDataset as SNNTrajectoryDataset, set_seed as snn_set_seed
from src.train_dsf_dt import TrajectoryDataset as DSFTrajectoryDataset, set_seed as dsf_set_seed, collect_trajectories as dsf_collect_trajectories
from src.run_benchmark import evaluate_model

# Environments to test
ENVIRONMENTS = ["CartPole-v1", "Acrobot-v1", "MountainCar-v0", "Pendulum-v1"]

def collect_shared_dataset(env_name):
    """Collect a shared dataset for both models"""
    print(f"Collecting shared dataset for {env_name}...")
    
    # Use the DSF collection function as it is already implemented
    trajectories, act_dim = dsf_collect_trajectories(env_name)
    
    # Save the dataset
    with open(f"shared_offline_data_{env_name}.pkl", "wb") as f:
        pickle.dump(trajectories, f)
    
    print(f"Collected {len(trajectories)} trajectories for {env_name}")
    return trajectories, act_dim

def train_model(model_class, trajectories, act_dim, env_name, is_continuous, model_name):
    """Train a model (SNN-DT or DSF-DT) with shared dataset"""
    print(f"Training {model_name} with shared dataset for {env_name}...")
    
    from torch.utils.data import DataLoader
    from src.utils.helpers import simple_logger, save_checkpoint
    
    set_seed_fn = snn_set_seed if model_name == "SNN-DT" else dsf_set_seed
    set_seed_fn(SEED)
    os.makedirs("checkpoints", exist_ok=True)
    
    # Build dataset & loader
    dataset_class = SNNTrajectoryDataset if model_name == "SNN-DT" else DSFTrajectoryDataset
    dataset = dataset_class(trajectories, max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Model & optimizer
    dt_conf = dt_config.copy()
    dt_conf.update(
        state_dim=dataset[0]["states"].shape[-1],
        act_dim=act_dim,
        max_length=max_length,
    )
    if model_name == "DSF-DT":
        dt_conf['num_training_steps'] = dt_epochs * len(loader)
        dt_conf['warmup_ratio'] = 0.1

    model = model_class(**dt_conf).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss() if is_continuous else nn.CrossEntropyLoss()
    
    all_losses = []
    for epoch in range(dt_epochs):
        total_loss = 0.0
        for batch in loader:
            states = batch["states"].to(DEVICE)
            actions = batch["actions"].to(DEVICE)
            returns = batch["returns_to_go"].to(DEVICE)
            times = batch["timesteps"].to(DEVICE)
            
            if is_continuous:
                actions_in = actions.to(torch.float32)
                targets = actions
            else:
                actions_in = nn.functional.one_hot(
                    actions.squeeze(-1).long(), num_classes=act_dim
                ).to(torch.float32)
                targets = actions.view(-1)

            _, action_preds, _ = model(states, actions_in, None if model_name == "SNN-DT" else returns, returns, times)
            
            if is_continuous:
                loss = loss_fn(action_preds, targets)
            else:
                logits = action_preds.view(-1, act_dim)
                loss = loss_fn(logits, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        all_losses.append(avg_loss)
        simple_logger({"epoch": epoch, f"avg_offline_loss_{env_name}_{model_name}": avg_loss}, epoch)
        save_checkpoint(model, optimizer, f"checkpoints/offline_{model_name}_{env_name}_{epoch}.pt")
        print(f"Epoch {epoch} ({model_name}, {env_name}): Avg Loss = {avg_loss:.4f}")
    
    print(f"{model_name} training complete for {env_name}.")
    return model, all_losses

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed.")
    parser.add_argument("--skip-training", action="store_true", help="Skip training and only evaluate.")
    args = parser.parse_args()
    
    all_results = []
    all_training_losses = []

    for env_name in ENVIRONMENTS:
        print(f"\n=== Running Comparison on {env_name} ===")
        
        # Determine if the environment has a continuous action space
        env = gym.make(env_name)
        is_continuous = isinstance(env.action_space, gym.spaces.Box)
        act_dim = env.action_space.shape[0] if is_continuous else env.action_space.n
        env.close()

        if not args.skip_training:
            trajectories, _ = collect_shared_dataset(env_name)
            
            # Train SNN-DT
            _, snn_losses = train_model(SNNDecisionTransformer, trajectories, act_dim, env_name, is_continuous, "SNN-DT")
            
            # Train DSF-DT
            _, dsf_losses = train_model(DecisionSpikeFormer, trajectories, act_dim, env_name, is_continuous, "DSF-DT")
            
            all_training_losses.append({
                "env": env_name,
                "snn_loss": snn_losses,
                "dsf_loss": dsf_losses
            })

        # Evaluate both models
        print(f"Evaluating models for {env_name}...")
        snn_results = evaluate_model(SNNDecisionTransformer, dt_config.copy(), env_name, "snn-dt", args.seed)
        dsf_results = evaluate_model(DecisionSpikeFormer, dt_config.copy(), env_name, "dsf-dt", args.seed)
        
        if snn_results and dsf_results:
            all_results.append(snn_results)
            all_results.append(dsf_results)

    # Save all results to a single CSV
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv("full_comparison_results.csv", index=False)
        print("\nFull comparison results saved to full_comparison_results.csv")

    # Save all training losses
    if all_training_losses:
        losses_df = pd.DataFrame(all_training_losses)
        losses_df.to_csv("full_training_losses.csv", index=False)
        print("Full training losses saved to full_training_losses.csv")

    # Generate summary report
    if all_results:
        summary = "## SNN-DT vs DSF-DT Comparison Summary\n\n"
        summary += "| Environment | Model | Avg Return | Std Return | Avg Latency (ms) | Avg Spikes/Episode | Total Params |\n"
        summary += "|-------------|-------|------------|------------|------------------|--------------------|--------------|\n"
        for res in all_results:
            summary += f"| {res['env_name']} | {res['model_name']} | {res['avg_return']:.2f} | {res['std_return']:.2f} | {res['avg_latency_ms']:.2f} | {res['avg_spikes_per_episode']:.0f} | {res['total_params']:,} |\n"
        
        with open("comparison_summary.md", "w") as f:
            f.write(summary)
        print("\nComparison summary saved to comparison_summary.md")

if __name__ == "__main__":
    main()
