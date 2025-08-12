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

import importlib

from src.utils.trajectory_buffer import TrajectoryBuffer
from src.utils.helpers import compute_returns_to_go
from src.models.snn_dt_patch import SNNDecisionTransformer
from src.models.dsf_dt import DecisionSpikeFormer
from src.train_snn_dt import TrajectoryDataset as SNNTrajectoryDataset, set_seed as snn_set_seed
from src.train_dsf_dt import TrajectoryDataset as DSFTrajectoryDataset, set_seed as dsf_set_seed, collect_trajectories as dsf_collect_trajectories
from src.run_benchmark import evaluate_model

def load_config(env_name):
    """Dynamically load the configuration file for the given environment."""
    try:
        config_module_name = f"src.configs.{env_name.lower().replace('-', '_')}_config"
        config = importlib.import_module(config_module_name)
        return config
    except ImportError:
        raise FileNotFoundError(f"Configuration file for environment '{env_name}' not found.")

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

def train_model(model_class, trajectories, act_dim, env_name, is_continuous, model_name, config):
    """Train a model (SNN-DT or DSF-DT) with shared dataset"""
    print(f"Training {model_name} with shared dataset for {env_name}...")
    
    from torch.utils.data import DataLoader, random_split
    from src.utils.helpers import simple_logger, save_checkpoint
    
    set_seed_fn = snn_set_seed if model_name == "SNN-DT" else dsf_set_seed
    set_seed_fn(config.SEED)
    os.makedirs("checkpoints", exist_ok=True)
    
    # Build dataset
    dataset_class = SNNTrajectoryDataset if model_name == "SNN-DT" else DSFTrajectoryDataset
    dataset = dataset_class(trajectories, config.max_length)
    
    # Split dataset into training and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config.dt_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.dt_batch_size, shuffle=False)
    
    # Model & optimizer
    dt_conf = config.dt_config.copy()
    dt_conf.update(
        state_dim=dataset[0]["states"].shape[-1],
        act_dim=act_dim,
        max_length=config.max_length,
    )
    if model_name == "DSF-DT":
        dt_conf['num_training_steps'] = config.dt_epochs * len(train_loader)
        dt_conf['warmup_ratio'] = 0.1

    model = model_class(**dt_conf).to(config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    loss_fn = nn.MSELoss() if is_continuous else nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    best_model_path = None
    
    all_losses = []
    for epoch in range(config.dt_epochs):
        # Training loop
        model.train()
        total_train_loss = 0.0
        for batch in train_loader:
            states = batch["states"].to(config.DEVICE)
            actions = batch["actions"].to(config.DEVICE)
            returns = batch["returns_to_go"].to(config.DEVICE)
            times = batch["timesteps"].to(config.DEVICE)
            
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
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation loop
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                states = batch["states"].to(config.DEVICE)
                actions = batch["actions"].to(config.DEVICE)
                returns = batch["returns_to_go"].to(config.DEVICE)
                times = batch["timesteps"].to(config.DEVICE)
                
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
                
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        all_losses.append({"train_loss": avg_train_loss, "val_loss": avg_val_loss})
        
        simple_logger({
            "epoch": epoch, 
            f"avg_train_loss_{env_name}_{model_name}": avg_train_loss,
            f"avg_val_loss_{env_name}_{model_name}": avg_val_loss
        }, epoch)
        
        print(f"Epoch {epoch} ({model_name}, {env_name}): Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = f"checkpoints/best_{model_name}_{env_name}_seed{config.SEED}.pt"
            save_checkpoint(model, optimizer, best_model_path)
            print(f"New best model saved to {best_model_path}")

    print(f"{model_name} training complete for {env_name}.")
    
    # Load the best model for evaluation
    if best_model_path:
        state_dict = torch.load(best_model_path)
        model.load_state_dict(state_dict['model_state'])
        
    return model, all_losses

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", type=str, required=True, help="Name of the environment (e.g., CartPole-v1).")
    parser.add_argument("--seeds", nargs='+', type=int, default=[42, 123, 567], help="List of random seeds.")
    args = parser.parse_args()
    
    config = load_config(args.env_name)
    env_name = config.ENV_NAME
    
    all_results = []
    all_training_losses = []

    # Determine if the environment has a continuous action space
    env = gym.make(env_name)
    is_continuous = isinstance(env.action_space, gym.spaces.Box)
    act_dim = env.action_space.shape[0] if is_continuous else env.action_space.n
    env.close()

    trajectories, _ = collect_shared_dataset(env_name)

    for seed in args.seeds:
        print(f"\n=== Running Comparison on {env_name} with Seed {seed} ===")
        
        # Train SNN-DT
        snn_model, snn_losses = train_model(SNNDecisionTransformer, trajectories, act_dim, env_name, is_continuous, "SNN-DT", config)
        
        # Train DSF-DT
        dsf_model, dsf_losses = train_model(DecisionSpikeFormer, trajectories, act_dim, env_name, is_continuous, "DSF-DT", config)
        
        all_training_losses.append({
            "seed": seed,
            "env": env_name,
            "snn_loss": snn_losses,
            "dsf_loss": dsf_losses
        })

        # Evaluate both models
        print(f"Evaluating models for {env_name} with seed {seed}...")
        snn_results = evaluate_model(snn_model, env_name, "snn-dt", seed)
        dsf_results = evaluate_model(dsf_model, env_name, "dsf-dt", seed)
        
        if snn_results and dsf_results:
            all_results.append(snn_results)
            all_results.append(dsf_results)

    # Save all results to a single CSV
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(f"comparison_results_{env_name}_multi_seed.csv", index=False)
        print(f"\nMulti-seed comparison results saved to comparison_results_{env_name}_multi_seed.csv")

        # Aggregate results
        agg_results = results_df.groupby('model_name').agg({
            'avg_return': ['mean', 'std'],
            'std_return': ['mean', 'std'],
            'avg_latency_ms': ['mean', 'std'],
            'avg_spikes_per_episode': ['mean', 'std'],
            'total_params': 'first'
        }).reset_index()
        
        agg_results.columns = ['_'.join(col).strip() for col in agg_results.columns.values]
        agg_results.rename(columns={'model_name_': 'model_name', 'total_params_first': 'total_params'}, inplace=True)

        # Generate summary report
        summary = f"## SNN-DT vs DSF-DT Multi-Seed Comparison Summary ({env_name})\n\n"
        summary += "| Model | Avg Return (Mean ± Std) | Avg Latency (ms) (Mean ± Std) | Avg Spikes/Episode (Mean ± Std) | Total Params |\n"
        summary += "|-------|-------------------------|-------------------------------|---------------------------------|--------------|\n"
        for _, row in agg_results.iterrows():
            summary += (f"| {row['model_name']} | "
                        f"{row['avg_return_mean']:.2f} ± {row['avg_return_std']:.2f} | "
                        f"{row['avg_latency_ms_mean']:.2f} ± {row['avg_latency_ms_std']:.2f} | "
                        f"{row['avg_spikes_per_episode_mean']:.0f} ± {row['avg_spikes_per_episode_std']:.0f} | "
                        f"{row['total_params']:,} |\n")
        
        with open(f"comparison_summary_{env_name}_multi_seed.md", "w") as f:
            f.write(summary)
        print(f"\nMulti-seed comparison summary saved to comparison_summary_{env_name}_multi_seed.md")


    # Save all training losses
    if all_training_losses:
        with open(f"training_losses_{env_name}_multi_seed.pkl", "wb") as f:
            pickle.dump(all_training_losses, f)
        print(f"Training losses saved to training_losses_{env_name}_multi_seed.pkl")

if __name__ == "__main__":
    main()
