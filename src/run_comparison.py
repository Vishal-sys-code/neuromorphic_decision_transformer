"""
Run comparison between SNN-DT and DSF-DT on CartPole-v1
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
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

def collect_shared_dataset(env_name="CartPole-v1"):
    """Collect a shared dataset for both models"""
    print(f"Collecting shared dataset for {env_name}...")
    
    # Use the DSF collection function since it's already implemented
    trajectories, act_dim = dsf_collect_trajectories(env_name)
    
    # Save the dataset
    with open(f"shared_offline_data_{env_name}.pkl", "wb") as f:
        pickle.dump(trajectories, f)
    
    print(f"Collected {len(trajectories)} trajectories")
    return trajectories, act_dim

def train_snn_dt_with_shared_data(trajectories, act_dim_from_env, env_name="CartPole-v1"):
    """Train SNN-DT with shared dataset"""
    print("Training SNN-DT with shared dataset...")
    
    from torch.utils.data import DataLoader
    from src.utils.helpers import simple_logger, save_checkpoint
    
    # Set seed
    snn_set_seed(SEED)
    os.makedirs("checkpoints", exist_ok=True)
    
    # Build dataset & loader
    dataset = SNNTrajectoryDataset(trajectories, max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Model & optimizer
    dt_conf = dt_config.copy()
    dt_conf.update(
        state_dim=dataset[0]["states"].shape[-1],
        act_dim=act_dim_from_env,
        max_length=max_length,
    )
    model = SNNDecisionTransformer(**dt_conf).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # Training loop
    all_losses = []
    for epoch in range(dt_epochs):
        total_loss = 0.0
        for batch in loader:
            states = batch["states"].to(DEVICE)
            actions = batch["actions"].to(DEVICE)
            returns = batch["returns_to_go"].to(DEVICE)
            times = batch["timesteps"].to(DEVICE)
            
            # one-hot actions for input embedding
            actions_in = torch.nn.functional.one_hot(
                actions.squeeze(-1), num_classes=dt_conf["act_dim"]
            ).to(torch.float32)
            
            # forward: predict next actions
            _, action_preds, _ = model(states, actions_in, None, returns, times)
            # compute CE loss on all positions
            logits = action_preds.view(-1, dt_conf["act_dim"])
            targets = actions.view(-1)
            loss = loss_fn(logits, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        all_losses.append(avg_loss)
        simple_logger({"epoch": epoch, "avg_offline_loss": avg_loss}, epoch)
        save_checkpoint(model, optimizer, f"checkpoints/offline_dt_{env_name}_{epoch}.pt")
        
        print(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")
    
    print("SNN-DT training complete.")
    return model, all_losses

def train_dsf_dt_with_shared_data(trajectories, act_dim_from_env, env_name="CartPole-v1"):
    """Train DSF-DT with shared dataset"""
    print("Training DSF-DT with shared dataset...")
    
    from torch.utils.data import DataLoader
    from src.utils.helpers import simple_logger, save_checkpoint
    import torch.nn as nn
    
    # Set seed
    dsf_set_seed(SEED)
    os.makedirs("checkpoints", exist_ok=True)
    
    # Build dataset & loader
    dataset = DSFTrajectoryDataset(trajectories, max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Model & optimizer
    dt_conf = dt_config.copy()
    dt_conf.update(
        state_dim=dataset[0]["states"].shape[-1],
        act_dim=act_dim_from_env,
        max_length=max_length,
    )
    # Add num_training_steps to the config for the model
    dt_conf['num_training_steps'] = dt_epochs * len(loader)
    dt_conf['warmup_ratio'] = 0.1
    
    model = DecisionSpikeFormer(**dt_conf).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    
    # Training loop
    all_losses = []
    for epoch in range(dt_epochs):
        total_loss = 0.0
        for batch in loader:
            states = batch["states"].to(DEVICE)
            actions = batch["actions"].to(DEVICE)
            returns = batch["returns_to_go"].to(DEVICE)
            times = batch["timesteps"].to(DEVICE)
            
            # dsf expects actions to be float
            actions_in = actions.to(torch.float32)
            
            _, action_preds, _ = model(states, actions_in, returns, times)
            
            logits = action_preds.view(-1, dt_conf["act_dim"])
            targets = actions.view(-1)
            loss = loss_fn(logits, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        all_losses.append(avg_loss)
        simple_logger({"epoch": epoch, "avg_offline_loss": avg_loss}, epoch)
        save_checkpoint(model, optimizer, f"checkpoints/offline_dsf_{env_name}_{epoch}.pt")
        
        print(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")
    
    print("DSF-DT training complete.")
    return model, all_losses

def evaluate_both_models(env_name="CartPole-v1", seed=SEED):
    """Evaluate both models and compare results"""
    print("Evaluating both models...")
    
    # Evaluate SNN-DT
    snn_results = evaluate_model(
        SNNDecisionTransformer, 
        dt_config.copy(), 
        env_name, 
        "snn-dt", 
        seed
    )
    
    # Evaluate DSF-DT
    dsf_results = evaluate_model(
        DecisionSpikeFormer, 
        dt_config.copy(), 
        env_name, 
        "dsf-dt", 
        seed
    )
    
    # Combine results
    if snn_results and dsf_results:
        results_df = pd.DataFrame([snn_results, dsf_results])
        results_file = "comparison_results.csv"
        results_df.to_csv(results_file, index=False)
        print(f"Comparison results saved to {results_file}")
        
        # Print comparison
        print("\n=== Comparison Results ===")
        print(f"Environment: {env_name}")
        print(f"Seed: {seed}")
        print("\nSNN-DT Results:")
        print(f"  Average Return: {snn_results['avg_return']:.2f} ± {snn_results['std_return']:.2f}")
        print(f"  Average Latency: {snn_results['avg_latency_ms']:.2f} ms")
        print(f"  Average Spikes: {snn_results['avg_spikes_per_episode']:.0f}")
        print(f"  Total Parameters: {snn_results['total_params']:,}")
        
        print("\nDSF-DT Results:")
        print(f"  Average Return: {dsf_results['avg_return']:.2f} ± {dsf_results['std_return']:.2f}")
        print(f"  Average Latency: {dsf_results['avg_latency_ms']:.2f} ms")
        print(f"  Average Spikes: {dsf_results['avg_spikes_per_episode']:.0f}")
        print(f"  Total Parameters: {dsf_results['total_params']:,}")
        
        # Performance comparison
        print("\n=== Performance Comparison ===")
        return_diff = snn_results['avg_return'] - dsf_results['avg_return']
        latency_diff = snn_results['avg_latency_ms'] - dsf_results['avg_latency_ms']
        spikes_diff = snn_results['avg_spikes_per_episode'] - dsf_results['avg_spikes_per_episode']
        
        print(f"Return Difference (SNN-DT - DSF-DT): {return_diff:.2f}")
        print(f"Latency Difference (SNN-DT - DSF-DT): {latency_diff:.2f} ms")
        print(f"Spikes Difference (SNN-DT - DSF-DT): {spikes_diff:.0f}")
        
        if return_diff > 0:
            print("SNN-DT has higher returns")
        elif return_diff < 0:
            print("DSF-DT has higher returns")
        else:
            print("Both models have equal returns")
            
        return results_df
    
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Gym environment name.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed.")
    parser.add_argument("--skip-training", action="store_true", help="Skip training and only evaluate.")
    args = parser.parse_args()
    
    env_name = args.env
    seed = args.seed
    
    print(f"=== SNN-DT vs DSF-DT Comparison on {env_name} ===")
    print(f"Random seed: {seed}")
    
    if not args.skip_training:
        # 1. Collect shared dataset
        trajectories, act_dim = collect_shared_dataset(env_name)
        
        # 2. Train SNN-DT
        snn_model, snn_losses = train_snn_dt_with_shared_data(trajectories, act_dim, env_name)
        
        # 3. Train DSF-DT
        dsf_model, dsf_losses = train_dsf_dt_with_shared_data(trajectories, act_dim, env_name)
        
        # 4. Save training losses
        losses_df = pd.DataFrame({
            "epoch": list(range(dt_epochs)),
            "snn_loss": snn_losses,
            "dsf_loss": dsf_losses
        })
        losses_df.to_csv("training_losses.csv", index=False)
        print("Training losses saved to training_losses.csv")
    else:
        print("Skipping training phase...")
    
    # 5. Evaluate both models
    results_df = evaluate_both_models(env_name, seed)
    
    if results_df is not None:
        print("\n=== Comparison Complete ===")
        print("Results saved to comparison_results.csv")
    else:
        print("Evaluation failed.")

if __name__ == "__main__":
    main()