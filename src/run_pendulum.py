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

from src.configs.pendulum_config import (
    DEVICE, SEED,
    offline_steps,
    dt_batch_size as batch_size,
    dt_epochs,
    gamma,
    max_length,
    lr,
    dt_config,
    ENV_NAME,
)
from src.utils.trajectory_buffer import TrajectoryBuffer
from src.utils.helpers import compute_returns_to_go
from src.models.snn_dt_patch import SNNDecisionTransformer
from src.models.dsf_dt import DecisionSpikeFormer
from src.train_snn_dt import TrajectoryDataset as SNNTrajectoryDataset, set_seed as snn_set_seed
from src.train_dsf_dt import TrajectoryDataset as DSFTrajectoryDataset, set_seed as dsf_set_seed, collect_trajectories as dsf_collect_trajectories
from src.run_benchmark import evaluate_model

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
    
    from torch.utils.data import DataLoader, random_split
    from src.utils.helpers import simple_logger, save_checkpoint
    
    set_seed_fn = snn_set_seed if model_name == "SNN-DT" else dsf_set_seed
    set_seed_fn(SEED)
    os.makedirs("checkpoints", exist_ok=True)
    
    # Build dataset
    dataset_class = SNNTrajectoryDataset if model_name == "SNN-DT" else DSFTrajectoryDataset
    dataset = dataset_class(trajectories, max_length)
    
    # Split dataset into training and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Model & optimizer
    dt_conf = dt_config.copy()
    dt_conf.update(
        state_dim=dataset[0]["states"].shape[-1],
        act_dim=act_dim,
        max_length=max_length,
    )
    if model_name == "DSF-DT":
        dt_conf['num_training_steps'] = dt_epochs * len(train_loader)
        dt_conf['warmup_ratio'] = 0.1

    model = model_class(**dt_conf).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss() if is_continuous else nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    best_model_path = None
    
    all_losses = []
    for epoch in range(dt_epochs):
        # Training loop
        model.train()
        total_train_loss = 0.0
        for batch in train_loader:
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
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation loop
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
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
            best_model_path = f"checkpoints/best_{model_name}_{env_name}.pt"
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
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed.")
    args = parser.parse_args()
    
    all_results = []
    all_training_losses = []

    env_name = ENV_NAME
    print(f"\n=== Running Comparison on {env_name} ===")
    
    # Determine if the environment has a continuous action space
    env = gym.make(env_name)
    is_continuous = isinstance(env.action_space, gym.spaces.Box)
    act_dim = env.action_space.shape[0] if is_continuous else env.action_space.n
    env.close()

    trajectories, _ = collect_shared_dataset(env_name)
    
    # Train SNN-DT
    snn_model, snn_losses = train_model(SNNDecisionTransformer, trajectories, act_dim, env_name, is_continuous, "SNN-DT")
    
    # Train DSF-DT
    dsf_model, dsf_losses = train_model(DecisionSpikeFormer, trajectories, act_dim, env_name, is_continuous, "DSF-DT")
    
    all_training_losses.append({
        "env": env_name,
        "snn_loss": snn_losses,
        "dsf_loss": dsf_losses
    })

    # Evaluate both models
    print(f"Evaluating models for {env_name}...")
    snn_results = evaluate_model(snn_model, env_name, "snn-dt", args.seed)
    dsf_results = evaluate_model(dsf_model, env_name, "dsf-dt", args.seed)
    
    if snn_results and dsf_results:
        all_results.append(snn_results)
        all_results.append(dsf_results)

    # Save all results to a single CSV
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(f"comparison_results_{env_name}.csv", index=False)
        print(f"\nComparison results saved to comparison_results_{env_name}.csv")

    # Save all training losses
    if all_training_losses:
        # This part needs to be adapted to handle the new loss format (list of dicts)
        # For now, let's just save the raw data. A more sophisticated plot script can parse this.
        with open(f"training_losses_{env_name}.pkl", "wb") as f:
            pickle.dump(all_training_losses, f)
        print(f"Training losses saved to training_losses_{env_name}.pkl")

    # Generate summary report
    if all_results:
        summary = f"## SNN-DT vs DSF-DT Comparison Summary ({env_name})\n\n"
        summary += "| Model | Avg Return | Std Return | Avg Latency (ms) | Avg Spikes/Episode | Total Params |\n"
        summary += "|-------|------------|------------|------------------|--------------------|--------------|\n"
        for res in all_results:
            summary += f"| {res['model_name']} | {res['avg_return']:.2f} | {res['std_return']:.2f} | {res['avg_latency_ms']:.2f} | {res['avg_spikes_per_episode']:.0f} | {res['total_params']:,} |\n"
        
        with open(f"comparison_summary_{env_name}.md", "w") as f:
            f.write(summary)
        print(f"\nComparison summary saved to comparison_summary_{env_name}.md")

if __name__ == "__main__":
    main()