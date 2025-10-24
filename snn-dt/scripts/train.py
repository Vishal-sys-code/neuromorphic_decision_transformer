import argparse
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from src.models.cql import CQL
from src.models.dt import DecisionTransformer
from src.models.dsformer import DsFormer
from src.models.iql import IQL
from src.models.snn_dt import SnnDt
from src.utils.config import AttrDict
from src.utils.models import get_model
from src.utils.seed import seed_everything
from scripts.eval import evaluate_policy


class OfflineDataset(Dataset):
    def __init__(self, dataset_path):
        data = np.load(dataset_path)
        self.states = torch.from_numpy(data["states"]).float()
        self.actions = torch.from_numpy(data["actions"]).float()
        self.returns_to_go = torch.from_numpy(data["returns_to_go"]).float()
        self.timesteps = torch.from_numpy(data["timesteps"]).long()
        self.mask = torch.from_numpy(data["mask"]).float()

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return {
            "states": self.states[idx],
            "actions": self.actions[idx],
            "returns_to_go": self.returns_to_go[idx],
            "timesteps": self.timesteps[idx],
            "mask": self.mask[idx],
        }


class OfflineTransitionDataset(Dataset):
    def __init__(self, dataset_path):
        data = np.load(dataset_path)
        
        states, actions, rewards, next_states, dones = [], [], [], [], []
        
        for i in range(data["states"].shape[0]):
            mask = data["mask"][i]
            clip_len = int(mask.sum())
            
            for t in range(clip_len - 1):
                states.append(data["states"][i, t])
                actions.append(data["actions"][i, t])
                rewards.append(data["returns_to_go"][i, t] - data["returns_to_go"][i, t+1])
                next_states.append(data["states"][i, t+1])
                dones.append(0.0)
            
            # Add final transition
            states.append(data["states"][i, clip_len-1])
            actions.append(data["actions"][i, clip_len-1])
            rewards.append(data["returns_to_go"][i, clip_len-1])
            next_states.append(np.zeros_like(data["states"][i, clip_len-1])) # Placeholder
            dones.append(1.0)

        self.states = torch.from_numpy(np.array(states)).float()
        self.actions = torch.from_numpy(np.array(actions)).float()
        self.rewards = torch.from_numpy(np.array(rewards)).float().unsqueeze(-1)
        self.next_states = torch.from_numpy(np.array(next_states)).float()
        self.dones = torch.from_numpy(np.array(dones)).float().unsqueeze(-1)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return {
            "states": self.states[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_states": self.next_states[idx],
            "dones": self.dones[idx],
        }


def train(cfg):
    seed_everything(cfg.seed)
    
    # Create save directory
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load data and metadata
    if cfg.model.name in ["iql", "cql"]:
        dataset = OfflineTransitionDataset(cfg.dataset.path)
    else:
        dataset = OfflineDataset(cfg.dataset.path)

    with np.load(cfg.dataset.path, allow_pickle=True) as data:
        metadata = data["metadata"].item()
        if isinstance(metadata, str):
            metadata = yaml.safe_load(metadata)
    
    # Update config with dataset metadata
    cfg.dataset.state_dim = metadata["state_dim"]
    cfg.dataset.act_dim = metadata["act_dim"]
    cfg.dataset.max_timesteps = metadata["max_timesteps"]

    train_loader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=4,
    )

    # Initialize model and optimizer
    model = get_model(cfg).to(cfg.training.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay),
    )
    loss_fn = torch.nn.MSELoss()

    # Training loop
    metrics = []
    start_time = time.time()
    env = gym.make(cfg.env)

    for epoch in range(cfg.training.epochs):
        for batch in train_loader:
            model.train()

            for k, v in batch.items():
                batch[k] = v.to(cfg.training.device)
            
            if isinstance(model, (IQL, CQL)):
                losses = model.learn(batch)
                loss = losses["policy_loss"] # Use policy loss for logging
            else:
                optimizer.zero_grad()
                action_preds = model(batch)
                loss = loss_fn(action_preds[batch["mask"] > 0], batch["actions"][batch["mask"] > 0])
                loss.backward()
                optimizer.step()

        # Evaluation and Logging
        if (epoch + 1) % cfg.training.eval_every == 0:
            eval_results = evaluate_policy(model, env, cfg, episodes=10)
            epoch_time = time.time() - start_time
            log_str = f"Epoch {epoch+1}/{cfg.training.epochs} | Time: {epoch_time:.2f}s"
            
            # Spike counting
            if isinstance(model, (SnnDt, DsFormer)):
                spikes = model.count_spikes()
                log_str += f" | Spikes: {spikes}"
                eval_results["spikes"] = spikes

            if isinstance(model, (IQL, CQL)):
                log_str += f" | Actor Loss: {losses['policy_loss']:.4f} | Critic1 Loss: {losses['critic1_loss']:.4f}"
                metrics.append({"epoch": epoch, **losses, **eval_results, "time_s": epoch_time})
            else:
                log_str += f" | Loss: {loss.item():.4f}"
                metrics.append({"epoch": epoch, "loss": loss.item(), **eval_results, "time_s": epoch_time})
            log_str += f" | Eval Return: {eval_results['return_mean']:.2f}"
            print(log_str)
            
            # Apply plasticity
            if isinstance(model, SnnDt) and model.use_plasticity:
                model.apply_plasticity(eval_results["return_mean"])
        
        # Save checkpoint
        if (epoch + 1) % cfg.training.checkpoint_every == 0:
            torch.save(model.state_dict(), save_dir / f"ckpt_epoch_{epoch+1}.pt")

    # Save metrics
    df = pd.DataFrame(metrics)
    df.to_csv(save_dir / "metrics.csv", index=False)
    
    # Update summary
    summary_path = Path(cfg.save_dir).parent.parent / "summary.csv"
    summary_df = pd.DataFrame([{"model": cfg.model.name, "env": cfg.env, "seed": cfg.seed, "return_mean": df["return_mean"].max()}])
    if summary_path.exists():
        summary_df.to_csv(summary_path, mode="a", header=False, index=False)
    else:
        summary_df.to_csv(summary_path, index=False)

    print("Training complete.")
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/sdt_cartpole.yaml", help="Path to config file.")
    parser.add_argument("--model", type=str, required=True, help="Name of the model to train (dt, snn_dt, dsformer, iql, cql).")
    parser.add_argument("--env", type=str, required=True, help="Environment name (e.g., CartPole-v1).")
    parser.add_argument("--save-dir", type=str, default="results/run", help="Directory to save results.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    
    # Convert to AttrDict for easy access
    cfg = AttrDict(cfg)

    # Override with CLI args
    cfg.model.name = args.model
    cfg.env = args.env
    cfg.seed = args.seed
    cfg.save_dir = args.save_dir
    
    # Construct dataset path from env name
    cfg.dataset.path = f"data/{args.env}/dataset.npz"

    train(cfg)


if __name__ == "__main__":
    main()