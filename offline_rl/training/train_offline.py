"""
Offline Training Pipeline for Spiking Decision Transformer on MuJoCo / D4RL.

Trains SNN-DT policy on offline continuous control trajectories using behavior
cloning conditioned on returns-to-go, with AdamW optimization, learning rate warmup,
and neuromorphic activity monitoring.
"""

import argparse
import os
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

from offline_rl.datasets.d4rl_loader import D4RLLoader
from offline_rl.datasets.trajectory_dataset import TrajectorySequenceDataset
from offline_rl.models.snn_dt_mujoco import ContinuousSnnDecisionTransformer


class OfflineTrainer:
    """Trainer orchestrator for offline continuous control DT."""

    def __init__(
        self,
        model: ContinuousSnnDecisionTransformer,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[Any] = None,
        device: str = "cpu",
        grad_norm_clip: float = 0.25,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.grad_norm_clip = grad_norm_clip

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()

        states = batch["states"].to(self.device)
        actions = batch["actions"].to(self.device)
        returns_to_go = batch["returns_to_go"].to(self.device)
        timesteps = batch["timesteps"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        action_preds, diagnostics = self.model(
            states=states,
            actions=actions,
            returns_to_go=returns_to_go,
            timesteps=timesteps,
            attention_mask=attention_mask,
        )

        # Compute MSE loss masked by attention_mask (B, K)
        act_dim = actions.shape[-1]
        loss_unmasked = F.mse_loss(action_preds, actions, reduction="none")  # (B, K, act_dim)
        mask_expanded = attention_mask.unsqueeze(-1).expand_as(loss_unmasked)
        loss = (loss_unmasked * mask_expanded).sum() / mask_expanded.sum()

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm_clip)
        self.optimizer.step()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        lr = self.optimizer.param_groups[0]["lr"]

        metrics = {
            "train/loss": float(loss.item()),
            "train/grad_norm": float(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm),
            "train/lr": float(lr),
            "train/mean_spike_rate": float(diagnostics.get("mean_spike_rate", 0.0)),
        }
        return metrics

    def train_epoch(self, num_steps: int = 100) -> Dict[str, float]:
        step = 0
        epoch_losses = []
        epoch_spike_rates = []

        loader_iter = iter(self.train_loader)
        while step < num_steps:
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(self.train_loader)
                batch = next(loader_iter)

            metrics = self.train_step(batch)
            epoch_losses.append(metrics["train/loss"])
            epoch_spike_rates.append(metrics["train/mean_spike_rate"])
            step += 1

        return {
            "epoch/mean_loss": float(np.mean(epoch_losses)),
            "epoch/mean_spike_rate": float(np.mean(epoch_spike_rates)),
        }


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_offline(config_path: str, data_path: Optional[str] = None, output_dir: str = "checkpoints/offline_rl"):
    config = load_config(config_path)
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[OfflineRL] Using device: {device}")

    # Load or generate trajectories
    loader = D4RLLoader(gamma=config.get("gamma", 1.0))
    if data_path and os.path.exists(data_path):
        print(f"[OfflineRL] Loading trajectories from {data_path}")
        trajectories = loader.load_from_hdf5(data_path)
    else:
        print("[OfflineRL] No local dataset path provided or file not found. Generating synthetic dataset for initialization.")
        trajectories = loader.create_synthetic_dataset(
            num_trajectories=50,
            traj_len=100,
            state_dim=config["state_dim"],
            act_dim=config["act_dim"],
        )

    # Prepare dataset and dataloader
    dataset = TrajectorySequenceDataset(
        trajectories=trajectories,
        context_len=config.get("context_len", 20),
        scale=config.get("scale", 1000.0),
        max_ep_len=config.get("max_ep_len", 1000),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config["training"].get("batch_size", 64),
        shuffle=True,
        drop_last=True,
    )

    # Initialize model
    m_cfg = config.get("model", {})
    model = ContinuousSnnDecisionTransformer(
        state_dim=config["state_dim"],
        act_dim=config["act_dim"],
        embed_dim=m_cfg.get("embed_dim", 128),
        n_layer=m_cfg.get("n_layer", 3),
        n_head=m_cfg.get("n_head", 4),
        max_ep_len=config.get("max_ep_len", 1000),
        context_len=config.get("context_len", 20),
        snn_timesteps=m_cfg.get("snn_timesteps", 4),
        tau=m_cfg.get("tau", 20.0),
        v_th=m_cfg.get("v_th", 0.5),
        dropout=m_cfg.get("dropout", 0.1),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"].get("learning_rate", 1e-4)),
        weight_decay=float(config["training"].get("weight_decay", 1e-4)),
    )

    trainer = OfflineTrainer(
        model=model,
        train_loader=dataloader,
        optimizer=optimizer,
        device=device,
        grad_norm_clip=config["training"].get("grad_norm_clip", 0.25),
    )

    max_iters = config["training"].get("max_iters", 5)
    steps_per_iter = config["training"].get("num_steps_per_iter", 100)

    for iteration in range(1, max_iters + 1):
        t0 = time.time()
        metrics = trainer.train_epoch(num_steps=steps_per_iter)
        elapsed = time.time() - t0
        print(
            f"Iter {iteration}/{max_iters} ({elapsed:.1f}s) | "
            f"Loss: {metrics['epoch/mean_loss']:.4f} | "
            f"Spike Rate: {metrics['epoch/mean_spike_rate']:.4f}"
        )

    ckpt_path = os.path.join(output_dir, f"{config['env_name']}_snn_dt.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "state_mean": dataset.state_mean,
            "state_std": dataset.state_std,
            "config": config,
        },
        ckpt_path,
    )
    print(f"[OfflineRL] Saved checkpoint to {ckpt_path}")
    return model, dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SNN-DT on MuJoCo offline data")
    parser.add_argument("--config", type=str, default="offline_rl/configs/hopper_medium.yaml")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="checkpoints/offline_rl")
    args = parser.parse_args()

    train_offline(args.config, args.data, args.output_dir)
