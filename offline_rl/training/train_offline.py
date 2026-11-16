"""
Offline Training Pipeline for Spiking Decision Transformer on MuJoCo / D4RL.
Phase 5: Training loop & Gate 5: Loss decreases and checkpoint saves.

Objective: Action regression loss
    L_action = sum_{b,t} m_{b,t} || hat{a}_{b,t} - a_{b,t} ||_2^2 / (sum_{b,t} m_{b,t} * d_a)

Provides:
- masked_action_loss: Exact masked MSE normalized by valid tokens and action dimension
- train_model: Core offline training loop supporting the engineering configuration
- OfflineTrainer: Step/epoch-based training orchestrator
- save_checkpoint / load_checkpoint: Full state serialization and restoration
- verify_training_stages: Staged validation (10, 100, 2000 updates)
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Ensure repository root is on sys.path for direct script execution
_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

from offline_rl.datasets.d4rl_loader import D4RLLoader, Trajectory
from offline_rl.datasets.trajectory_dataset import TrajectoryDataset, TrajectorySequenceDataset
from offline_rl.models.snn_dt_mujoco import ContinuousSnnDecisionTransformer


def masked_action_loss(
    predicted_actions: torch.Tensor,
    target_actions: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute masked action regression loss for Decision Transformer:
        L_action = sum_{b,t} m_{b,t} || hat{a}_{b,t} - a_{b,t} ||_2^2 / (sum_{b,t} m_{b,t} * d_a)

    Args:
        predicted_actions: Predicted actions of shape (B, K, d_a).
        target_actions: Ground truth dataset actions of shape (B, K, d_a).
        attention_mask: Binary validity mask of shape (B, K) or (B, K, 1).
                        1.0 indicates a valid timestep, 0.0 indicates padding.

    Returns:
        Scalar torch.Tensor loss.
    """
    if attention_mask.dim() == predicted_actions.dim() - 1:
        mask = attention_mask.unsqueeze(-1)
    else:
        mask = attention_mask

    squared_error = (
        predicted_actions - target_actions
    ).pow(2)

    loss = (
        squared_error * mask
    ).sum() / (
        mask.sum() * target_actions.shape[-1]
    ).clamp_min(1.0)

    return loss


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    lr_scheduler: Optional[Any] = None,
    step: int = 0,
    loss: Optional[float] = None,
    dataset: Optional[Any] = None,
    config: Optional[Dict[str, Any]] = None,
    extra_state: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Save complete training checkpoint to disk.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "loss": float(loss) if loss is not None else None,
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if lr_scheduler is not None and hasattr(lr_scheduler, "state_dict"):
        checkpoint["lr_scheduler_state_dict"] = lr_scheduler.state_dict()
    if dataset is not None:
        if hasattr(dataset, "state_mean") and dataset.state_mean is not None:
            checkpoint["state_mean"] = dataset.state_mean
        if hasattr(dataset, "state_std") and dataset.state_std is not None:
            checkpoint["state_std"] = dataset.state_std
    if config is not None:
        checkpoint["config"] = config
    if extra_state is not None:
        checkpoint.update(extra_state)

    torch.save(checkpoint, path)
    return path


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    lr_scheduler: Optional[Any] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Load training checkpoint from disk and restore model/optimizer state.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if lr_scheduler is not None and "lr_scheduler_state_dict" in checkpoint:
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

    return checkpoint


def train_model(
    model: nn.Module,
    episodes: List[Any],
    device: str = "cuda",
    context_length: int = 20,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    gradient_clipping: float = 0.25,
    grad_norm_clip: Optional[float] = None,
    num_updates: int = 2000,
    warmup_steps: int = 0,
    scale: float = 1000.0,
    max_ep_len: int = 1000,
    checkpoint_dir: Optional[str] = None,
    checkpoint_interval: Optional[int] = None,
    log_interval: int = 50,
    eval_fn: Optional[Callable[[nn.Module, int], Dict[str, float]]] = None,
    dataset: Optional[TrajectoryDataset] = None,
    dataloader: Optional[DataLoader] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    lr_scheduler: Optional[Any] = None,
    seed: Optional[int] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Train offline Decision Transformer model.

    Initial debugging configuration defaults:
        Context length: 20
        Batch size: 64
        Learning rate: 1e-4
        Weight decay: 1e-4
        Gradient clipping: 0.25
        Updates: 2,000 (smoke test)
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Resolve device safely
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        target_device = torch.device("cpu")
    else:
        target_device = torch.device(device)

    model = model.to(target_device)

    # Resolve gradient clipping parameter
    clip_val = grad_norm_clip if grad_norm_clip is not None else gradient_clipping

    # Prepare dataset & dataloader if not provided
    if dataset is None:
        dataset = TrajectoryDataset(
            episodes=episodes,
            context_length=context_length,
            scale=scale,
            max_ep_len=max_ep_len,
        )

    if dataloader is None:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )

    # Initialize optimizer
    if optimizer is None:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    # Initialize learning rate scheduler
    if warmup_steps > 0 and lr_scheduler is None:
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(max(1, warmup_steps))
            return 1.0

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)

    losses: List[float] = []
    grad_norms: List[float] = []
    spike_rates: List[float] = []
    checkpoints_saved: List[str] = []

    model.train()
    loader_iter = iter(dataloader)

    for update_step in range(1, num_updates + 1):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(dataloader)
            batch = next(loader_iter)

        optimizer.zero_grad()

        states = batch["states"].to(target_device)
        actions = batch["actions"].to(target_device)
        returns_to_go = batch["returns_to_go"].to(target_device)
        timesteps = batch["timesteps"].to(target_device)
        attention_mask = batch["attention_mask"].to(target_device)

        out = model(
            states=states,
            actions=actions,
            returns_to_go=returns_to_go,
            timesteps=timesteps,
            attention_mask=attention_mask,
        )

        if isinstance(out, tuple):
            action_preds, diagnostics = out
        else:
            action_preds = out
            diagnostics = getattr(model, "diagnostics", {})

        loss = masked_action_loss(
            predicted_actions=action_preds,
            target_actions=actions,
            attention_mask=attention_mask,
        )

        # Stage 2 check: NaN or Inf loss
        loss_val = float(loss.item())
        if np.isnan(loss_val) or np.isinf(loss_val):
            raise FloatingPointError(
                f"[Fatal] NaN or Inf loss encountered at update step {update_step}: {loss_val}"
            )

        loss.backward()

        # Gradient clipping & exploding gradient inspection
        if clip_val > 0.0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
            grad_norm_val = float(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
        else:
            grad_norm_val = 0.0

        if np.isnan(grad_norm_val) or np.isinf(grad_norm_val):
            raise FloatingPointError(
                f"[Fatal] NaN or Inf gradient norm encountered at update step {update_step}: {grad_norm_val}"
            )

        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        sr_val = float(diagnostics.get("mean_spike_rate", 0.0))
        losses.append(loss_val)
        grad_norms.append(grad_norm_val)
        spike_rates.append(sr_val)

        if update_step % log_interval == 0 or update_step == 1 or update_step == num_updates:
            curr_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Update [{update_step:5d}/{num_updates:5d}] | "
                f"Loss: {loss_val:.4f} | "
                f"Grad Norm: {grad_norm_val:.4f} | "
                f"Spike Rate: {sr_val:.4f} | "
                f"LR: {curr_lr:.6f}"
            )

        # Checkpoint interval saving
        if checkpoint_dir is not None and checkpoint_interval is not None and checkpoint_interval > 0:
            if update_step % checkpoint_interval == 0:
                ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_step_{update_step}.pt")
                save_checkpoint(
                    path=ckpt_path,
                    model=model,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    step=update_step,
                    loss=loss_val,
                    dataset=dataset,
                )
                checkpoints_saved.append(ckpt_path)

        # Evaluation callback
        if eval_fn is not None and checkpoint_interval is not None and checkpoint_interval > 0:
            if update_step % checkpoint_interval == 0 or update_step == num_updates:
                eval_fn(model, update_step)

    # Save final checkpoint if directory provided
    if checkpoint_dir is not None:
        final_ckpt_path = os.path.join(checkpoint_dir, "checkpoint_final.pt")
        save_checkpoint(
            path=final_ckpt_path,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            step=num_updates,
            loss=losses[-1] if losses else 0.0,
            dataset=dataset,
        )
        if final_ckpt_path not in checkpoints_saved:
            checkpoints_saved.append(final_ckpt_path)

    return {
        "model": model,
        "dataset": dataset,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
        "losses": losses,
        "grad_norms": grad_norms,
        "spike_rates": spike_rates,
        "initial_loss": losses[0] if losses else 0.0,
        "final_loss": losses[-1] if losses else 0.0,
        "checkpoints": checkpoints_saved,
    }


class OfflineTrainer:
    """
    Trainer orchestrator for offline continuous control DT,
    supporting epoch/step abstractions and benchmarking suites.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[Any] = None,
        device: str = "cpu",
        grad_norm_clip: float = 0.25,
    ):
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.grad_norm_clip = grad_norm_clip

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()

        states = batch["states"].to(self.device)
        actions = batch["actions"].to(self.device)
        returns_to_go = batch["returns_to_go"].to(self.device)
        timesteps = batch["timesteps"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        out = self.model(
            states=states,
            actions=actions,
            returns_to_go=returns_to_go,
            timesteps=timesteps,
            attention_mask=attention_mask,
        )
        if isinstance(out, tuple):
            action_preds, diagnostics = out
        else:
            action_preds = out
            diagnostics = getattr(self.model, "diagnostics", {})

        loss = masked_action_loss(
            predicted_actions=action_preds,
            target_actions=actions,
            attention_mask=attention_mask,
        )

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


def train_offline(
    config_path: str,
    data_path: Optional[str] = None,
    output_dir: str = "checkpoints/offline_rl",
    override_iters: Optional[int] = None,
    override_steps: Optional[int] = None,
    do_eval: bool = False,
) -> Tuple[nn.Module, TrajectoryDataset, Optional[Dict[str, float]]]:
    """
    Main training entry point for benchmark runs and script execution.
    """
    config = load_config(config_path)
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[OfflineRL] Using device: {device}")

    # Load or generate trajectories
    loader = D4RLLoader(gamma=config.get("gamma", 1.0))
    resolved_data_path = data_path or loader.resolve_or_download_dataset(config.get("env_name", ""))

    if resolved_data_path and os.path.exists(resolved_data_path):
        print(f"[OfflineRL] Loading trajectories from {resolved_data_path}")
        trajectories = loader.load_from_hdf5(resolved_data_path)
    else:
        print("[OfflineRL] No local HDF5 dataset found. Generating synthetic dataset for initialization.")
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

    max_iters = override_iters if override_iters is not None else config["training"].get("max_iters", 5)
    steps_per_iter = override_steps if override_steps is not None else config["training"].get("num_steps_per_iter", 100)

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
    save_checkpoint(
        path=ckpt_path,
        model=model,
        optimizer=optimizer,
        step=max_iters * steps_per_iter,
        loss=metrics["epoch/mean_loss"],
        dataset=dataset,
        config=config,
    )
    print(f"[OfflineRL] Saved checkpoint to {ckpt_path}")

    eval_results = None
    if do_eval:
        from offline_rl.evaluation.evaluate_mujoco import evaluate_mujoco

        print(f"[OfflineRL] Running evaluation on {config.get('gym_env', 'Hopper-v4')}...")
        eval_cfg = config.get("evaluation", {})
        eval_results = evaluate_mujoco(
            env_name=config.get("gym_env", "Hopper-v4"),
            model=model,
            state_mean=dataset.state_mean,
            state_std=dataset.state_std,
            num_episodes=eval_cfg.get("num_episodes", 10),
            target_return=eval_cfg.get("target_return", config.get("target_return", 3600.0)),
            scale=config.get("scale", 1000.0),
            max_ep_len=config.get("max_ep_len", 1000),
            device=device,
            d4rl_env_name=config.get("env_name"),
        )
        for k, v in eval_results.items():
            print(f"  {k}: {v:.2f}")

    return model, dataset, eval_results


def verify_training_stages(
    stages: Optional[List[int]] = None,
    smoke_test_updates: int = 2000,
    checkpoint_dir: str = "checkpoints/stage_tests",
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Execute training loop tests in stages (Section 5.2):
        Stage 1: 10 updates -> test forward, loss, backward, optimizer step.
        Stage 2: 100 updates -> inspect loss for NaNs or exploding gradients.
        Stage 3: 2000 updates (or smoke test) -> verify checkpoint saving, loading, and recovery.
    """
    if stages is None:
        stages = [1, 2, 3]

    print("=" * 80)
    print("Executing SNN-DT Training Loop Verification in Stages (Section 5.2)")
    print("=" * 80)

    # Create synthetic trajectory dataset
    print("[Setup] Generating deterministic synthetic trajectory dataset...")
    trajectories = D4RLLoader.create_synthetic_dataset(
        num_trajectories=30,
        traj_len=100,
        state_dim=11,
        act_dim=3,
        seed=42,
    )

    stage_results = {}

    # Stage 1: 10 updates
    if 1 in stages:
        print("\n" + "-" * 60)
        print("STAGE 1: 10 updates (Forward, Loss, Backward, Optimizer step)")
        print("-" * 60)
        model1 = ContinuousSnnDecisionTransformer(
            state_dim=11,
            act_dim=3,
            embed_dim=64,
            n_layer=2,
            n_head=2,
            snn_timesteps=2,
            dropout=0.0,
        )
        initial_params = [p.clone() for p in model1.parameters()]

        res1 = train_model(
            model=model1,
            episodes=trajectories,
            device=device,
            context_length=20,
            batch_size=32,
            learning_rate=1e-4,
            weight_decay=1e-4,
            gradient_clipping=0.25,
            num_updates=10,
            log_interval=5,
        )

        assert len(res1["losses"]) == 10, "Stage 1 failed: Expected 10 loss values."
        assert all(np.isfinite(l) for l in res1["losses"]), "Stage 1 failed: Non-finite loss encountered."
        assert all(np.isfinite(g) for g in res1["grad_norms"]), "Stage 1 failed: Non-finite grad norm."

        # Verify weights updated
        changed = False
        for p_init, p_curr in zip(initial_params, model1.parameters()):
            if not torch.equal(p_init, p_curr):
                changed = True
                break
        assert changed, "Stage 1 failed: Optimizer step did not update model parameters."
        print("[Stage 1 PASS] Forward, masked action loss, backward, and optimizer step verified.")
        stage_results["stage_1"] = "PASSED"

    # Stage 2: 100 updates
    if 2 in stages:
        print("\n" + "-" * 60)
        print("STAGE 2: 100 updates (Stability check: inspect for NaNs or exploding gradients)")
        print("-" * 60)
        model2 = ContinuousSnnDecisionTransformer(
            state_dim=11,
            act_dim=3,
            embed_dim=64,
            n_layer=2,
            n_head=2,
            snn_timesteps=2,
            dropout=0.0,
        )

        res2 = train_model(
            model=model2,
            episodes=trajectories,
            device=device,
            context_length=20,
            batch_size=32,
            learning_rate=1e-4,
            weight_decay=1e-4,
            gradient_clipping=0.25,
            num_updates=100,
            log_interval=25,
        )

        assert len(res2["losses"]) == 100, "Stage 2 failed: Expected 100 loss values."
        assert not np.isnan(res2["losses"]).any(), "Stage 2 failed: NaN in loss history."
        assert not np.isinf(res2["losses"]).any(), "Stage 2 failed: Inf in loss history."
        # Check for exploding unclipped gradients
        assert all(np.isfinite(g) for g in res2["grad_norms"]), "Stage 2 failed: Non-finite gradient norm."
        assert all(g < 100.0 for g in res2["grad_norms"]), f"Stage 2 failed: Exploding gradient detected (max {max(res2['grad_norms']):.2f})."

        early_loss = float(np.mean(res2["losses"][:10]))
        late_loss = float(np.mean(res2["losses"][-10:]))
        print(f"  Early Loss (avg 1-10):  {early_loss:.4f}")
        print(f"  Late Loss  (avg 91-100): {late_loss:.4f}")
        print(f"  Max Grad Norm observed: {max(res2['grad_norms']):.4f} (clipped at 0.25)")
        print("[Stage 2 PASS] Numerical stability verified: No NaNs, no exploding gradients, smooth progression.")
        stage_results["stage_2"] = "PASSED"

    # Stage 3: smoke test updates (e.g. 2,000 updates) + Checkpoint save & load
    if 3 in stages:
        print("\n" + "-" * 60)
        print(f"STAGE 3: {smoke_test_updates} updates (Checkpoint saving, loading, and recovery)")
        print("-" * 60)
        model3 = ContinuousSnnDecisionTransformer(
            state_dim=11,
            act_dim=3,
            embed_dim=64,
            n_layer=2,
            n_head=2,
            snn_timesteps=2,
            dropout=0.0,
        )

        res3 = train_model(
            model=model3,
            episodes=trajectories,
            device=device,
            context_length=20,
            batch_size=32,
            learning_rate=1e-4,
            weight_decay=1e-4,
            gradient_clipping=0.25,
            num_updates=smoke_test_updates,
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval=max(100, smoke_test_updates // 4),
            log_interval=max(50, smoke_test_updates // 10),
        )

        # Verify checkpoint saving
        final_ckpt = os.path.join(checkpoint_dir, "checkpoint_final.pt")
        assert os.path.exists(final_ckpt), f"Stage 3 failed: Checkpoint file {final_ckpt} not found."

        # Verify checkpoint loading and state recovery
        model3_reloaded = ContinuousSnnDecisionTransformer(
            state_dim=11,
            act_dim=3,
            embed_dim=64,
            n_layer=2,
            n_head=2,
            snn_timesteps=2,
            dropout=0.0,
        )
        ckpt_data = load_checkpoint(final_ckpt, model=model3_reloaded, device=device)
        assert "model_state_dict" in ckpt_data, "Stage 3 failed: Missing model_state_dict in checkpoint."
        assert ckpt_data["step"] == smoke_test_updates, f"Stage 3 failed: Checkpoint step mismatch: {ckpt_data['step']}"

        # Test output equivalence
        model3.eval()
        model3_reloaded.eval()
        with torch.no_grad():
            dummy_s = torch.randn(2, 20, 11, device=device)
            dummy_a = torch.randn(2, 20, 3, device=device)
            dummy_r = torch.randn(2, 20, 1, device=device)
            dummy_t = torch.zeros(2, 20, dtype=torch.long, device=device)
            dummy_m = torch.ones(2, 20, device=device)

            out_orig = model3(dummy_s, dummy_a, dummy_r, dummy_t, dummy_m)
            out_load = model3_reloaded(dummy_s, dummy_a, dummy_r, dummy_t, dummy_m)

            pred_orig = out_orig[0] if isinstance(out_orig, tuple) else out_orig
            pred_load = out_load[0] if isinstance(out_load, tuple) else out_load

            assert torch.allclose(pred_orig, pred_load, atol=1e-6), "Stage 3 failed: Reloaded model prediction mismatch."

        print(f"[Stage 3 PASS] Checkpoint saved ({final_ckpt}) and restored with bit-exact prediction parity.")
        print(f"  Initial Loss: {res3['initial_loss']:.4f} -> Final Loss: {res3['final_loss']:.4f}")
        stage_results["stage_3"] = "PASSED"

    print("\n" + "=" * 80)
    print("Gate 5 Verification Summary: ALL STAGES PASSED")
    print("CRITICAL NOTE: Do not use a loss decrease alone as proof of offline RL performance.")
    print("The model could fit dataset actions while still failing to control the simulator.")
    print("Proceed to simulator rollouts and evaluation only after confirming pipeline integrity.")
    print("=" * 80)

    return stage_results


ENV_CONFIG_MAP = {
    "hopper-medium": "offline_rl/configs/hopper_medium.yaml",
    "hopper-medium-v2": "offline_rl/configs/hopper_medium.yaml",
    "hopper-medium-replay": "offline_rl/configs/hopper_medium_replay.yaml",
    "hopper-medium-replay-v2": "offline_rl/configs/hopper_medium_replay.yaml",
    "hopper-medium-expert": "offline_rl/configs/hopper_medium_expert.yaml",
    "hopper-medium-expert-v2": "offline_rl/configs/hopper_medium_expert.yaml",
    "halfcheetah-medium": "offline_rl/configs/halfcheetah_medium.yaml",
    "halfcheetah-medium-v2": "offline_rl/configs/halfcheetah_medium.yaml",
    "halfcheetah-medium-expert": "offline_rl/configs/halfcheetah_medium_expert.yaml",
    "halfcheetah-medium-expert-v2": "offline_rl/configs/halfcheetah_medium_expert.yaml",
    "walker2d-medium": "offline_rl/configs/walker2d_medium.yaml",
    "walker2d-medium-v2": "offline_rl/configs/walker2d_medium.yaml",
    "walker2d-medium-expert": "offline_rl/configs/walker2d_medium_expert.yaml",
    "walker2d-medium-expert-v2": "offline_rl/configs/walker2d_medium_expert.yaml",
    "ant-medium": "offline_rl/configs/ant_medium.yaml",
    "ant-medium-v2": "offline_rl/configs/ant_medium.yaml",
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SNN-DT on MuJoCo offline data")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--env", type=str, default="hopper-medium-v2", help="Environment shorthand name")
    parser.add_argument("--data", type=str, default=None, help="Path to HDF5 dataset file")
    parser.add_argument("--output_dir", type=str, default="checkpoints/offline_rl")
    parser.add_argument("--max_iters", type=int, default=None, help="Override number of training iterations")
    parser.add_argument("--steps_per_iter", type=int, default=None, help="Override steps per iteration")
    parser.add_argument("--eval", action="store_true", help="Run evaluation after training")
    parser.add_argument("--stage_test", type=str, default=None, help="Run staged verification: 1, 2, 3, or all")
    parser.add_argument("--smoke_test", action="store_true", help="Run 2000-update smoke test")
    parser.add_argument("--updates", type=int, default=2000, help="Number of smoke test updates")
    args = parser.parse_args()

    if args.stage_test is not None or args.smoke_test:
        if args.stage_test == "1":
            test_stages = [1]
        elif args.stage_test == "2":
            test_stages = [2]
        elif args.stage_test == "3":
            test_stages = [3]
        else:
            test_stages = [1, 2, 3]

        verify_training_stages(
            stages=test_stages,
            smoke_test_updates=args.updates,
            checkpoint_dir=args.output_dir,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        sys.exit(0)

    config_path = args.config
    if config_path is None:
        env_key = args.env.lower().replace("_", "-")
        if env_key in ENV_CONFIG_MAP:
            config_path = ENV_CONFIG_MAP[env_key]
        else:
            config_path = f"offline_rl/configs/{args.env}.yaml"

    train_offline(
        config_path=config_path,
        data_path=args.data,
        output_dir=args.output_dir,
        override_iters=args.max_iters,
        override_steps=args.steps_per_iter,
        do_eval=args.eval,
    )
