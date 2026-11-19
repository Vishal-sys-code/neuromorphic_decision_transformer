"""
Phase 1: Establish a Trustworthy Hopper-medium Baseline.

Rigorous controlled comparison:
1. Full SNN-DT (Phase-shifted spike encoding + Dendritic routing + Vectorized LIF)
2. SNN-DT without Dendritic Routing (Uniform head averaging)

Training sweep checkpoints:
- 1,000 updates (Reference checkpoint budget)
- 5,000 updates (Early learning)
- 10,000 updates (Intermediate learning)
- 25,000 updates (Longer training)

Evaluation Protocol:
- Evaluates each checkpoint on the exact same 15 episodes in Hopper-v4 (fixed seeds)
- Records: Validation Action MSE, Normalized D4RL Return, Mean Spike Rate, Inference Latency,
  and Standardized CMOS 45nm Energy Proxy.
"""

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import gymnasium as gym
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from offline_rl.datasets.d4rl_loader import D4RLLoader, Trajectory
from offline_rl.datasets.trajectory_dataset import TrajectoryDataset
from offline_rl.evaluation.efficiency_metrics import (
    TheoreticalOpsProfile,
    compute_energy_proxy,
    compute_theoretical_ops,
)
from offline_rl.evaluation.evaluate_mujoco import (
    compute_d4rl_normalized_score,
    evaluate_mujoco,
)
from offline_rl.models.snn_dt_mujoco import (
    ContinuousSnnDecisionTransformer,
    create_ablation_model,
)
from offline_rl.training.train_offline import (
    masked_action_loss,
    save_checkpoint,
)


@dataclass
class MilestoneEvaluation:
    update_step: int
    val_loss_mse: float
    mean_return: float
    std_return: float
    normalized_score: float
    mean_spike_rate: float
    inference_latency_ms: float
    full_cmos_energy_uj_per_step: float
    spike_event_energy_pj_per_step: float
    checkpoint_path: str


def load_hopper_dataset(
    data_path: str = "data/d4rl_raw/hopper_medium-v2.hdf5",
    train_split: float = 0.85,
    context_length: int = 20,
    seed: int = 42,
) -> Tuple[TrajectoryDataset, TrajectoryDataset, np.ndarray, np.ndarray]:
    """Loads Hopper-medium D4RL dataset and returns train and validation splits."""
    loader = D4RLLoader()
    resolved_path = loader.resolve_or_download_dataset("hopper-medium-v2", auto_download=False)
    if not resolved_path or not os.path.exists(resolved_path):
        if os.path.exists(data_path):
            resolved_path = data_path
        else:
            print(f"[Phase1] Dataset not found locally. Auto-downloading hopper-medium-v2...")
            resolved_path = loader.resolve_or_download_dataset("hopper-medium-v2", auto_download=True)

    trajectories = loader.load_from_hdf5(resolved_path, max_state_dim=11)
    state_mean, state_std = loader.compute_state_statistics(trajectories)

    np.random.seed(seed)
    indices = np.random.permutation(len(trajectories))
    split_idx = int(len(trajectories) * train_split)
    train_trajs = [trajectories[i] for i in indices[:split_idx]]
    val_trajs = [trajectories[i] for i in indices[split_idx:]]

    train_ds = TrajectoryDataset(
        trajectories=train_trajs,
        context_length=context_length,
        state_mean=state_mean,
        state_std=state_std,
        scale=1000.0,
    )
    val_ds = TrajectoryDataset(
        trajectories=val_trajs,
        context_length=context_length,
        state_mean=state_mean,
        state_std=state_std,
        scale=1000.0,
    )

    return train_ds, val_ds, state_mean, state_std


def evaluate_val_loss(
    model: nn.Module,
    val_loader: DataLoader,
    device: str = "cpu",
    max_batches: int = 50,
) -> float:
    """Computes masked action MSE loss across validation batches."""
    model.eval()
    losses = []
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            states = batch["states"].to(device)
            actions = batch["actions"].to(device)
            rtg = batch["returns_to_go"].to(device)
            timesteps = batch["timesteps"].to(device)
            mask = batch["attention_mask"].to(device)

            if hasattr(model, "reset_state"):
                model.reset_state()

            out = model(
                states=states,
                actions=actions,
                returns_to_go=rtg,
                timesteps=timesteps,
                attention_mask=mask,
            )
            preds = out[0] if isinstance(out, tuple) else out
            loss = masked_action_loss(preds, actions, mask)
            losses.append(loss.item())

    return float(np.mean(losses)) if losses else 0.0


def train_and_evaluate_sweep(
    model_type: str,  # "full" or "ablation_b" (no dendritic routing)
    train_loader: DataLoader,
    val_loader: DataLoader,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    milestones: List[int],
    checkpoint_dir: str,
    device: str = "cpu",
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    clip_val: float = 0.25,
    num_eval_episodes: int = 15,
    eval_seeds: Sequence[int] = (42, 100, 200),
    target_return: float = 3600.0,
    scale: float = 1000.0,
    gym_env: str = "Hopper-v4",
    d4rl_env_name: str = "hopper-medium-v2",
) -> List[MilestoneEvaluation]:
    """
    Trains one architecture through the prescribed update milestones,
    saving checkpoints and evaluating on the identical 15 episodes at each milestone.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    is_routing_enabled = (model_type == "full")
    model_label = "SNN-DT Full" if is_routing_enabled else "SNN-DT (No Routing)"

    print("\n" + "=" * 80)
    print(f"Starting Training: {model_label} | Milestones: {milestones} updates")
    print(f"Device: {device} | LR: {learning_rate} | Clip: {clip_val}")
    print("=" * 80)

    # Initialize model
    model = create_ablation_model(
        ablation_type=model_type,
        state_dim=11,
        act_dim=3,
        embed_dim=128,
        n_layer=3,
        n_head=4,
        snn_timesteps=4,
        dropout=0.1,
    )
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    max_steps = max(milestones)
    evaluations: List[MilestoneEvaluation] = []
    step = 0
    iter_loader = iter(train_loader)
    t_start = time.time()

    model.train()
    while step < max_steps:
        step += 1
        try:
            batch = next(iter_loader)
        except StopIteration:
            iter_loader = iter(train_loader)
            batch = next(iter_loader)

        optimizer.zero_grad()
        if hasattr(model, "reset_state"):
            model.reset_state()

        states = batch["states"].to(device)
        actions = batch["actions"].to(device)
        rtg = batch["returns_to_go"].to(device)
        timesteps = batch["timesteps"].to(device)
        mask = batch["attention_mask"].to(device)

        out = model(
            states=states,
            actions=actions,
            returns_to_go=rtg,
            timesteps=timesteps,
            attention_mask=mask,
        )
        preds = out[0] if isinstance(out, tuple) else out
        loss = masked_action_loss(preds, actions, mask)
        loss.backward()

        if clip_val > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)

        optimizer.step()

        if step % 250 == 0 or step == 1 or step in milestones:
            elapsed = time.time() - t_start
            print(f"[{model_label}] Update [{step:5d}/{max_steps:5d}] | Loss: {loss.item():.4f} | Time: {elapsed:.1f}s")

        # Reached a checkpoint milestone
        if step in milestones:
            ckpt_path = os.path.join(checkpoint_dir, f"{model_type}_step_{step}.pt")
            save_checkpoint(
                path=ckpt_path,
                model=model,
                optimizer=optimizer,
                step=step,
                loss=loss.item(),
                extra_state={"state_mean": state_mean, "state_std": state_std, "model_type": model_type},
            )
            print(f"[{model_label}] Checkpoint saved at step {step}: {ckpt_path}")

            # 1. Evaluate validation loss (action MSE)
            val_loss = evaluate_val_loss(model, val_loader, device=device)

            # 2. Evaluate in Hopper-v4 on identical 15 episodes across fixed seeds
            print(f"[{model_label}] Evaluating step {step} on {num_eval_episodes} episodes in {gym_env}...")
            eval_res = evaluate_mujoco(
                env_name=gym_env,
                model=model,
                state_mean=state_mean,
                state_std=state_std,
                num_episodes=num_eval_episodes,
                target_return=target_return,
                scale=scale,
                device=device,
                deterministic=True,
                d4rl_env_name=d4rl_env_name,
                seeds=eval_seeds,
            )

            # Operations and energy calculations
            mean_sr = float(eval_res.get("mean_spike_rate", 0.0))
            ops = compute_theoretical_ops(state_dim=11, action_dim=3, mean_spike_rate=mean_sr)
            proxy = compute_energy_proxy(ops, episode_length=max(1, int(eval_res.get("episode_length", 1000))))
            full_cmos_uj = float(proxy.snndt_energy_nj_per_step * 1e-3)
            spike_pj = float(eval_res.get("spike_count", 0.0)) * 0.5

            milestone_eval = MilestoneEvaluation(
                update_step=step,
                val_loss_mse=val_loss,
                mean_return=float(eval_res.get("mean_return", 0.0)),
                std_return=float(eval_res.get("std_return", 0.0)),
                normalized_score=float(eval_res.get("normalized_score", 0.0)),
                mean_spike_rate=mean_sr,
                inference_latency_ms=float(eval_res.get("inference_latency_ms", 0.0)),
                full_cmos_energy_uj_per_step=full_cmos_uj,
                spike_event_energy_pj_per_step=spike_pj,
                checkpoint_path=ckpt_path,
            )
            evaluations.append(milestone_eval)

            print(
                f"[{model_label} @ {step:5d}] Val Loss: {val_loss:.4f} | "
                f"Norm Score: {milestone_eval.normalized_score:.2f} | "
                f"Return: {milestone_eval.mean_return:.1f} ± {milestone_eval.std_return:.1f} | "
                f"Spike Rate: {mean_sr:.4f} | "
                f"Latency: {milestone_eval.inference_latency_ms:.2f}ms"
            )
            model.train()

    return evaluations


def run_phase1_baseline_sweep(
    data_path: str = "data/d4rl_raw/hopper_medium-v2.hdf5",
    milestones: Sequence[int] = (1000, 5000, 10000, 25000),
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    device: str = "cpu",
    output_dir: str = "results/phase1",
    checkpoint_dir: str = "checkpoints/phase1",
    num_eval_episodes: int = 15,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Executes Phase 1: Establishes trustworthy Hopper-medium baseline sweep
    comparing Full SNN-DT vs SNN-DT without Dendritic Routing.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("=" * 80)
    print("PHASE 1: ESTABLISH TRUSTWORTHY HOPPER-MEDIUM BASELINE")
    print(f"Sweeping milestones: {list(milestones)} updates")
    print(f"Evaluation episodes: {num_eval_episodes} (fixed seeds 42, 100, 200)")
    print("=" * 80)

    # 1. Load dataset
    train_ds, val_ds, state_mean, state_std = load_hopper_dataset(
        data_path=data_path,
        train_split=0.85,
        context_length=20,
        seed=seed,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 2. Train Full SNN-DT across milestones
    full_evals = train_and_evaluate_sweep(
        model_type="full",
        train_loader=train_loader,
        val_loader=val_loader,
        state_mean=state_mean,
        state_std=state_std,
        milestones=list(milestones),
        checkpoint_dir=os.path.join(checkpoint_dir, "full"),
        device=device,
        learning_rate=learning_rate,
        num_eval_episodes=num_eval_episodes,
    )

    # 3. Train SNN-DT without Dendritic Routing (Ablation B) across milestones
    no_router_evals = train_and_evaluate_sweep(
        model_type="ablation_b",
        train_loader=train_loader,
        val_loader=val_loader,
        state_mean=state_mean,
        state_std=state_std,
        milestones=list(milestones),
        checkpoint_dir=os.path.join(checkpoint_dir, "no_dendritic_routing"),
        device=device,
        learning_rate=learning_rate,
        num_eval_episodes=num_eval_episodes,
    )

    # 4. Generate comparison tables
    report_dict = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "milestones": list(milestones),
        "dataset": "hopper-medium-v2",
        "reference_scores": {
            "DecisionTransformer_Chen2021": 67.6,
            "DecisionSpikeFormer_CVPR2025": 63.5,
        },
        "full_snndt": [asdict(e) for e in full_evals],
        "no_dendritic_routing": [asdict(e) for e in no_router_evals],
    }

    # Save JSON report
    json_path = os.path.join(output_dir, "phase1_hopper_baseline_report.json")
    with open(json_path, "w") as f:
        json.dump(report_dict, f, indent=2)

    # Save Markdown Report
    md_lines = [
        "# Phase 1: Hopper-medium Controlled Baseline Sweep\n",
        "### Reference Baselines on Hopper-medium (D4RL-v2):",
        "- **Decision Transformer (Dense ANN, Chen et al. 2021)**: 67.6 ± 1.0",
        "- **DecisionSpikeFormer (Spiking SNN, CVPR 2025)**: 63.5 ± 28.4\n",
        "### Empirical Milestone Progression across Optimizer Updates:\n",
        "| Architecture | Updates | Val Loss (MSE) | Mean Return | Norm Score | Spike Rate | Latency | CMOS Energy / Step |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for e in full_evals:
        md_lines.append(
            f"| **SNN-DT Full** | {e.update_step:,} | {e.val_loss_mse:.4f} | "
            f"{e.mean_return:.1f} ± {e.std_return:.1f} | **{e.normalized_score:.1f}** | "
            f"{e.mean_spike_rate:.4f} | {e.inference_latency_ms:.2f} ms | ~{e.full_cmos_energy_uj_per_step:.1f} µJ |"
        )

    for e in no_router_evals:
        md_lines.append(
            f"| **SNN-DT (No Routing)** | {e.update_step:,} | {e.val_loss_mse:.4f} | "
            f"{e.mean_return:.1f} ± {e.std_return:.1f} | **{e.normalized_score:.1f}** | "
            f"{e.mean_spike_rate:.4f} | {e.inference_latency_ms:.2f} ms | ~{e.full_cmos_energy_uj_per_step:.1f} µJ |"
        )

    md_lines.append("\n### Key Empirical Findings:")
    md_lines.append("1. **Convergence**: Did longer training (10k, 25k) close the gap with DecisionSpikeFormer (63.5)?")
    md_lines.append("2. **Dendritic Routing Impact**: Does learned head gating outperform uniform head averaging as update steps increase, or does routing introduce policy optimization interference?")
    md_lines.append("3. **Energy Profile**: Audited CMOS 45nm energy proxy consistently accounts for dense projections (~339 µJ) and spiking attention additions.")

    md_path = os.path.join(output_dir, "phase1_hopper_baseline_table.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")

    print("\n" + "=" * 80)
    print("PHASE 1 BASELINE SWEEP COMPLETE:")
    print("=" * 80)
    print("\n".join(md_lines))
    print(f"\nSaved reports to:\n  - {json_path}\n  - {md_path}")

    return report_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1: Hopper Baseline Controlled Sweep")
    parser.add_argument("--data-path", type=str, default="data/d4rl_raw/hopper_medium-v2.hdf5")
    parser.add_argument("--milestones", type=int, nargs="+", default=[1000, 5000, 10000, 25000])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="results/phase1")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/phase1")
    parser.add_argument("--num-eval-episodes", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true", help="Run 10 and 20 updates for testing")
    args = parser.parse_args()

    milestones = [10, 20] if args.smoke_test else args.milestones
    eval_eps = 2 if args.smoke_test else args.num_eval_episodes

    run_phase1_baseline_sweep(
        data_path=args.data_path,
        milestones=milestones,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        num_eval_episodes=eval_eps,
        seed=args.seed,
    )
