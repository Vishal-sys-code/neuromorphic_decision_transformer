"""
Phase 2: Diagnose Continuous Action Prediction for SNN-DT.

Systematic diagnosis addressing:
1. Predicted versus dataset action distributions (mean, std, ranges).
2. Action saturation at boundaries (-1.0 and +1.0) due to Tanh bounding.
3. Per-action-dimension MSE (thigh, leg, foot torque motors).
4. State normalization integrity and out-of-distribution rollout drift.
5. Return-to-Go (RTG) conditioning fidelity across target returns.
6. Gradient norm decomposition across spike pathways vs dense action head.

Determines whether poor returns stem from representation quality, optimization dynamics,
or rollout distribution shift.
"""

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from offline_rl.datasets.d4rl_loader import D4RLLoader
from offline_rl.datasets.trajectory_dataset import TrajectoryDataset
from offline_rl.evaluation.evaluate_mujoco import evaluate_episode
from offline_rl.models.snn_dt_mujoco import (
    ContinuousSnnDecisionTransformer,
    create_ablation_model,
)
from offline_rl.training.train_offline import masked_action_loss


@dataclass
class ActionDistributionStats:
    dim: int
    data_mean: float
    data_std: float
    data_min: float
    data_max: float
    pred_mean: float
    pred_std: float
    pred_min: float
    pred_max: float
    saturation_pct_pos: float  # pred >= +0.95
    saturation_pct_neg: float  # pred <= -0.95
    dim_mse: float


@dataclass
class GradientNormBreakdown:
    embedding_norm: float
    phase_encoder_norm: float
    spiking_attn_norm: float
    dendritic_router_norm: float
    action_head_norm: float
    total_norm: float


@dataclass
class DiagnosticReport:
    checkpoint_path: str
    total_val_mse: float
    per_dim_stats: List[ActionDistributionStats]
    total_action_saturation_pct: float
    gradient_breakdown: GradientNormBreakdown
    rtg_correlation_r: float
    rtg_target_returns: List[float]
    rtg_achieved_returns: List[float]
    state_norm_drift_max: float
    diagnosis_verdict: str


def run_action_diagnostics(
    checkpoint_path: str,
    data_path: str = "data/d4rl_raw/hopper_medium-v2.hdf5",
    gym_env: str = "Hopper-v4",
    device: str = "cpu",
    batch_size: int = 64,
    num_eval_episodes: int = 5,
    output_dir: str = "results/diagnostics",
) -> DiagnosticReport:
    """
    Performs comprehensive diagnostic analysis on a given SNN-DT checkpoint.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print(f"RUNNING ACTION PREDICTION DIAGNOSTICS")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Environment: {gym_env} | Device: {device}")
    print("=" * 80)

    # 1. Load dataset
    loader = D4RLLoader()
    resolved_path = loader.resolve_or_download_dataset("hopper-medium-v2", auto_download=False)
    if not resolved_path or not os.path.exists(resolved_path):
        resolved_path = data_path

    trajs = loader.load_from_hdf5(resolved_path, max_state_dim=11)
    state_mean, state_std = loader.compute_state_statistics(trajs)

    val_ds = TrajectoryDataset(
        trajectories=trajs[-150:],  # Use last 150 trajectories for validation
        context_length=20,
        state_mean=state_mean,
        state_std=state_std,
        scale=1000.0,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 2. Instantiate and load model
    state_dim = 11
    act_dim = 3

    model = ContinuousSnnDecisionTransformer(
        state_dim=state_dim,
        action_dim=act_dim,
        embed_dim=128,
        n_layer=3,
        n_head=4,
        snn_timesteps=4,
        dropout=0.0,
    )

    if os.path.exists(checkpoint_path) and os.path.getsize(checkpoint_path) > 0:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
        elif isinstance(ckpt, dict):
            model.load_state_dict(ckpt, strict=False)
        print(f"[Diagnostics] Successfully loaded checkpoint weights.")
    else:
        print(f"[Diagnostics Warning] Checkpoint {checkpoint_path} not found. Analyzing baseline initialization.")

    model.to(device)

    # 3. Validation action prediction and distribution inspection
    model.eval()
    all_target_actions = []
    all_predicted_actions = []
    max_eval_batches = 30

    with torch.no_grad():
        for b_idx, batch in enumerate(val_loader):
            if b_idx >= max_eval_batches:
                break
            states = batch["states"].to(device)
            actions = batch["actions"].to(device)
            rtg = batch["returns_to_go"].to(device)
            timesteps = batch["timesteps"].to(device)
            mask = batch["attention_mask"].to(device)

            if hasattr(model, "reset_state"):
                model.reset_state()

            out = model(states=states, actions=actions, returns_to_go=rtg, timesteps=timesteps, attention_mask=mask)
            preds = out[0] if isinstance(out, tuple) else out

            # Extract valid masked tokens
            mask_bool = (mask > 0.5).cpu()
            valid_targets = actions.cpu()[mask_bool].numpy()
            valid_preds = preds.cpu()[mask_bool].numpy()

            all_target_actions.append(valid_targets)
            all_predicted_actions.append(valid_preds)

    targets = np.concatenate(all_target_actions, axis=0)
    preds = np.concatenate(all_predicted_actions, axis=0)

    total_mse = float(np.mean((preds - targets) ** 2))

    per_dim_stats: List[ActionDistributionStats] = []
    saturated_count = 0
    total_elements = preds.size

    for d in range(act_dim):
        t_d = targets[:, d]
        p_d = preds[:, d]

        pos_sat = float(np.mean(p_d >= 0.95) * 100.0)
        neg_sat = float(np.mean(p_d <= -0.95) * 100.0)
        saturated_count += int(np.sum(p_d >= 0.95) + np.sum(p_d <= -0.95))
        d_mse = float(np.mean((p_d - t_d) ** 2))

        stat = ActionDistributionStats(
            dim=d,
            data_mean=float(np.mean(t_d)),
            data_std=float(np.std(t_d)),
            data_min=float(np.min(t_d)),
            data_max=float(np.max(t_d)),
            pred_mean=float(np.mean(p_d)),
            pred_std=float(np.std(p_d)),
            pred_min=float(np.min(p_d)),
            pred_max=float(np.max(p_d)),
            saturation_pct_pos=pos_sat,
            saturation_pct_neg=neg_sat,
            dim_mse=d_mse,
        )
        per_dim_stats.append(stat)

    total_sat_pct = float(saturated_count / max(1, total_elements) * 100.0)

    # 4. Gradient Norm Breakdown across pathways
    model.train()
    model.zero_grad()
    if hasattr(model, "reset_state"):
        model.reset_state()

    sample_batch = next(iter(val_loader))
    s_b = sample_batch["states"].to(device)
    a_b = sample_batch["actions"].to(device)
    r_b = sample_batch["returns_to_go"].to(device)
    t_b = sample_batch["timesteps"].to(device)
    m_b = sample_batch["attention_mask"].to(device)

    out_p = model(states=s_b, actions=a_b, returns_to_go=r_b, timesteps=t_b, attention_mask=m_b)
    preds_p = out_p[0] if isinstance(out_p, tuple) else out_p
    loss = masked_action_loss(preds_p, a_b, m_b)
    loss.backward()

    def get_group_norm(keywords: List[str]) -> float:
        sq_sum = 0.0
        for name, p in model.named_parameters():
            if p.grad is not None and any(kw in name for kw in keywords):
                sq_sum += float(p.grad.norm().item()) ** 2
        return math.sqrt(sq_sum)

    embed_norm = get_group_norm(["embed", "state_embedding", "action_embedding", "return_embedding"])
    phase_norm = get_group_norm(["phase_encoder"])
    attn_norm = get_group_norm(["q_proj", "k_proj", "v_proj"])
    router_norm = get_group_norm(["router"])
    head_norm = get_group_norm(["action_head", "head"])
    total_g_norm = math.sqrt(sum(float(p.grad.norm().item()) ** 2 for p in model.parameters() if p.grad is not None))

    grad_breakdown = GradientNormBreakdown(
        embedding_norm=embed_norm,
        phase_encoder_norm=phase_norm,
        spiking_attn_norm=attn_norm,
        dendritic_router_norm=router_norm,
        action_head_norm=head_norm,
        total_norm=total_g_norm,
    )

    # 5. RTG Conditioning Fidelity & Rollout Observation Drift
    target_returns = [1000.0, 1800.0, 2400.0, 3000.0, 3600.0]
    achieved_returns = []
    max_drift = 0.0

    env = gym.make(gym_env)
    for tgt in target_returns:
        ep_res = evaluate_episode(
            env=env,
            model=model,
            state_mean=state_mean,
            state_std=state_std,
            target_return=tgt,
            scale=1000.0,
            device=device,
            seed=42,
            action_clip=True,
        )
        achieved_returns.append(ep_res.total_reward)
        # Check normalization drift in rollout states
        if ep_res.trajectory is not None and len(ep_res.trajectory.states) > 0:
            rollout_s = np.array(ep_res.trajectory.states)
            normed_s = (rollout_s - state_mean) / np.where(state_std > 1e-6, state_std, 1.0)
            cur_drift = float(np.max(np.abs(normed_s)))
            if cur_drift > max_drift:
                max_drift = cur_drift
    env.close()

    # Calculate Pearson correlation r between target return and achieved return
    x = np.array(target_returns)
    y = np.array(achieved_returns)
    if np.std(x) > 1e-6 and np.std(y) > 1e-6:
        r_corr = float(np.corrcoef(x, y)[0, 1])
    else:
        r_corr = 0.0

    # 6. Diagnosis verdict
    verdict_reasons = []
    if total_sat_pct > 25.0:
        verdict_reasons.append(f"High action saturation ({total_sat_pct:.1f}% >= 0.95) indicates Tanh bounding bottleneck.")
    if r_corr < 0.3:
        verdict_reasons.append(f"Low RTG correlation (r={r_corr:.3f}) shows policy is failing to condition on return-to-go.")
    if router_norm < 1e-4:
        verdict_reasons.append("Vanishing gradients in dendritic router MLP suggests gating weights are starved of gradient flow.")
    if max_drift > 8.0:
        verdict_reasons.append(f"Severe state distribution drift during rollout (max normalized obs = {max_drift:.1f} sigma).")

    if not verdict_reasons:
        verdict = "Action representation and optimization dynamics appear healthy. Increase update steps."
    else:
        verdict = " | ".join(verdict_reasons)

    report = DiagnosticReport(
        checkpoint_path=checkpoint_path,
        total_val_mse=total_mse,
        per_dim_stats=per_dim_stats,
        total_action_saturation_pct=total_sat_pct,
        gradient_breakdown=grad_breakdown,
        rtg_correlation_r=r_corr,
        rtg_target_returns=target_returns,
        rtg_achieved_returns=achieved_returns,
        state_norm_drift_max=max_drift,
        diagnosis_verdict=verdict,
    )

    # Save Markdown report
    md_lines = [
        f"# Phase 2: Action Prediction Diagnostics Report",
        f"**Checkpoint**: `{checkpoint_path}`\n",
        f"### 1. Overall Metrics:",
        f"- **Action MSE**: {total_mse:.4f}",
        f"- **Action Saturation (|a| >= 0.95)**: {total_sat_pct:.2f}%",
        f"- **RTG Conditioning Correlation (r)**: {r_corr:.3f}",
        f"- **Max Rollout State Drift**: {max_drift:.2f} σ\n",
        f"### 2. Per-Dimension Action Prediction Breakdown:",
        f"| Dim | Joint Name | Data Mean ± Std | Pred Mean ± Std | Dim MSE | Saturation (+ / -) |",
        f"| :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    joint_names = ["Thigh Joint", "Leg Joint", "Foot Joint"]
    for s in per_dim_stats:
        j_name = joint_names[s.dim] if s.dim < len(joint_names) else f"Dim {s.dim}"
        md_lines.append(
            f"| {s.dim} | {j_name} | {s.data_mean:.2f} ± {s.data_std:.2f} | "
            f"{s.pred_mean:.2f} ± {s.pred_std:.2f} | {s.dim_mse:.4f} | "
            f"+{s.saturation_pct_pos:.1f}% / -{s.saturation_pct_neg:.1f}% |"
        )

    md_lines.extend([
        f"\n### 3. Gradient Norm Breakdown Across Pathways:",
        f"| Pathway | Gradient Norm || Total Norm Contribution |",
        f"| :--- | :---: | :--- | :---: |",
        f"| **Embedding Layers** | {grad_breakdown.embedding_norm:.4f} | | {grad_breakdown.embedding_norm / max(1e-6, grad_breakdown.total_norm) * 100.0:.1f}% |",
        f"| **Phase Spike Encoder** | {grad_breakdown.phase_encoder_norm:.4f} | | {grad_breakdown.phase_encoder_norm / max(1e-6, grad_breakdown.total_norm) * 100.0:.1f}% |",
        f"| **Spiking Attention (Q, K, V)** | {grad_breakdown.spiking_attn_norm:.4f} | | {grad_breakdown.spiking_attn_norm / max(1e-6, grad_breakdown.total_norm) * 100.0:.1f}% |",
        f"| **Dendritic Router MLP** | {grad_breakdown.dendritic_router_norm:.4f} | | {grad_breakdown.dendritic_router_norm / max(1e-6, grad_breakdown.total_norm) * 100.0:.1f}% |",
        f"| **Action Head (W_o)** | {grad_breakdown.action_head_norm:.4f} | | {grad_breakdown.action_head_norm / max(1e-6, grad_breakdown.total_norm) * 100.0:.1f}% |",
        f"| **Total Gradient Norm** | **{grad_breakdown.total_norm:.4f}** | | **100.0%** |\n",
        f"### 4. Return-to-Go (RTG) Response Profile:",
        f"| Target Return (G) | " + " | ".join(f"{t:.0f}" for t in target_returns) + " | Tracking Correlation (r) |",
        f"| :--- | " + " | ".join(":---:" for _ in target_returns) + " | :---: |",
        f"| **Achieved Return** | " + " | ".join(f"{a:.1f}" for a in achieved_returns) + f" | **{r_corr:.3f}** |\n",
        f"### 5. Diagnostic Verdict:",
        f"> **{verdict}**",
    ])

    md_path = os.path.join(output_dir, "action_diagnostics_report.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")

    json_path = os.path.join(output_dir, "action_diagnostics_report.json")
    with open(json_path, "w") as f:
        json.dump(asdict(report), f, indent=2)

    print("\n" + "\n".join(md_lines))
    print(f"\nDiagnostic reports saved to:\n  - {md_path}\n  - {json_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: Action Prediction Diagnostics")
    parser.add_argument("--checkpoint-path", type=str, default="checkpoints/offline_rl/hopper-medium-v2_snn_dt.pt")
    parser.add_argument("--data-path", type=str, default="data/d4rl_raw/hopper_medium-v2.hdf5")
    parser.add_argument("--gym-env", type=str, default="Hopper-v4")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default="results/diagnostics")
    args = parser.parse_args()

    run_action_diagnostics(
        checkpoint_path=args.checkpoint_path,
        data_path=args.data_path,
        gym_env=args.gym_env,
        device=args.device,
        output_dir=args.output_dir,
    )
