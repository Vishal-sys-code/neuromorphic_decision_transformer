"""
Controlled Ablation Experiment Runner: Beating Decision SpikeFormer PSSA (CVPR 2025).

Executes the official experiment matrix on Hopper-medium:
- E0: SNN-DT Baseline (Corrected pipeline)
- E1: SNN-DT + Local Spiking Attention (PSSA, window S=8)
- E2: SNN-DT + Temporal Spiking Attention (TSSA, T-concatenation)
- E3: SNN-DT + PSSA + TSSA (Dual-branch, uniform fusion)
- E4: E3 + Dendritic Routing (Learned adaptive gating)
- E5: E4 + PTBN (Progressive Threshold-dependent Batch Normalization)
- E6: Best model + Adaptive V_th

Evaluates across the Target-Return Diagnostic Grid:
    G_0 in {1800, 2400, 3000, 3600}
Tracks: Normalized Score, Return, Ep Length, State Drift (sigma), Saturation, and RTG correlation (r).
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
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from offline_rl.datasets.d4rl_loader import D4RLLoader, Trajectory
from offline_rl.datasets.trajectory_dataset import TrajectoryDataset
from offline_rl.evaluation.evaluate_mujoco import compute_d4rl_normalized_score, evaluate_episode
from offline_rl.models.snn_dt_h import SNNDTH, create_snndt_h_ablation
from offline_rl.training.train_offline import masked_action_loss, save_checkpoint


@dataclass
class TargetReturnDiagnostic:
    target_return: float
    mean_return: float
    std_return: float
    normalized_score: float
    mean_ep_len: float
    termination_reasons: Dict[str, int]
    action_saturation_pct: float
    max_state_drift_sigma: float


@dataclass
class AblationExperimentResult:
    ablation_id: str
    architecture_name: str
    seed: int
    updates: int
    val_action_mse: float
    mean_spike_rate: float
    target_return_diagnostics: List[TargetReturnDiagnostic]
    rtg_correlation_r: float
    best_target_return: float
    best_normalized_score: float
    checkpoint_path: str
    passed_pssa_target: bool  # True if score > 74.1 on Hopper-medium


def load_hopper_dataset(
    env_name: str = "hopper-medium-v2",
    data_path: str = "data/d4rl_raw/hopper_medium-v2.hdf5",
    train_split: float = 0.85,
    context_length: int = 20,
    seed: int = 42,
) -> Tuple[TrajectoryDataset, TrajectoryDataset, np.ndarray, np.ndarray]:
    """Load Hopper dataset with train/validation splits."""
    loader = D4RLLoader()
    resolved = loader.resolve_or_download_dataset(env_name, auto_download=False)
    if not resolved or not os.path.exists(resolved):
        if os.path.exists(data_path):
            resolved = data_path
        else:
            print(f"[Dataset] Attempting auto-download of {env_name}...")
            resolved = loader.resolve_or_download_dataset(env_name, auto_download=True)

    if resolved and os.path.exists(resolved):
        trajectories = loader.load_from_hdf5(resolved, max_state_dim=11)
    else:
        print("[Dataset Warning] Real D4RL HDF5 not found. Using synthetic trajectory dataset for smoke testing.")
        trajectories = loader.create_synthetic_dataset(num_trajectories=50, traj_len=200, state_dim=11, act_dim=3)

    state_mean, state_std = loader.compute_state_statistics(trajectories)

    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(trajectories))
    split = int(len(trajectories) * train_split)
    train_trajs = [trajectories[i] for i in indices[:split]]
    val_trajs = [trajectories[i] for i in indices[split:]]

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


def evaluate_target_grid(
    model: SNNDTH,
    gym_env: str,
    d4rl_name: str,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    target_grid: Sequence[float] = (1800.0, 2400.0, 3000.0, 3600.0),
    eval_seeds: Sequence[int] = (42, 100, 200),
    device: str = "cpu",
    scale: float = 1000.0,
) -> Tuple[List[TargetReturnDiagnostic], float, float, float]:
    """
    Evaluates policy across the diagnostic return grid.
    Returns:
        diagnostics_list, best_target, best_score, rtg_correlation_r
    """
    model.eval()
    results = []
    achieved_returns_for_corr = []

    env = gym.make(gym_env)
    safe_std = np.where(state_std > 1e-6, state_std, 1.0)

    for target in target_grid:
        ep_returns = []
        ep_lens = []
        term_reasons = {"fall": 0, "timeout": 0}
        total_actions = 0
        saturated_actions = 0
        max_drift = 0.0

        for s in eval_seeds:
            res = evaluate_episode(
                env=env,
                model=model,
                state_mean=state_mean,
                state_std=state_std,
                target_return=target,
                scale=scale,
                device=device,
                seed=s,
                action_clip=True,
                record_trajectory=True,
            )
            ep_returns.append(res.total_reward)
            ep_lens.append(res.episode_length)

            if res.episode_length >= 1000:
                term_reasons["timeout"] += 1
            else:
                term_reasons["fall"] += 1

            if res.trajectory is not None and len(res.trajectory.actions) > 0:
                acts = res.trajectory.actions
                total_actions += acts.size
                saturated_actions += int(np.sum(np.abs(acts) >= 0.95))

                normed_s = (res.trajectory.states - state_mean) / safe_std
                drift = float(np.max(np.abs(normed_s)))
                if drift > max_drift:
                    max_drift = drift

        mean_ret = float(np.mean(ep_returns))
        std_ret = float(np.std(ep_returns))
        mean_len = float(np.mean(ep_lens))
        sat_pct = float(saturated_actions / max(1, total_actions) * 100.0)

        try:
            norm_score = compute_d4rl_normalized_score(mean_ret, d4rl_name)
        except Exception:
            norm_score = 0.0

        diag = TargetReturnDiagnostic(
            target_return=float(target),
            mean_return=mean_ret,
            std_return=std_ret,
            normalized_score=norm_score,
            mean_ep_len=mean_len,
            termination_reasons=term_reasons,
            action_saturation_pct=sat_pct,
            max_state_drift_sigma=max_drift,
        )
        results.append(diag)
        achieved_returns_for_corr.append(mean_ret)

    env.close()

    # RTG Correlation
    x = np.array(target_grid)
    y = np.array(achieved_returns_for_corr)
    if np.std(x) > 1e-6 and np.std(y) > 1e-6:
        r_corr = float(np.corrcoef(x, y)[0, 1])
    else:
        r_corr = 0.0

    # Best target return
    best_idx = int(np.argmax([d.normalized_score for d in results]))
    best_target = results[best_idx].target_return
    best_score = results[best_idx].normalized_score

    return results, best_target, best_score, r_corr


def run_ablation_experiment(
    ablation_id: str,
    updates: int = 25000,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    clip_val: float = 0.25,
    seed: int = 42,
    device: str = "cpu",
    output_dir: str = "results/pssa_ablations",
    checkpoint_dir: str = "checkpoints/pssa_ablations",
    target_grid: Sequence[float] = (1800.0, 2400.0, 3000.0, 3600.0),
    eval_seeds: Sequence[int] = (42, 100, 200),
    data_path: str = "data/d4rl_raw/hopper_medium-v2.hdf5",
    gym_env: str = "Hopper-v4",
    d4rl_name: str = "hopper-medium-v2",
) -> AblationExperimentResult:
    """
    Executes a complete single ablation run:
    1. Instantiates architecture
    2. Trains for prescribed update budget
    3. Evaluates on multi-target diagnostic grid
    4. Generates formal report
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    arch_names = {
        "e0": "E0: SNN-DT Baseline (Corrected Pipeline)",
        "e1": "E1: SNN-DT + Local Spiking Attention (PSSA, S=8)",
        "e2": "E2: SNN-DT + Temporal Spiking Attention (TSSA, T-concat)",
        "e3": "E3: SNN-DT + PSSA + TSSA (Dual-Branch Uniform)",
        "e4": "E4: SNN-DT-H Dual + Dendritic Routing Fusion",
        "e5": "E5: SNN-DT-H Full (Dual + Routing + PTBN Normalization)",
        "e6": "E6: SNN-DT-H + RTG-Adaptive V_th Modulation",
    }
    arch_title = arch_names.get(ablation_id.lower(), f"Ablation {ablation_id}")

    print("\n" + "=" * 80)
    print(f"RUNNING EXPERIMENT: {arch_title}")
    print(f"Updates: {updates} | Seed: {seed} | Device: {device}")
    print(f"Target Diagnostic Grid: {list(target_grid)}")
    print(f"PSSA Target to Beat: 74.1 on Hopper-medium")
    print("=" * 80)

    # 1. Load data
    train_ds, val_ds, state_mean, state_std = load_hopper_dataset(
        env_name=d4rl_name,
        data_path=data_path,
        train_split=0.85,
        context_length=20,
        seed=seed,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 2. Build model
    model = create_snndt_h_ablation(
        ablation_id=ablation_id,
        state_dim=11,
        action_dim=3,
        hidden_dim=128,
        n_layer=3,
        n_head=4,
        snn_timesteps=4,
        tau=20.0,
        v_th=0.5,
        alpha=25.0,
        window_size=8,
        ptbn_transition_steps=10000,
        dropout=0.1,
    )
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # 3. Train
    model.train()
    loader_iter = iter(train_loader)
    t_start = time.time()
    losses = []

    for step in range(1, updates + 1):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch = next(loader_iter)

        optimizer.zero_grad()
        if hasattr(model, "reset_state"):
            model.reset_state()

        states = batch["states"].to(device)
        actions = batch["actions"].to(device)
        rtg = batch["returns_to_go"].to(device)
        timesteps = batch["timesteps"].to(device)
        mask = batch["attention_mask"].to(device)

        preds = model(
            states=states,
            actions=actions,
            returns_to_go=rtg,
            timesteps=timesteps,
            attention_mask=mask,
            step=step,
        )

        loss = masked_action_loss(preds, actions, mask)
        loss.backward()

        if clip_val > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)

        optimizer.step()
        losses.append(loss.item())

        if step % 1000 == 0 or step == 1 or step == updates:
            elapsed = time.time() - t_start
            sr = model.diagnostics.get("mean_spike_rate", 0.0)
            print(f"[{ablation_id.upper()}] Step [{step:5d}/{updates:5d}] | Loss: {loss.item():.4f} | SpikeRate: {sr:.3f} | Elapsed: {elapsed:.1f}s")

    # 4. Save Checkpoint
    ckpt_path = os.path.join(checkpoint_dir, f"{ablation_id}_seed{seed}_step{updates}.pt")
    save_checkpoint(
        path=ckpt_path,
        model=model,
        optimizer=optimizer,
        step=updates,
        loss=losses[-1],
        extra_state={
            "state_mean": state_mean,
            "state_std": state_std,
            "ablation_id": ablation_id,
        },
    )

    # 5. Validation Action MSE
    model.eval()
    val_losses = []
    with torch.no_grad():
        for i, v_batch in enumerate(val_loader):
            if i >= 30:
                break
            v_s = v_batch["states"].to(device)
            v_a = v_batch["actions"].to(device)
            v_r = v_batch["returns_to_go"].to(device)
            v_t = v_batch["timesteps"].to(device)
            v_m = v_batch["attention_mask"].to(device)
            v_preds = model(v_s, v_a, v_r, v_t, v_m, reset_state=True)
            v_loss = masked_action_loss(v_preds, v_a, v_m)
            val_losses.append(v_loss.item())
    val_mse = float(np.mean(val_losses)) if val_losses else 0.0

    # 6. Evaluate across diagnostic grid
    print(f"\nEvaluating {ablation_id.upper()} across target return grid: {list(target_grid)}...")
    diags, best_target, best_score, r_corr = evaluate_target_grid(
        model=model,
        gym_env=gym_env,
        d4rl_name=d4rl_name,
        state_mean=state_mean,
        state_std=state_std,
        target_grid=target_grid,
        eval_seeds=eval_seeds,
        device=device,
    )

    if "expert" in d4rl_name.lower():
        pssa_target = 110.9
    elif "replay" in d4rl_name.lower():
        pssa_target = 96.3
    else:
        pssa_target = 74.1
    passed = best_score > pssa_target

    result = AblationExperimentResult(
        ablation_id=ablation_id,
        architecture_name=arch_title,
        seed=seed,
        updates=updates,
        val_action_mse=val_mse,
        mean_spike_rate=float(model.diagnostics.get("mean_spike_rate", 0.0)),
        target_return_diagnostics=diags,
        rtg_correlation_r=r_corr,
        best_target_return=best_target,
        best_normalized_score=best_score,
        checkpoint_path=ckpt_path,
        passed_pssa_target=passed,
    )

    # 7. Generate Markdown and JSON reports
    report_json = os.path.join(output_dir, f"{ablation_id}_report.json")
    with open(report_json, "w") as f:
        json.dump(asdict(result), f, indent=2)

    report_md = os.path.join(output_dir, f"{ablation_id}_report.md")
    md_lines = [
        f"# Ablation Report: {arch_title}",
        f"- **Ablation ID**: `{ablation_id.upper()}`",
        f"- **Updates**: {updates} | **Seed**: {seed}",
        f"- **Val Action MSE**: {val_mse:.4f}",
        f"- **Mean Spike Rate**: {result.mean_spike_rate:.4f}",
        f"- **RTG Conditioning Correlation (r)**: {r_corr:.3f}",
        f"- **Best Score**: **{best_score:.2f}** (at Target Return $G_0 = {best_target:.0f}$)",
        f"- **PSSA Baseline Target (74.1)**: {'[EXCEEDED]' if passed else '[NOT EXCEEDED]'}\n",
        f"### Target-Return Diagnostic Grid Breakdown:",
        f"| Target $G_0$ | Mean Return ± Std | Norm Score | Ep Length | Falls / Timeouts | Saturation | Max State Drift |",
        f"| :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for d in diags:
        md_lines.append(
            f"| {d.target_return:.0f} | {d.mean_return:.1f} ± {d.std_return:.1f} | "
            f"**{d.normalized_score:.1f}** | {d.mean_ep_len:.0f} | "
            f"{d.termination_reasons['fall']} / {d.termination_reasons['timeout']} | "
            f"{d.action_saturation_pct:.1f}% | {d.max_state_drift_sigma:.2f}σ |"
        )
    with open(report_md, "w") as f:
        f.write("\n".join(md_lines) + "\n")

    print("\n" + "\n".join(md_lines))
    print(f"\nSaved report to:\n  - {report_md}\n  - {report_json}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SNN-DT-H Ablation Experiments")
    parser.add_argument("--ablation", type=str, default="e1", choices=["e0", "e1", "e2", "e3", "e4", "e5", "e6", "all"])
    parser.add_argument("--updates", type=int, default=25000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default="results/pssa_ablations")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/pssa_ablations")
    parser.add_argument("--data-path", type=str, default="data/d4rl_raw/hopper_medium-v2.hdf5")
    parser.add_argument("--gym-env", type=str, default="Hopper-v4")
    parser.add_argument("--d4rl-name", type=str, default="hopper-medium-v2")
    parser.add_argument("--target-grid", type=float, nargs="+", default=[1800.0, 2400.0, 3000.0, 3600.0])
    args = parser.parse_args()

    if args.ablation == "all":
        matrix = ["e0", "e1", "e2", "e3", "e4", "e5"]
        for ab in matrix:
            run_ablation_experiment(
                ablation_id=ab,
                updates=args.updates,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                seed=args.seed,
                device=args.device,
                output_dir=args.output_dir,
                checkpoint_dir=args.checkpoint_dir,
                data_path=args.data_path,
                gym_env=args.gym_env,
                d4rl_name=args.d4rl_name,
                target_grid=args.target_grid,
            )
    else:
        run_ablation_experiment(
            ablation_id=args.ablation,
            updates=args.updates,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            seed=args.seed,
            device=args.device,
            output_dir=args.output_dir,
            checkpoint_dir=args.checkpoint_dir,
            data_path=args.data_path,
            gym_env=args.gym_env,
            d4rl_name=args.d4rl_name,
            target_grid=args.target_grid,
        )
