"""
Unit and integration tests for offline RL training loop (Phase 5 / Gate 5).

Verifies Gate 5 criteria:
1. Masked action regression loss matches mathematical specification:
   L_action = sum_{b,t} m_{b,t} || hat{a}_{b,t} - a_{b,t} ||_2^2 / (sum_{b,t} m_{b,t} * d_a)
2. Padding tokens (m_{b,t} = 0) do not contribute to loss or gradient flow.
3. Stage 1 (10 updates): forward, loss, backward, and optimizer step work.
4. Stage 2 (100 updates): inspect loss for NaNs or exploding gradients.
5. Stage 3: checkpoint saving, loading, and bit-exact recovery.
6. Training progression: loss decreases on training trajectories.
"""

import os
import shutil
import tempfile
import numpy as np
import pytest
import torch
import torch.nn as nn

from offline_rl.datasets.d4rl_loader import D4RLLoader
from offline_rl.datasets.trajectory_dataset import TrajectoryDataset
from offline_rl.models.snn_dt_mujoco import ContinuousSnnDecisionTransformer
from offline_rl.training.train_offline import (
    OfflineTrainer,
    load_checkpoint,
    masked_action_loss,
    save_checkpoint,
    train_model,
    train_offline,
    verify_training_stages,
)


@pytest.fixture
def synthetic_episodes():
    """Deterministic synthetic episodes fixture."""
    return D4RLLoader.create_synthetic_dataset(
        num_trajectories=15,
        traj_len=60,
        state_dim=11,
        act_dim=3,
        seed=42,
    )


@pytest.fixture
def small_model():
    """Compact SNN-DT model for fast testing."""
    return ContinuousSnnDecisionTransformer(
        state_dim=11,
        act_dim=3,
        embed_dim=64,
        n_layer=2,
        n_head=2,
        snn_timesteps=2,
        dropout=0.0,
    )


def test_masked_action_loss_exact_computation():
    """Verify masked_action_loss matches analytical formula and dimension normalization."""
    B, K, act_dim = 2, 3, 2
    pred = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[0.5, 1.5], [2.5, 3.5], [4.5, 5.5]],
        ],
        dtype=torch.float32,
    )
    target = torch.zeros((B, K, act_dim), dtype=torch.float32)
    # Mask: batch 0 has 2 valid steps, batch 1 has 3 valid steps -> total valid steps = 5
    mask = torch.tensor(
        [
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )

    # Valid squared errors:
    # Batch 0 step 1: 3^2 + 4^2 = 9 + 16 = 25
    # Batch 0 step 2: 5^2 + 6^2 = 25 + 36 = 61
    # Batch 1 step 0: 0.5^2 + 1.5^2 = 0.25 + 2.25 = 2.5
    # Batch 1 step 1: 2.5^2 + 3.5^2 = 6.25 + 12.25 = 18.5
    # Batch 1 step 2: 4.5^2 + 5.5^2 = 20.25 + 30.25 = 50.5
    # Sum valid squared error = 25 + 61 + 2.5 + 18.5 + 50.5 = 157.5
    # Denominator = 5 valid steps * 2 act_dim = 10.0
    # Expected loss = 157.5 / 10.0 = 15.75

    loss = masked_action_loss(pred, target, mask)
    assert torch.isclose(loss, torch.tensor(15.75), atol=1e-5)


def test_masked_action_loss_padding_invariance():
    """Verify that values in padded positions do NOT affect loss or gradients."""
    B, K, act_dim = 2, 4, 3
    pred = torch.randn(B, K, act_dim, requires_grad=True)
    target = torch.randn(B, K, act_dim)
    mask = torch.tensor(
        [
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )

    loss1 = masked_action_loss(pred, target, mask)
    loss1.backward()
    grad1 = pred.grad.clone()

    # Modify padded positions in pred and target
    pred_modified = pred.detach().clone()
    pred_modified[:, :1, :] += 100.0  # modifies only masked-out steps
    pred_modified.requires_grad_(True)

    target_modified = target.clone()
    target_modified[:, :1, :] -= 500.0

    loss2 = masked_action_loss(pred_modified, target_modified, mask)
    loss2.backward()
    grad2 = pred_modified.grad.clone()

    assert torch.isclose(loss1, loss2, atol=1e-6)
    # Gradients on padded steps must be exactly 0
    assert torch.all(grad1[0, :2, :] == 0.0)
    assert torch.all(grad1[1, :1, :] == 0.0)
    # Active gradients must be identical
    assert torch.allclose(grad1, grad2, atol=1e-6)


def test_masked_action_loss_edge_cases():
    """Verify zero error and zero-mask clamp_min stability."""
    pred = torch.randn(2, 5, 3)
    mask = torch.ones(2, 5)
    # Perfect prediction -> 0 loss
    loss_zero = masked_action_loss(pred, pred, mask)
    assert torch.isclose(loss_zero, torch.tensor(0.0), atol=1e-6)

    # All-zero mask -> clamp_min(1.0) prevents NaN or division by zero
    zero_mask = torch.zeros(2, 5)
    loss_empty = masked_action_loss(pred, pred + 1.0, zero_mask)
    assert torch.isclose(loss_empty, torch.tensor(0.0), atol=1e-6)
    assert not torch.isnan(loss_empty)


def test_stage_1_ten_updates_step_works(synthetic_episodes, small_model):
    """
    Stage 1 (10 updates):
    Test that forward, loss, backward, and optimizer step work.
    """
    initial_weights = [p.clone() for p in small_model.parameters()]

    history = train_model(
        model=small_model,
        episodes=synthetic_episodes,
        device="cpu",
        context_length=20,
        batch_size=16,
        learning_rate=1e-4,
        weight_decay=1e-4,
        gradient_clipping=0.25,
        num_updates=10,
        log_interval=5,
    )

    losses = history["losses"]
    grad_norms = history["grad_norms"]

    # 1. Forward, loss, backward executed 10 times
    assert len(losses) == 10
    assert len(grad_norms) == 10
    assert all(np.isfinite(l) and l > 0.0 for l in losses)
    assert all(np.isfinite(g) and g > 0.0 for g in grad_norms)

    # 2. Optimizer step occurred and updated weights
    weights_changed = False
    for p_init, p_curr in zip(initial_weights, small_model.parameters()):
        if not torch.equal(p_init, p_curr):
            weights_changed = True
            break
    assert weights_changed, "Weights did not update after 10 optimizer steps."


def test_stage_2_hundred_updates_stability(synthetic_episodes, small_model):
    """
    Stage 2 (100 updates):
    Inspect loss for NaNs or exploding gradients.
    """
    history = train_model(
        model=small_model,
        episodes=synthetic_episodes,
        device="cpu",
        context_length=20,
        batch_size=32,
        learning_rate=1e-4,
        weight_decay=1e-4,
        gradient_clipping=0.25,
        num_updates=100,
        log_interval=20,
    )

    losses = history["losses"]
    grad_norms = history["grad_norms"]
    spike_rates = history["spike_rates"]

    assert len(losses) == 100
    # No NaNs or Infs
    assert not np.isnan(losses).any(), "Found NaN in loss history"
    assert not np.isinf(losses).any(), "Found Inf in loss history"

    # No exploding gradients: all grad norms are finite and well-behaved
    assert all(np.isfinite(g) for g in grad_norms), "Found non-finite gradient norm in history"
    assert all(g < 100.0 for g in grad_norms), f"Exploding unclipped gradient detected: {max(grad_norms)}"

    # Loss decrease inspection
    early_mean = float(np.mean(losses[:15]))
    late_mean = float(np.mean(losses[-15:]))
    assert late_mean < early_mean, f"Expected loss decrease: early={early_mean:.4f}, late={late_mean:.4f}"

    # Neuromorphic spike rate tracking
    assert all(sr >= 0.0 for sr in spike_rates)


def test_stage_3_checkpoint_saving_and_loading(synthetic_episodes, small_model, tmp_path):
    """
    Stage 3:
    Verify checkpoint saving and loading:
    - Serialization saves model_state_dict, optimizer_state_dict, step, dataset stats
    - Reloading restores model with bit-exact prediction parity
    """
    ckpt_dir = str(tmp_path / "checkpoints")

    history = train_model(
        model=small_model,
        episodes=synthetic_episodes,
        device="cpu",
        context_length=20,
        batch_size=16,
        learning_rate=1e-4,
        weight_decay=1e-4,
        gradient_clipping=0.25,
        num_updates=25,
        checkpoint_dir=ckpt_dir,
        checkpoint_interval=10,
        log_interval=10,
    )

    expected_ckpt = os.path.join(ckpt_dir, "checkpoint_final.pt")
    assert os.path.exists(expected_ckpt), "Final checkpoint was not saved."
    assert os.path.exists(os.path.join(ckpt_dir, "checkpoint_step_10.pt"))
    assert os.path.exists(os.path.join(ckpt_dir, "checkpoint_step_20.pt"))

    # Reload model
    fresh_model = ContinuousSnnDecisionTransformer(
        state_dim=11,
        act_dim=3,
        embed_dim=64,
        n_layer=2,
        n_head=2,
        snn_timesteps=2,
        dropout=0.0,
    )

    ckpt_data = load_checkpoint(expected_ckpt, model=fresh_model, device="cpu")
    assert ckpt_data["step"] == 25
    assert "model_state_dict" in ckpt_data
    assert "optimizer_state_dict" in ckpt_data
    assert "state_mean" in ckpt_data
    assert "state_std" in ckpt_data

    # Bit-exact prediction parity check
    small_model.eval()
    fresh_model.eval()
    with torch.no_grad():
        s = torch.randn(2, 20, 11)
        a = torch.randn(2, 20, 3)
        r = torch.randn(2, 20, 1)
        t = torch.zeros(2, 20, dtype=torch.long)
        m = torch.ones(2, 20)

        out_orig = small_model(s, a, r, t, m)[0]
        out_loaded = fresh_model(s, a, r, t, m)[0]

        assert torch.allclose(out_orig, out_loaded, atol=1e-6)


def test_offline_trainer_epoch_mode(synthetic_episodes, small_model):
    """Verify OfflineTrainer epoch abstraction and step metrics."""
    dataset = TrajectoryDataset(synthetic_episodes, context_length=20)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    optimizer = torch.optim.AdamW(small_model.parameters(), lr=1e-4)

    trainer = OfflineTrainer(
        model=small_model,
        train_loader=loader,
        optimizer=optimizer,
        device="cpu",
        grad_norm_clip=0.25,
    )

    epoch_metrics = trainer.train_epoch(num_steps=10)
    assert "epoch/mean_loss" in epoch_metrics
    assert "epoch/mean_spike_rate" in epoch_metrics
    assert np.isfinite(epoch_metrics["epoch/mean_loss"])


def test_verify_training_stages_runner():
    """Verify staged validation orchestrator for stages 1 and 2."""
    results = verify_training_stages(stages=[1, 2], device="cpu")
    assert results.get("stage_1") == "PASSED"
    assert results.get("stage_2") == "PASSED"
