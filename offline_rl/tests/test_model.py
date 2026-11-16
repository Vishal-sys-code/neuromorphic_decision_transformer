"""
Unit tests for continuous SNN Decision Transformer model architecture (Phase 4 / Gate 4).

Verifies:
1. Surrogate gradient autograd function.
2. Vectorized LIF dynamics over biological time T.
3. PhaseSpikeEncoder modulation and gradient flow.
4. DendriticRouter per-head gating and gradient flow.
5. ThreeFactorPlasticity eligibility trace and weight update.
6. SnnDtBackbone forward and backward pass.
7. Gate 4 sanity test (Section 4.3):
   states [B, K, state_dim], actions [B, K, action_dim], rtg [B, K, 1],
   timesteps [B, K], mask [B, K] -> pred [B, K, action_dim].
8. Gradient verification: EVERY trainable parameter participating in surrogate-gradient
   learning receives non-zero gradients.
9. Single-step autoregressive get_action() rollout inference.
"""

import torch
import torch.nn as nn
import pytest

from offline_rl.models.snn_dt_mujoco import (
    ContinuousSnnDecisionTransformer,
    DendriticRouter,
    PhaseSpikeEncoder,
    SNNDTMuJoCo,
    SnnDtBackbone,
    SpikingTransformerBlock,
    SurrogateSpike,
    ThreeFactorPlasticity,
    VectorizedLIF,
    surrogate_spike,
)


def test_surrogate_spike_gradient():
    """Verify surrogate gradient computes non-zero derivative for sub/supra-threshold values."""
    v = torch.tensor([0.2, 0.5, 0.8], requires_grad=True)
    v_th = 0.5
    alpha = 20.0

    spikes = surrogate_spike(v, v_th, alpha)
    # Forward pass: >= 0.5 triggers spike 1.0, < 0.5 gives 0.0
    assert torch.equal(spikes, torch.tensor([0.0, 1.0, 1.0]))

    loss = spikes.sum()
    loss.backward()
    assert v.grad is not None
    assert torch.all(v.grad > 0.0)


def test_vectorized_lif():
    """Verify VectorizedLIF neuron temporal integration and shape invariance."""
    T, B, L, D = 4, 2, 5, 16
    lif = VectorizedLIF(tau=20.0, v_th=0.5, alpha=25.0)

    currents = torch.randn(T, B, L, D, requires_grad=True)
    spikes_time, final_v = lif(currents)

    assert spikes_time.shape == (T, B, L, D)
    assert final_v.shape == (B, L, D)
    # Binary spikes in {0.0, 1.0}
    assert torch.all((spikes_time == 0.0) | (spikes_time == 1.0))

    loss = spikes_time.sum()
    loss.backward()
    assert currents.grad is not None
    assert currents.grad.abs().max().item() > 0.0


def test_phase_spike_encoder():
    """Verify PhaseSpikeEncoder modulates currents across biological time T and receives gradients."""
    T = 4
    B, L, D = 2, 6, 32
    encoder = PhaseSpikeEncoder(d_model=D, T=T)

    tokens = torch.randn(B, L, D, requires_grad=True)
    phase_mod = encoder(tokens)

    assert phase_mod.shape == (T, B, L, 1)
    # Modulation factor should be positive
    assert torch.all(phase_mod > 0.0)

    loss = phase_mod.square().mean()
    loss.backward()

    assert encoder.freqs.grad is not None
    assert encoder.phases.grad is not None
    assert encoder.proj_in.weight.grad is not None
    assert encoder.proj_out.weight.grad is not None


def test_dendritic_router():
    """Verify DendriticRouter per-head routing softmax gating and gradient flow."""
    B, L, H, head_dim = 2, 6, 4, 16
    router = DendriticRouter(head_dim=head_dim, n_heads=H)

    y_heads = torch.randn(B, L, H, head_dim, requires_grad=True)
    y_routed, alpha = router(y_heads)

    assert y_routed.shape == (B, L, head_dim)
    assert alpha.shape == (B, L, H)
    # Routing coefficients alpha_h must sum to 1 across heads
    assert torch.allclose(alpha.sum(dim=-1), torch.ones(B, L), atol=1e-5)

    loss = y_routed.square().mean()
    loss.backward()

    assert router.mlp[0].weight.grad is not None
    assert router.mlp[2].weight.grad is not None
    assert y_heads.grad is not None


def test_three_factor_plasticity():
    """Verify ThreeFactorPlasticity eligibility trace accumulation and local weight update."""
    in_dim, out_dim = 16, 16
    plasticity = ThreeFactorPlasticity(weight_shape=(out_dim, in_dim), eta=1e-2, lambda_decay=0.9)

    pre = torch.randn(8, in_dim)
    post = torch.randn(8, out_dim)

    # Initial trace is zeros
    assert torch.equal(plasticity.E, torch.zeros(out_dim, in_dim))

    plasticity.update_trace(pre, post)
    # Trace is now non-zero
    assert plasticity.E.abs().sum().item() > 0.0

    # Test weight application
    weight = nn.Parameter(torch.zeros(out_dim, in_dim))
    plasticity.apply(weight, reward=1.0)
    assert weight.abs().sum().item() > 0.0


def test_gate4_snndt_mujoco_sanity_test():
    """
    Exact Section 4.3 Gate 4 Sanity Test:
    Verifies model forward/backward contract on MuJoCo continuous control dimensions.
    """
    B = 8
    K = 20
    state_dim = 11
    action_dim = 3

    model = SNNDTMuJoCo(
        hidden_dim=64,
        state_dim=state_dim,
        action_dim=action_dim,
        n_layer=2,
        n_head=4,
        snn_timesteps=4,
    )

    states = torch.randn(B, K, state_dim)
    actions = torch.randn(B, K, action_dim)
    rtg = torch.randn(B, K, 1)
    timesteps = torch.arange(K).repeat(B, 1)
    mask = torch.ones(B, K)

    pred = model(
        states,
        actions,
        rtg,
        timesteps,
        mask,
    )

    # Required interface assertions
    assert pred.shape == (B, K, action_dim), f"Expected shape {(B, K, action_dim)}, got {pred.shape}"
    # Action bounding in [-1, 1] for continuous control
    assert torch.all(pred >= -1.0) and torch.all(pred <= 1.0)

    loss = pred.square().mean()
    loss.backward()
    print("Forward/backward passed")


def test_all_trainable_parameters_receive_gradients():
    """
    Section 4.3 requirement:
    Verify that every trainable parameter that is supposed to participate in
    surrogate-gradient learning receives gradients.
    """
    B = 8
    K = 20
    state_dim = 11
    action_dim = 3
    hidden_dim = 64

    # Test with explicitly injected research backbone
    backbone = SnnDtBackbone(
        hidden_dim=hidden_dim,
        n_heads=4,
        n_layers=2,
        snn_timesteps=4,
        tau=20.0,
        v_th=0.5,
        alpha=25.0,
    )

    model = SNNDTMuJoCo(
        backbone=backbone,
        hidden_dim=hidden_dim,
        state_dim=state_dim,
        action_dim=action_dim,
    )

    states = torch.randn(B, K, state_dim)
    actions = torch.randn(B, K, action_dim)
    rtg = torch.randn(B, K, 1)
    timesteps = torch.arange(K).repeat(B, 1)
    mask = torch.ones(B, K)

    pred = model(states, actions, rtg, timesteps, mask)
    loss = pred.square().mean()
    loss.backward()

    # Check every trainable parameter
    for name, param in model.named_parameters():
        assert param.requires_grad, f"Parameter {name} should require grad"
        assert param.grad is not None, f"Parameter {name} did not receive grad (grad is None)"

        if "embed_timestep" in name:
            # Active timesteps 0..K-1 must receive non-zero gradients
            active_grad = param.grad[:K].abs().max().item()
            assert active_grad > 0.0, f"Active rows of {name} did not receive gradients"
        else:
            grad_max = param.grad.abs().max().item()
            assert grad_max > 0.0, f"Parameter {name} has zero gradient: grad_max={grad_max}"


def test_continuous_snn_decision_transformer_backward_compat():
    """Verify ContinuousSnnDecisionTransformer returns (action_preds, diagnostics)."""
    batch_size = 4
    context_len = 10
    state_dim = 11
    act_dim = 3

    model = ContinuousSnnDecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        embed_dim=64,
        n_layer=2,
        n_head=2,
        context_len=context_len,
        snn_timesteps=4,
        action_tanh=True,
    )

    states = torch.randn(batch_size, context_len, state_dim)
    actions = torch.randn(batch_size, context_len, act_dim)
    returns_to_go = torch.randn(batch_size, context_len, 1)
    timesteps = torch.randint(0, 50, (batch_size, context_len))
    mask = torch.ones(batch_size, context_len)

    action_preds, diagnostics = model(
        states=states,
        actions=actions,
        returns_to_go=returns_to_go,
        timesteps=timesteps,
        attention_mask=mask,
    )

    assert action_preds.shape == (batch_size, context_len, act_dim)
    assert torch.all(action_preds >= -1.0) and torch.all(action_preds <= 1.0)
    assert "mean_spike_rate" in diagnostics
    assert diagnostics["mean_spike_rate"] >= 0.0

    loss = action_preds.sum()
    loss.backward()
    assert model.state_embedding.weight.grad is not None
    assert model.action_head.weight.grad is not None


def test_autoregressive_get_action():
    """Verify single-step online inference method for gym environment rollouts."""
    state_dim = 11
    act_dim = 3
    context_len = 10

    model = SNNDTMuJoCo(
        state_dim=state_dim,
        action_dim=act_dim,
        hidden_dim=32,
        n_layer=1,
        n_head=2,
        context_len=context_len,
    )
    model.eval()

    states = torch.randn(5, state_dim)
    actions = torch.randn(5, act_dim)
    returns_to_go = torch.randn(5, 1)
    timesteps = torch.arange(5)

    action = model.get_action(states, actions, returns_to_go, timesteps)
    assert action.shape == (act_dim,)
    assert torch.all(action >= -1.0) and torch.all(action <= 1.0)


if __name__ == "__main__":
    test_surrogate_spike_gradient()
    test_vectorized_lif()
    test_phase_spike_encoder()
    test_dendritic_router()
    test_three_factor_plasticity()
    test_gate4_snndt_mujoco_sanity_test()
    test_all_trainable_parameters_receive_gradients()
    test_continuous_snn_decision_transformer_backward_compat()
    test_autoregressive_get_action()
    print("\nAll Gate 4 model tests passed successfully!")
