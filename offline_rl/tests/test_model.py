"""
Unit tests for continuous SNN Decision Transformer model architecture.
"""

import torch

from offline_rl.models.snn_dt_mujoco import (
    ContinuousSnnDecisionTransformer,
    SurrogateSpike,
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
    B, L, D = 2, 5, 16
    timesteps = 4
    lif = VectorizedLIF(dim=D, timesteps=timesteps, tau=20.0, v_th=0.5)

    current = torch.randn(B, L, D, requires_grad=True)
    spike_rate, spikes = lif(current)

    assert spike_rate.shape == (B, L, D)
    assert spikes.shape == (timesteps, B, L, D)
    # Spike rate is average of binary spikes, so in [0, 1]
    assert torch.all((spike_rate >= 0.0) & (spike_rate <= 1.0))

    loss = spike_rate.sum()
    loss.backward()
    assert current.grad is not None


def test_continuous_snn_decision_transformer_forward_backward():
    """Verify SNN-DT forward pass, action output bounds, diagnostics, and backward pass."""
    batch_size = 4
    context_len = 10
    state_dim = 11  # Hopper obs
    act_dim = 3     # Hopper action

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

    # Validate output shape
    assert action_preds.shape == (batch_size, context_len, act_dim)
    # With tanh, actions must strictly be in [-1, 1]
    assert torch.all(action_preds >= -1.0) and torch.all(action_preds <= 1.0)

    # Validate diagnostics
    assert "mean_spike_rate" in diagnostics
    assert diagnostics["mean_spike_rate"] >= 0.0

    # Validate backward pass
    loss = action_preds.sum()
    loss.backward()

    # Check gradients on embeddings, transformer layers, and head
    assert model.embed_state.weight.grad is not None
    assert model.head.weight.grad is not None


def test_autoregressive_get_action():
    """Verify single-step online inference method for gym environment rollouts."""
    state_dim = 11
    act_dim = 3
    context_len = 10

    model = ContinuousSnnDecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        embed_dim=32,
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
    test_continuous_snn_decision_transformer_forward_backward()
    test_autoregressive_get_action()
    print("All model tests passed successfully!")
