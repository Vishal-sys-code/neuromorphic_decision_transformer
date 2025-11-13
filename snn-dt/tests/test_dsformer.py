import torch
import pytest
from src.models.dsformer import SpikingAttention

def test_spiking_attention_produces_spikes():
    """
    Tests that the SpikingAttention block generates non-zero spikes with scaled inputs.
    """
    d_model = 64
    n_heads = 4
    batch_size = 2
    seq_len = 10
    lif_tau = 10.0
    surrogate_k = 25.0

    # Initialize the block
    block = SpikingAttention(d_model, n_heads, lif_tau=lif_tau, surrogate_k=surrogate_k)

    # Create a random input tensor and scale it up to increase the likelihood of spiking
    x = torch.randn(batch_size, seq_len, d_model) * 100.0

    # Forward pass
    attn_output = block(x, None, None)

    # Assertions
    assert attn_output.shape == (batch_size, seq_len, d_model), "Output shape is incorrect"
    assert block.spike_count > 0, f"Spike count should be > 0, but was {block.spike_count}"

def test_fake_lif_produces_spikes():
    """
    Tests that the SpikingAttention block with FakeLIF generates non-zero spikes.
    """
    d_model = 64
    n_heads = 4
    batch_size = 2
    seq_len = 10
    lif_tau = 10.0
    surrogate_k = 25.0

    # Initialize the block with FakeLIF enabled
    block = SpikingAttention(d_model, n_heads, lif_tau=lif_tau, surrogate_k=surrogate_k, use_fake_lif=True)

    # Create a random input tensor and scale it up to increase the likelihood of spiking
    x = torch.randn(batch_size, seq_len, d_model) * 5.0

    # Forward pass
    attn_output = block(x, None, None)

    # Assertions
    assert attn_output.shape == (batch_size, seq_len, d_model), "Output shape is incorrect"
    assert block.spike_count > 0, f"Spike count should be > 0, but was {block.spike_count}"