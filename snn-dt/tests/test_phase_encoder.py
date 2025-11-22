import torch
import pytest
from src.models.phase_spike_encoder import PhaseSpikeEncoder

def test_phase_spike_encoder_output_shape():
    d_model = 128
    n_heads = 4
    T = 20
    batch_size = 2
    seq_len = 10
    
    encoder = PhaseSpikeEncoder(d_model, n_heads, T)
    timesteps = torch.randint(0, T, (batch_size, seq_len))
    
    spikes = encoder(timesteps)
    
    assert spikes.shape == (batch_size, seq_len, d_model)
    
def test_phase_spike_encoder_binary():
    d_model = 32
    n_heads = 4
    T = 10
    encoder = PhaseSpikeEncoder(d_model, n_heads, T)
    encoder.eval() # Set to eval mode for binary spikes
    timesteps = torch.arange(10).unsqueeze(0) # (1, 10)
    
    spikes = encoder(timesteps)
    
    # Check if all values are 0 or 1
    assert torch.all((spikes == 0) | (spikes == 1))

def test_phase_spike_encoder_gradients():
    d_model = 32
    n_heads = 4
    T = 10
    encoder = PhaseSpikeEncoder(d_model, n_heads, T)
    encoder.train() # Set to train mode for gradients
    timesteps = torch.arange(10).unsqueeze(0)
    
    spikes = encoder(timesteps)
    loss = spikes.sum()
    loss.backward()
    
    # Gradients should flow to omega and phi
    assert encoder.omega.grad is not None
    assert encoder.phi.grad is not None