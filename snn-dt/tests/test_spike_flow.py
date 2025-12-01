import torch
import pytest
from src.models.snn_dt import SnnDt

class MockConfig:
    class ModelConfig:
        d_model = 64
        n_heads = 4
        n_layers = 1
        seq_len = 5
    class DatasetConfig:
        state_dim = 10
        act_dim = 2
        max_timesteps = 20
        is_discrete = False
    class SnnConfig:
        lif_tau = 0.02  # Faster integration
        surrogate_k = 1.0
        v_th = 0.1  # Lower firing threshold
        current_scale = 1.0
        use_plasticity = True
        eta_local = 0.01
        lambda_decay = 0.95

    model = ModelConfig()
    dataset = DatasetConfig()
    snn = SnnConfig()

def test_snn_dt_spike_flow():
    cfg = MockConfig()
    model = SnnDt(cfg)
    
    batch_size = 2
    seq_len = 5
    
    batch = {
        "states": torch.ones(batch_size, seq_len, cfg.dataset.state_dim),
        "actions": torch.ones(batch_size, seq_len, cfg.dataset.act_dim),
        "returns_to_go": torch.ones(batch_size, seq_len, 1),
        "timesteps": torch.randint(0, cfg.dataset.max_timesteps, (batch_size, seq_len))
    }
    
    # Forward pass
    out = model(batch)
    
    # Check output shape
    assert out.shape == (batch_size, seq_len, cfg.dataset.act_dim)
    
    # Check spikes are being counted
    assert model.count_spikes() > 0
    
    # Check plasticity updated trace (indirectly via no error)
    model.apply_plasticity(reward=1.0)

def test_snn_dt_zero_spikes_if_high_threshold():
    cfg = MockConfig()
    cfg.snn.v_th = 1000.0 # High threshold
    model = SnnDt(cfg)
    
    batch = {
        "states": torch.randn(2, 5, cfg.dataset.state_dim),
        "actions": torch.randn(2, 5, cfg.dataset.act_dim),
        "returns_to_go": torch.randn(2, 5, 1),
        "timesteps": torch.randint(0, 20, (2, 5))
    }
    
    model(batch)
    
    # Should be 0 spikes (Phase spikes might happen, but blocks won't fire)
    # PhaseSpikeEncoder is deterministic and not LIF based.
    # Block LIFs should be silent.
    assert model.count_spikes() == 0.0