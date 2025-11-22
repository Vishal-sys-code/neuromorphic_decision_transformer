import torch
import pytest
from src.models.dendritic_router import DendriticRouter

def test_dendritic_router_output_shape():
    d_model = 128
    n_heads = 4
    head_dim = d_model // n_heads
    batch = 2
    seq = 10
    
    router = DendriticRouter(d_model, n_heads)
    y_heads = torch.randn(batch, seq, n_heads, head_dim)
    
    y_routed, alpha = router(y_heads)
    
    assert y_routed.shape == (batch, seq, head_dim)
    assert alpha.shape == (batch, seq, n_heads)

def test_dendritic_router_gating():
    d_model = 32
    n_heads = 2
    head_dim = 16
    router = DendriticRouter(d_model, n_heads)
    
    y_heads = torch.ones(1, 1, n_heads, head_dim)
    # If all inputs are 1, output should be sum(alpha) * 1 = 1 * 1 = 1 (since alpha sums to 1)
    
    y_routed, alpha = router(y_heads)
    
    # alpha sums to 1 over heads
    assert torch.allclose(alpha.sum(dim=-1), torch.tensor(1.0))
    
    # Output should be close to 1
    assert torch.allclose(y_routed, torch.tensor(1.0))

def test_dendritic_router_selectivity():
    # Test that different inputs produce different gates
    d_model = 32
    n_heads = 2
    head_dim = 16
    router = DendriticRouter(d_model, n_heads)
    
    y1 = torch.randn(1, 1, n_heads, head_dim)
    y2 = torch.randn(1, 1, n_heads, head_dim)
    
    _, alpha1 = router(y1)
    _, alpha2 = router(y2)
    
    assert not torch.allclose(alpha1, alpha2)