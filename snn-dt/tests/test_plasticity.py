import torch
import pytest
from src.models.three_factor_plasticity import ThreeFactorPlasticity

def test_plasticity_trace_update():
    eta = 0.1
    lam = 0.5
    plasticity = ThreeFactorPlasticity(eta, lam)
    
    # 1 input, 1 output
    trace = torch.zeros(1, 1)
    pre = torch.tensor([[1.0]]) # (B=1, in=1)
    post = torch.tensor([[1.0]]) # (B=1, out=1)
    
    # Update trace
    # E = 0.5 * 0 + (1 * 1) = 1
    trace = plasticity(trace, pre, post)
    assert trace.item() == 1.0
    
    # Next step
    # pre=0, post=0
    pre = torch.zeros(1, 1)
    post = torch.zeros(1, 1)
    # E = 0.5 * 1 + 0 = 0.5
    trace = plasticity(trace, pre, post)
    assert trace.item() == 0.5

def test_plasticity_weight_update():
    eta = 0.1
    lam = 0.5
    plasticity = ThreeFactorPlasticity(eta, lam)
    
    layer = torch.nn.Linear(1, 1, bias=False)
    layer.weight.data.fill_(0.0)
    
    trace = torch.tensor([[1.0]])
    reward = 2.0
    
    # Delta W = eta * R * E = 0.1 * 2 * 1 = 0.2
    plasticity.update_weights(layer, trace, reward)
    
    assert torch.allclose(layer.weight.data, torch.tensor([[0.2]]))
