import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pytest
import torch

from offline_rl.models.snn_dt_h import SNNDTH, create_snndt_h_ablation

def test_all_snndt_h_ablations_forward_backward():
    ablations = ["e0", "e1", "e2", "e3", "e4", "e5"]
    B, K, S_dim, A_dim = 2, 10, 11, 3
    
    states = torch.randn(B, K, S_dim, requires_grad=False)
    actions = torch.randn(B, K, A_dim, requires_grad=False)
    rtg = torch.randn(B, K, 1, requires_grad=False)
    timesteps = torch.randint(0, 500, (B, K))
    mask = torch.ones(B, K)

    for ab in ablations:
        model = create_snndt_h_ablation(ab, state_dim=S_dim, action_dim=A_dim, hidden_dim=64, n_layer=2, n_head=4, snn_timesteps=4)
        model.train()
        pred, diag = model(states, actions, rtg, timesteps, return_diagnostics=True)
        assert pred.shape == (B, K, A_dim), f"Shape mismatch for {ab}: {pred.shape}"
        assert torch.isfinite(pred).all(), f"NaNs in predictions for {ab}"
        
        # Test backward pass
        loss = (pred - actions).pow(2).mean()
        loss.backward()
        
        # Check gradients exist
        has_grad = any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        assert has_grad, f"No gradients computed for {ab}"
        print(f"Ablation {ab} PASSED: loss={loss.item():.4f}, spike_rate={diag['mean_spike_rate']:.4f}")

if __name__ == "__main__":
    test_all_snndt_h_ablations_forward_backward()
