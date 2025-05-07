import torch
from src.models.spiking_layers import SpikingSelfAttention
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def test_spiking_self_attention_backprop():
    torch.manual_seed(0)
    B, S, E, H, T = 2, 5, 16, 2, 8
    x = torch.randn(B, S, E, requires_grad=True)
    sa = SpikingSelfAttention(embed_dim=E, num_heads=H, time_window=T)
    out = sa(x)
    # shape check
    assert out.shape == (B, S, E)
    # backprop
    loss = out.mean()
    loss.backward()
    # gradients should flow back to input
    assert x.grad is not None and x.grad.abs().sum() > 0