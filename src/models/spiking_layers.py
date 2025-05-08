"""
Author: Vishal Pandey
Email: pandeyvishal.mlprof@gmail.com
"""

import math
import torch.nn as nn
import torch
from .snn_lif import LIFNeuronLayer

def rate_encode(x: torch.Tensor, time_window: int, x_min: float = 0.0, x_max: float = 1.0):
    """
    Differentiable rate coding approximation:
    For each element in x (normalized to [0,1]), approximate spike firing probability over time window using sigmoid.
    Returns: Tensor of shape [T, *x.shape], dtype float.
    """
    # normalize to [0,1]
    p = (x - x_min) / (x_max - x_min)
    p = p.clamp(0.0, 1.0)
    shape = [time_window] + [1] * x.dim()
    t_idx = torch.arange(time_window, device=x.device).view(*shape).float()
    # Use sigmoid to approximate spike firing probability
    spikes = torch.sigmoid(10 * (p.unsqueeze(0) * time_window - t_idx))
    return spikes

class SpikingSelfAttention(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int = 1,
                 time_window: int = 10):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.time_window = time_window

        # Spiking projections for Q, K, V
        self.q_proj = LIFNeuronLayer(embed_dim, embed_dim)
        self.k_proj = LIFNeuronLayer(embed_dim, embed_dim)
        self.v_proj = LIFNeuronLayer(embed_dim, embed_dim)
        # Final spiking output projection
        self.out_proj = LIFNeuronLayer(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor):
        """
        x: [batch, seq_len, embed_dim]
        returns: [batch, seq_len, embed_dim]
        """
        B, S, E = x.shape
        # 1) Encode embeddings into spike trains
        spikes = rate_encode(x, self.time_window)  # [T, B, S, E]

        # 2) Accumulate Q, K, V over time
        Q_acc = torch.zeros(B, self.num_heads, S, self.head_dim, device=x.device)
        K_acc = Q_acc.clone()
        V_acc = Q_acc.clone()
        q_state = k_state = v_state = None

        for t in range(self.time_window):
            inp = spikes[t].reshape(B * S, E)
            q_spk, q_state = self.q_proj(inp, q_state)
            k_spk, k_state = self.k_proj(inp, k_state)
            v_spk, v_state = self.v_proj(inp, v_state)

            # reshape to [B, heads, S, head_dim]
            qh = q_spk.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
            kh = k_spk.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
            vh = v_spk.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

            Q_acc += qh
            K_acc += kh
            V_acc += vh

        # 3) Compute attention scores & weights
        # scores: [B, heads, S, S]
        scores = torch.einsum('bhqd,bhkd->bhqk', Q_acc, K_acc) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        # weighted sum: [B, heads, S, head_dim]
        out_h = torch.einsum('bhqk,bhkd->bhqd', attn, V_acc)

        # 4) Merge heads & final spiking projection
        out = out_h.transpose(1, 2).contiguous().view(B * S, E)
        out_spk, _ = self.out_proj(out, None)
        return out_spk.view(B, S, E)