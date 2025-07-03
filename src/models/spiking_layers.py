"""
Author: Vishal Pandey
Email: pandeyvishal.mlprof@gmail.com
"""

import math
import torch.nn as nn
import torch
from .snn_lif import LIFNeuronLayer # This import is from the original file, kept for other components
from dataclasses import dataclass

@dataclass
class LIFParameters:
    threshold: float = 1.0
    decay_constant: float = 0.9
    reset_potential: float = 0.0

class LIFCell(nn.Module):
    """
    A Leaky Integrate-and-Fire (LIF) neuron cell that processes input current
    and produces spikes. It does not have its own weights (linear layer).
    It maintains membrane potential state.
    """
    def __init__(self, input_size: int, hidden_size: int, p: LIFParameters):
        super().__init__()
        # input_size and hidden_size are often the same (num_neurons) as this cell doesn't project.
        # We'll primarily use hidden_size as the number of neurons in this cell.
        self.num_neurons = hidden_size
        self.p = p

    def forward(self, current_input: torch.Tensor, v_prev: torch.Tensor = None):
        """
        Args:
            current_input: Input current tensor of shape (batch_size, num_neurons).
            v_prev: Optional previous membrane potential, shape (batch_size, num_neurons).
                    If None, it's initialized to zeros.

        Returns:
            spikes: Binary spike tensor of shape (batch_size, num_neurons).
            v_new_reset: Updated membrane potential after spiking and reset.
        """
        batch_size = current_input.shape[0]
        if v_prev is None:
            v_prev = torch.zeros(batch_size, self.num_neurons, device=current_input.device)

        # Membrane potential update: v(t) = v(t-1) * decay + I(t)
        v_new = v_prev * self.p.decay_constant + current_input

        # Spike generation: s(t) = 1 if v(t) > threshold, else 0
        spikes = (v_new > self.p.threshold).float()

        # Reset mechanism: if spiked, potential goes to reset_potential
        v_new_reset = v_new.clone() # Work on a clone for reset
        v_new_reset[spikes == 1] = self.p.reset_potential # Reset only spiking neurons

        return spikes, v_new_reset

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
        spikes_input_encoded = rate_encode(x, self.time_window)  # [T, B, S, E]

        # 2) Accumulate Q, K, V over time
        Q_acc = torch.zeros(B, self.num_heads, S, self.head_dim, device=x.device)
        K_acc = Q_acc.clone()
        V_acc = Q_acc.clone()
        q_state = k_state = v_state = None

        for t in range(self.time_window):
            inp = spikes_input_encoded[t].reshape(B * S, E)
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