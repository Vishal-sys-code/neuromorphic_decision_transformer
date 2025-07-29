import torch
import torch.nn as nn

class RateCoder(nn.Module):
    """
    A simple rate coder that converts continuous values into spike trains.
    """
    def __init__(self, embed_dim, window_length):
        super().__init__()
        self.T = window_length
        self.linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, x_embed):
        """
        Forward pass of the rate coder.

        Args:
            x_embed (torch.Tensor): A tensor of shape [B, L, d].

        Returns:
            torch.Tensor: A tensor of shape [B, L, d, T].
        """
        projected_val = self.linear(x_embed)
        rates = torch.sigmoid(projected_val)
        spike_trains = torch.bernoulli(rates.unsqueeze(-1).expand(-1, -1, -1, self.T))
        return spike_trains

class SpikingAttention(nn.Module):
    """
    A simplified spiking attention module.
    """
    def __init__(self, embed_dim, num_heads, window_length):
        super().__init__()
        self.T = window_length
        self.d_k = embed_dim // num_heads
        # Per-head projections: project d_k -> d_k
        self.w_q = nn.Linear(self.d_k, self.d_k, bias=False)
        self.w_k = nn.Linear(self.d_k, self.d_k, bias=False)
        self.w_v = nn.Linear(self.d_k, self.d_k, bias=False)

    def forward(self, head_spikes_input):
        """
        Forward pass of the spiking attention module.

        Args:
            head_spikes_input (torch.Tensor): A tensor of shape [B, L, H, d_k, T].

        Returns:
            torch.Tensor: A tensor of shape [B, L, H, d_k, T].
        """
        # Sum spikes over time window
        summed_spikes = head_spikes_input.sum(dim=-1)  # [B, L, H, d_k]
        # Apply per-head projections
        q = self.w_q(summed_spikes)
        k = self.w_k(summed_spikes)
        v = self.w_v(summed_spikes)

        # [B, L, H, d_k] -> [B, H, L, d_k]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)  # [B, H, L, L]
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, v)  # [B, H, L, d_k]

        # [B, H, L, d_k] -> [B, L, H, d_k]
        output = output.permute(0, 2, 1, 3).contiguous()

        # Expand the output to have a time dimension again
        return output.unsqueeze(-1).expand(-1, -1, -1, -1, self.T)