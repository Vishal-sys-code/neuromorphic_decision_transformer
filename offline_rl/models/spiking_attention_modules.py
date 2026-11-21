"""
Advanced Spiking Attention and Normalization Modules for SNN-DT-H.
Implements CVPR 2025 Decision SpikeFormer Innovations:
1. LocalSpikingAttention (PSSA) - Sliding causal window (S=8) with learnable pairwise positional bias
2. TemporalSpikingAttention (TSSA) - Cross-simulation-step concatenation across T x D
3. DendriticRoutingFusion - Learned adaptive gating between local and temporal streams
4. ProgressiveThresholdNorm (PTBN) - Threshold-dependent LayerNorm transitioning to BatchNorm
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalSpikingAttention(nn.Module):
    r"""
    Positional Spiking Self-Attention (PSSA) with Causal Sliding Window.
    Formula:
        A_i = Q_i * sum_{j=1}^N M_{ij} (P_{i-j} * K_j * V_j)
    where:
        M_{ij} = 1 for 0 <= i - j < S, and 0 otherwise.
        P_{i-j} is a learnable relative positional bias.

    Implemented using fast 1D depthwise causal convolutions (O(N * S * D))
    avoiding costly O(N^2) attention matrices and softmax.
    """

    def __init__(
        self,
        dim: int,
        window_size: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size

        # Learnable relative positional bias P \in R^{S x D}
        # Initialized to ones (identity modulation)
        self.pos_bias = nn.Parameter(torch.ones(window_size, dim))

        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for PSSA.
        Supports either:
            [B, N, D] (rate-coded or single simulation step)
            [T, B, N, D] (multi-step biological simulation)

        Returns:
            out: [B, N, D] (or [T, B, N, D] if multi-step is preserved)
        """
        if q.dim() == 4:
            # Multi-step: (T, B, N, D)
            T, B, N, D = q.shape
            # Flatten T into batch dimension for vectorized depthwise convolution
            q_flat = q.reshape(T * B, N, D)
            k_flat = k.reshape(T * B, N, D)
            v_flat = v.reshape(T * B, N, D)

            out_flat = self._forward_3d(q_flat, k_flat, v_flat, N, D)
            out = out_flat.view(T, B, N, D)
            # Average across simulation steps T
            return self.dropout(self.out_proj(out.mean(dim=0)))

        B, N, D = q.shape
        out = self._forward_3d(q, k, v, N, D)
        return self.dropout(self.out_proj(out))

    def _forward_3d(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        N: int,
        D: int,
    ) -> torch.Tensor:
        # Co-spiking coincidence product: Z = K * V, shape: (batch, N, D) -> (batch, D, N)
        z = (k * v).permute(0, 2, 1)

        # Causal left-padding of window_size - 1
        z_pad = F.pad(z, (self.window_size - 1, 0))

        # pos_bias: (window_size, D) -> depthwise conv weight: (D, 1, window_size)
        # Flip along window dimension so offset delta = i - j aligns causally with conv
        weight = self.pos_bias.flip(0).permute(1, 0).unsqueeze(1)

        conv_out = F.conv1d(z_pad, weight, groups=D)  # (batch, D, N)
        conv_out = conv_out.permute(0, 2, 1)  # (batch, N, D)

        # Query modulation: A_i = Q_i * sum_j M_{ij} P_{i-j} K_j V_j
        return q * conv_out


class TemporalSpikingAttention(nn.Module):
    r"""
    Temporal Spiking Self-Attention (TSSA).
    Concatenates spike representations across all simulation steps T before attention:
        Q_c, K_c, V_c \in R^{B x N x (T * D)}
    allowing direct cross-simulation-step dependency modeling without premature reduction.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 4,
        snn_timesteps: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.T = snn_timesteps
        self.concat_dim = dim * snn_timesteps

        assert self.concat_dim % n_heads == 0
        self.head_dim = self.concat_dim // n_heads

        self.q_proj = nn.Linear(self.concat_dim, self.concat_dim)
        self.k_proj = nn.Linear(self.concat_dim, self.concat_dim)
        self.v_proj = nn.Linear(self.concat_dim, self.concat_dim)

        self.out_proj = nn.Linear(self.concat_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        spikes_q_t: torch.Tensor,
        spikes_k_t: torch.Tensor,
        spikes_v_t: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Inputs:
            spikes_q_t, spikes_k_t, spikes_v_t: [T, B, N, D]
        Returns:
            out: [B, N, D]
        """
        T, B, N, D = spikes_q_t.shape

        # Concatenate across simulation steps T: (B, N, T * D)
        q_c = spikes_q_t.permute(1, 2, 0, 3).reshape(B, N, T * D)
        k_c = spikes_k_t.permute(1, 2, 0, 3).reshape(B, N, T * D)
        v_c = spikes_v_t.permute(1, 2, 0, 3).reshape(B, N, T * D)

        q = self.q_proj(q_c).view(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(k_c).view(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(v_c).view(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if causal_mask is not None:
            if causal_mask.dtype == torch.bool:
                scores = scores.masked_fill(causal_mask, float("-inf"))
            else:
                scores = scores + causal_mask

        weights = torch.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)

        attn_out = torch.matmul(weights, v)  # (B, H, N, head_dim)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, N, self.concat_dim)

        out = self.out_proj(attn_out)  # (B, N, D)
        return self.dropout(out)


class DendriticRoutingFusion(nn.Module):
    """
    Adaptive Dendritic Routing Fusion between Local (PSSA) and Temporal (TSSA) feature streams.
    Computes learned per-token gating:
        g = sigmoid(W_g [A_local; A_temporal] + b_g)
        A = g * A_local + (1 - g) * A_temporal
    """

    def __init__(self, dim: int, hidden: int = 64):
        super().__init__()
        self.dim = dim
        self.gate_mlp = nn.Sequential(
            nn.Linear(dim * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        a_local: torch.Tensor,
        a_temporal: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inputs:
            a_local: [B, N, D]
            a_temporal: [B, N, D]
        Returns:
            fused: [B, N, D]
            gate: [B, N, D] gating weights
        """
        concat = torch.cat([a_local, a_temporal], dim=-1)
        gate = self.gate_mlp(concat)  # (B, N, D)
        fused = gate * a_local + (1.0 - gate) * a_temporal
        return fused, gate


class ProgressiveThresholdNorm(nn.Module):
    """
    Progressive Threshold-dependent Batch Normalization (PTBN).
    Transitions smoothly during training from threshold-dependent LayerNorm (tdLN)
    to threshold-dependent BatchNorm (tdBN):
        PTBN(X) = theta * tdLN(X) + (1 - theta) * tdBN(X)
    where:
        theta = max(0, 1 - current_step / transition_steps)
    """

    def __init__(
        self,
        dim: int,
        v_th: float = 0.5,
        transition_steps: int = 10000,
        eps: float = 1e-5,
        momentum: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.v_th = float(v_th)
        self.transition_steps = max(1, transition_steps)
        self.eps = eps
        self.momentum = momentum

        # Parameters for tdLN
        self.gamma_ln = nn.Parameter(torch.ones(dim))
        self.beta_ln = nn.Parameter(torch.zeros(dim))

        # Parameters for tdBN
        self.gamma_bn = nn.Parameter(torch.ones(dim))
        self.beta_bn = nn.Parameter(torch.zeros(dim))
        self.register_buffer("running_mean", torch.zeros(dim))
        self.register_buffer("running_var", torch.ones(dim))

        # Step tracking
        self.register_buffer("current_step", torch.tensor(0, dtype=torch.long))

    def step_update(self, step: Optional[int] = None):
        """Update optimizer step count."""
        if step is not None:
            self.current_step.fill_(step)
        else:
            self.current_step.add_(1)

    def forward(
        self,
        x: torch.Tensor,
        step: Optional[int] = None,
        force_mode: Optional[str] = None,  # 'tdln', 'tdbn', or None (progressive)
    ) -> torch.Tensor:
        """
        x: [B, N, D]
        """
        if step is not None:
            self.current_step.fill_(step)

        B, N, D = x.shape

        # 1. Compute tdLN across feature dimension D
        mean_ln = x.mean(dim=-1, keepdim=True)
        var_ln = x.var(dim=-1, unbiased=False, keepdim=True)
        x_tdln = (x - mean_ln) / torch.sqrt(var_ln + self.eps) * self.v_th
        x_tdln = x_tdln * self.gamma_ln + self.beta_ln

        if force_mode == "tdln":
            return x_tdln

        # 2. Compute tdBN across batch and sequence
        if self.training:
            # Flatten to (B * N, D)
            x_flat = x.reshape(-1, D)
            mean_bn = x_flat.mean(dim=0)
            var_bn = x_flat.var(dim=0, unbiased=False)

            with torch.no_grad():
                self.running_mean.lerp_(mean_bn, self.momentum)
                self.running_var.lerp_(var_bn, self.momentum)
        else:
            mean_bn = self.running_mean
            var_bn = self.running_var

        x_tdbn = (x - mean_bn) / torch.sqrt(var_bn + self.eps) * self.v_th
        x_tdbn = x_tdbn * self.gamma_bn + self.beta_bn

        if force_mode == "tdbn":
            return x_tdbn

        # 3. Progressive blending: theta * tdLN + (1 - theta) * tdBN
        curr_step = float(self.current_step.item())
        theta = max(0.0, 1.0 - curr_step / float(self.transition_steps))

        return theta * x_tdln + (1.0 - theta) * x_tdbn
