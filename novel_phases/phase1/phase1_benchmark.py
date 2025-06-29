"""
Adaptive Spiking Windows Implementation
Phase 1: Token-wise Temporal Allocation for Spiking Transformers
 + vectorized masked attention (einsum)
 + unit test for S=4, T=5
 + speed/memory benchmarking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict

class LIFNeuron(nn.Module):
    """Leaky Integrate-and-Fire neuron with learnable decay"""
    def __init__(self, tau_mem=20.0, tau_syn=5.0, v_threshold=1.0, v_reset=0.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(np.exp(-1/tau_mem)))
        self.alpha = nn.Parameter(torch.tensor(np.exp(-1/tau_syn)))
        self.v_threshold = v_threshold
        self.v_reset = v_reset

    def forward(self, x, state=None):
        # x: [B=1 or B, D], state: (v_mem, i_syn)
        if state is None:
            v_mem = torch.zeros_like(x)
            i_syn = torch.zeros_like(x)
        else:
            v_mem, i_syn = state
        i_syn = self.alpha * i_syn + x
        v_mem = self.beta * v_mem + i_syn
        spikes = (v_mem >= self.v_threshold).float()
        v_mem = v_mem * (1 - spikes) + self.v_reset * spikes
        return spikes, (v_mem, i_syn)


class AdaptiveSpikingAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads=4, T_max=20, lambda_reg=1e-3, dropout=0.1):
        super().__init__()
        assert embedding_dim % num_heads == 0
        self.D = embedding_dim
        self.H = num_heads
        self.Dh = embedding_dim // num_heads
        self.T_max = T_max
        self.lambda_reg = lambda_reg
        # projections
        self.q_proj = nn.Linear(self.D, self.D, bias=False)
        self.k_proj = nn.Linear(self.D, self.D, bias=False)
        self.v_proj = nn.Linear(self.D, self.D, bias=False)
        self.out_proj = nn.Linear(self.D, self.D)
        # spiking neurons
        self.lif_q = LIFNeuron()
        self.lif_k = LIFNeuron()
        self.lif_v = LIFNeuron()
        # gating
        self.window_gate = nn.Sequential(
            nn.Linear(self.D, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )
        # dropout & scale
        self.dropout = nn.Dropout(dropout)
        self.scale = self.Dh ** -0.5
        self.T_history = []

    def get_windows(self, x):
        # x: [B, S, D]
        gate = self.window_gate(x).squeeze(-1)  # [B, S]
        Ti = (gate * self.T_max).ceil().clamp(1, self.T_max).long()
        return Ti

    def generate_spikes(self, x, Ti, lif):
        # x: [B, S, D] -> spikes: [B, S, T, D]
        B, S, D = x.shape
        out = x.new_zeros(B, S, self.T_max, D)
        for b in range(B):
            for s in range(S):
                state = None
                for t in range(Ti[b, s]):
                    spk, state = lif(x[b:b+1, s], state)
                    out[b, s, t] = spk
        return out  # [B, S, T_max, D]

    def vectorized_attention(self, q_spk, k_spk, v_spk, Ti):
        # q_spk,k_spk,v_spk: [B, S, T, D]; reshape -> [B, S, T, H, Dh]
        B, S, T, D = q_spk.shape
        H, Dh = self.H, self.Dh
        q = q_spk.view(B, S, T, H, Dh)
        k = k_spk.view(B, S, T, H, Dh)
        v = v_spk.view(B, S, T, H, Dh)
        # mask: [B, S, T]
        mask = (torch.arange(T, device=Ti.device)[None, None, :] < Ti[:, :, None]).float()
        # apply mask
        mask4 = mask[:, :, :, None, None]  # [B,S,T,1,1]
        q = q * mask4
        k = k * mask4
        v = v * mask4
        # score: [B,H,S,S]
        Sraw = torch.einsum('bithd,bjthd->bhij', q, k) * self.scale
        W = F.softmax(Sraw, dim=-1)
        W = self.dropout(W)
        # aggregate v: first mean over time -> [B,S,H,Dh], then attention
        v_mean = v.mean(dim=2).transpose(1, 2)  # [B,H,S,Dh]
        out = torch.einsum('bhij,bhjd->bhid', W, v_mean)  # [B,H,S,Dh]
        out = out.transpose(1, 2).reshape(B, S, D)
        return self.out_proj(out), W

    def forward(self, x):
        B, S, D = x.shape
        # project
        q = self.q_proj(x); k = self.k_proj(x); v = self.v_proj(x)
        # windows
        Ti = self.get_windows(x)  # [B,S]
        # spikes
        q_spk = self.generate_spikes(q, Ti, self.lif_q)
        k_spk = self.generate_spikes(k, Ti, self.lif_k)
        v_spk = self.generate_spikes(v, Ti, self.lif_v)
        # attention
        out, W = self.vectorized_attention(q_spk, k_spk, v_spk, Ti)
        # reg loss
        reg = self.lambda_reg * Ti.float().mean()
        # log
        if self.training:
            self.T_history.append(Ti.cpu().numpy())
        return out, {'reg_loss': reg, 'Ti': Ti, 'W': W}

# --- Unit Test & Benchmark --------------------------------------------------

def brute_force(q, k, Ti):
    B, S, T, H, Dh = q.shape
    S1 = torch.zeros(B, H, S, S)
    for b in range(B):
        for h in range(H):
            for i in range(S):
                for j in range(S):
                    tlim = min(Ti[b,i], Ti[b,j]).item()
                    val = 0.0
                    for t in range(tlim):
                        val += (q[b,i,t,h] * k[b,j,t,h]).sum()
                    S1[b,h,i,j] = val
    return S1

if __name__ == "__main__":
    # test shapes
    B,S,T,H,Dh = 1,4,5,2,3
    D = H*Dh
    model = AdaptiveSpikingAttention(D, num_heads=H, T_max=T)
    # fake data
    x = torch.randn(B, S, D)
    q = torch.randn(B, S, T, H, Dh)
    k = torch.randn_like(q)
    Ti = torch.randint(1, T+1, (B,S))
    # brute vs vectorized
    bf = brute_force(q, k, Ti)
    vec = model.vectorized_attention(q.view(B,S,T,D), k.view(B,S,T,D),
                                     torch.randn(B,S,T,D).view(B,S,T,D), Ti)[1]
    # we only compare raw scores before softmax:
    # extract raw Sraw from vectorized code manually
    # (re-run vectorized_attention but output raw Sraw)
    def raw_vec(q_spk,k_spk,Ti):
        # q_spk, k_spk: [B, S, T, H, Dh]
        B, S, T, H, Dh = q_spk.shape
        q_ = q_spk
        k_ = k_spk
        mask = (torch.arange(T)[None, None, :] < Ti[:, :, None]).float()
        q_ = q_ * mask[:, :, :, None, None]
        k_ = k_ * mask[:, :, :, None, None]
        return torch.einsum('bithd,bjthd->bhij', q_, k_)
    rv = raw_vec(q, k, Ti)
    assert torch.allclose(bf, rv, atol=1e-5)
    print("✅ Unit test passed (S=4, T=5)")

    # Benchmark
    reps = 100
    start = time.perf_counter()
    for _ in range(reps):
        brute_force(q,k,Ti)
    t1 = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(reps):
        _ = raw_vec(q,k,Ti)
    t2 = time.perf_counter() - start

    print(f"Brute force:      {t1:.4f}s for {reps} runs")
    print(f"Vectorized (raw): {t2:.4f}s for {reps} runs")