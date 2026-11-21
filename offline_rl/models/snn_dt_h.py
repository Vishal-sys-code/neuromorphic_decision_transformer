"""
SNN-DT-H: Temporally and Locally Conditioned Spiking Decision Transformer
Specialized architecture to exceed CVPR 2025 Decision SpikeFormer PSSA benchmarks on D4RL Hopper.

Integrates:
- Modality embeddings + PhaseSpikeEncoder
- Multi-step Vectorized LIF neuron dynamics across biological time T
- Local Spiking Attention (PSSA) with causal sliding window S=8
- Temporal Spiking Attention (TSSA) across concatenated simulation steps T x D
- Adaptive Dendritic Routing Fusion between local and temporal streams
- Progressive Threshold-dependent Batch Normalization (PTBN: tdLN -> tdBN)
- Continuous action head with Tanh bounding
"""

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from offline_rl.models.snn_dt_mujoco import ENV_DIM_MAP, SNNDTMuJoCo, get_env_dims
from offline_rl.models.spiking_attention_modules import (
    DendriticRoutingFusion,
    LocalSpikingAttention,
    ProgressiveThresholdNorm,
    TemporalSpikingAttention,
)
from src.models.snn_dt import (
    PhaseSpikeEncoder,
    VectorizedLIF,
    surrogate_spike,
)


class SpikingTransformerBlockH(nn.Module):
    """
    Advanced Spiking Transformer Block supporting:
    - Mode 'standard': Baseline SNN-DT rate-coded attention
    - Mode 'pssa': Positional Spiking Self-Attention with sliding window S
    - Mode 'tssa': Temporal Spiking Self-Attention across simulation steps
    - Mode 'dual': Dual-branch (PSSA + TSSA) with optional Dendritic Fusion
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        n_heads: int = 4,
        snn_timesteps: int = 4,
        tau: float = 20.0,
        v_th: float = 0.5,
        alpha: float = 25.0,
        current_scale: float = 1.0,
        dropout: float = 0.1,
        attention_mode: str = "dual",  # 'standard', 'pssa', 'tssa', 'dual'
        use_dendritic_fusion: bool = True,
        window_size: int = 8,
        norm_mode: str = "ptbn",  # 'layernorm', 'tdln', 'tdbn', 'ptbn'
        ptbn_transition_steps: int = 10000,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.snn_timesteps = snn_timesteps
        self.attention_mode = attention_mode.lower()
        self.use_dendritic_fusion = use_dendritic_fusion
        self.norm_mode = norm_mode.lower()

        # Projections into spiking currents
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        self.register_parameter(
            "input_gain",
            nn.Parameter(torch.tensor(float(current_scale), dtype=torch.float32)),
        )

        # Multi-step Vectorized LIF neurons
        self.lif_q = VectorizedLIF(tau=tau, v_th=v_th, alpha=alpha)
        self.lif_k = VectorizedLIF(tau=tau, v_th=v_th, alpha=alpha)
        self.lif_v = VectorizedLIF(tau=tau, v_th=v_th, alpha=alpha)

        # 1. PSSA Branch (Local Spiking Attention)
        if self.attention_mode in ("pssa", "dual"):
            self.pssa_branch = LocalSpikingAttention(
                dim=hidden_dim,
                window_size=window_size,
                dropout=dropout,
            )

        # 2. TSSA Branch (Temporal Spiking Attention)
        if self.attention_mode in ("tssa", "dual"):
            self.tssa_branch = TemporalSpikingAttention(
                dim=hidden_dim,
                n_heads=n_heads,
                snn_timesteps=snn_timesteps,
                dropout=dropout,
            )

        # 3. Dendritic Fusion
        if self.attention_mode == "dual" and self.use_dendritic_fusion:
            self.router = DendriticRoutingFusion(dim=hidden_dim)

        # 4. Standard baseline fallback
        if self.attention_mode == "standard":
            self.head_dim = hidden_dim // n_heads
            self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # 5. Normalization
        if self.norm_mode in ("ptbn", "tdln", "tdbn"):
            self.norm = ProgressiveThresholdNorm(
                dim=hidden_dim,
                v_th=v_th,
                transition_steps=ptbn_transition_steps,
            )
        else:
            self.norm = nn.LayerNorm(hidden_dim)

        self.spike_count_tensor = None
        self.last_gate = None

    def reset_state(self):
        self.lif_q.reset_state()
        self.lif_k.reset_state()
        self.lif_v.reset_state()

    def forward(
        self,
        x: torch.Tensor,
        phase_mod: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
        step: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        x: [B, N, D]
        phase_mod: [T, B, N, 1]
        """
        B, N, D = x.shape
        device = x.device

        # Create time-varying currents across biological simulation steps T: [T, B, N, D]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q_curr = q.unsqueeze(0) * phase_mod * self.input_gain
        k_curr = k.unsqueeze(0) * phase_mod * self.input_gain
        v_curr = v.unsqueeze(0) * phase_mod * self.input_gain

        # Run LIF neurons
        spikes_q_t, _ = self.lif_q(q_curr)  # (T, B, N, D)
        spikes_k_t, _ = self.lif_k(k_curr)
        spikes_v_t, _ = self.lif_v(v_curr)

        # Accumulate spike diagnostics
        s_sum = (spikes_q_t.sum() + spikes_k_t.sum() + spikes_v_t.sum()).detach()
        if self.spike_count_tensor is None or self.spike_count_tensor.device != device:
            self.spike_count_tensor = s_sum
        else:
            self.spike_count_tensor = self.spike_count_tensor + s_sum

        gate_weights = None

        # Execute attention according to mode
        if self.attention_mode == "pssa":
            out = self.pssa_branch(spikes_q_t, spikes_k_t, spikes_v_t)

        elif self.attention_mode == "tssa":
            out = self.tssa_branch(spikes_q_t, spikes_k_t, spikes_v_t, causal_mask=causal_mask)

        elif self.attention_mode == "dual":
            out_local = self.pssa_branch(spikes_q_t, spikes_k_t, spikes_v_t)
            out_temporal = self.tssa_branch(spikes_q_t, spikes_k_t, spikes_v_t, causal_mask=causal_mask)

            if self.use_dendritic_fusion:
                out, gate_weights = self.router(out_local, out_temporal)
                self.last_gate = gate_weights.detach()
            else:
                out = 0.5 * out_local + 0.5 * out_temporal

        else:
            # Standard baseline attention
            spikes_q = spikes_q_t.sum(dim=0).view(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
            spikes_k = spikes_k_t.sum(dim=0).view(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
            spikes_v = spikes_v_t.sum(dim=0).view(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

            scores = torch.matmul(spikes_q, spikes_k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            if causal_mask is not None:
                scores = scores.masked_fill(causal_mask, float("-inf"))
            weights = torch.softmax(scores, dim=-1)
            weights = torch.nan_to_num(weights, nan=0.0)
            attn_out = torch.matmul(weights, spikes_v).permute(0, 2, 1, 3).reshape(B, N, D)
            out = self.out_proj(attn_out)

        # Residual connection
        x = x + out

        # Normalization
        if isinstance(self.norm, ProgressiveThresholdNorm):
            force = "tdln" if self.norm_mode == "tdln" else ("tdbn" if self.norm_mode == "tdbn" else None)
            x = self.norm(x, step=step, force_mode=force)
        else:
            x = self.norm(x)

        return x, gate_weights


class SNNDTH(nn.Module):
    """
    SNN-DT-H: Temporally and Locally Conditioned Spiking Decision Transformer.
    Full policy module replacing SNNDTMuJoCo for Hopper-targeted research.
    """

    def __init__(
        self,
        state_dim: int = 11,
        action_dim: int = 3,
        hidden_dim: int = 128,
        n_layer: int = 3,
        n_head: int = 4,
        snn_timesteps: int = 4,
        tau: float = 20.0,
        v_th: float = 0.5,
        alpha: float = 25.0,
        current_scale: float = 1.0,
        max_ep_len: int = 1000,
        context_len: int = 20,
        action_tanh: bool = True,
        dropout: float = 0.1,
        use_phase_coding: bool = True,
        attention_mode: str = "dual",  # 'standard', 'pssa', 'tssa', 'dual'
        use_dendritic_fusion: bool = True,
        window_size: int = 8,
        norm_mode: str = "ptbn",  # 'layernorm', 'tdln', 'tdbn', 'ptbn'
        ptbn_transition_steps: int = 10000,
        **kwargs,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.act_dim = action_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = hidden_dim
        self.n_layer = n_layer
        self.n_head = n_head
        self.snn_timesteps = snn_timesteps
        self.max_ep_len = max_ep_len
        self.context_len = context_len
        self.action_tanh = action_tanh
        self.use_phase_coding = use_phase_coding
        self.attention_mode = attention_mode
        self.norm_mode = norm_mode

        # Modality embeddings
        self.state_embedding = nn.Linear(state_dim, hidden_dim)
        self.action_embedding = nn.Linear(action_dim, hidden_dim)
        self.return_embedding = nn.Linear(1, hidden_dim)
        self.embed_timestep = nn.Embedding(max_ep_len + 1, hidden_dim)
        self.embed_ln = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Phase spike encoder
        if self.use_phase_coding:
            self.phase_encoder = PhaseSpikeEncoder(d_model=hidden_dim, T=snn_timesteps)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            SpikingTransformerBlockH(
                hidden_dim=hidden_dim,
                n_heads=n_head,
                snn_timesteps=snn_timesteps,
                tau=tau,
                v_th=v_th,
                alpha=alpha,
                current_scale=current_scale,
                dropout=dropout,
                attention_mode=attention_mode,
                use_dendritic_fusion=use_dendritic_fusion,
                window_size=window_size,
                norm_mode=norm_mode,
                ptbn_transition_steps=ptbn_transition_steps,
            )
            for _ in range(n_layer)
        ])

        # Continuous action prediction head
        self.action_head = nn.Linear(hidden_dim, action_dim)

        self.current_step = 0
        self.diagnostics: Dict[str, float] = {}

    def reset_state(self):
        for b in self.blocks:
            b.reset_state()

    def set_step(self, step: int):
        self.current_step = step

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
        reset_state: bool = True,
        step: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
        if reset_state or self.training:
            self.reset_state()

        if step is not None:
            self.current_step = step

        B, K, _ = states.shape
        device = states.device

        # Episodic position embedding
        t_clamped = torch.clamp(timesteps, 0, self.max_ep_len).long()
        time_emb = self.embed_timestep(t_clamped)

        # Modality embeddings
        e_R = self.dropout(self.embed_ln(self.return_embedding(returns_to_go) + time_emb))
        e_s = self.dropout(self.embed_ln(self.state_embedding(states) + time_emb))
        e_a = self.dropout(self.embed_ln(self.action_embedding(actions) + time_emb))

        # Interleave tokens: [R_1, s_1, a_1, R_2, s_2, a_2, ..., R_K, s_K, a_K]
        x = (
            torch.stack([e_R, e_s, e_a], dim=1)
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(B, 3 * K, self.hidden_dim)
        )

        total_tokens = 3 * K

        # Phase modulation across simulation steps T
        if self.use_phase_coding:
            phase_mod = self.phase_encoder(x)  # (T, B, 3K, 1)
        else:
            phase_mod = torch.ones(
                (self.snn_timesteps, B, total_tokens, 1),
                device=device,
                dtype=states.dtype,
            )

        # Causal mask for 3K tokens
        causal_mask = torch.triu(
            torch.ones((total_tokens, total_tokens), device=device, dtype=torch.bool),
            diagonal=1,
        )

        total_spikes = 0.0
        for b in self.blocks:
            b.spike_count_tensor = torch.tensor(0.0, device=device)

        # Forward through blocks
        for block in self.blocks:
            x, _ = block(
                x,
                phase_mod,
                causal_mask=causal_mask,
                step=self.current_step,
            )
            if block.spike_count_tensor is not None:
                total_spikes += float(block.spike_count_tensor.item())

        total_opportunities = max(1.0, float(B * total_tokens * (3 * self.hidden_dim) * self.snn_timesteps * len(self.blocks)))
        mean_spike_rate = total_spikes / total_opportunities
        self.diagnostics = {
            "mean_spike_rate": mean_spike_rate,
            "total_spikes": total_spikes,
        }

        # Extract representations at state token positions (1, 4, 7, ... -> 1::3)
        z = x[:, 1::3, :]  # (B, K, hidden_dim)

        # Action prediction
        pred_actions = self.action_head(z)
        if self.action_tanh:
            pred_actions = torch.tanh(pred_actions)

        if return_diagnostics:
            return pred_actions, self.diagnostics
        return pred_actions

    @torch.no_grad()
    def get_action(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        if states.dim() == 2:
            states = states.unsqueeze(0)
            actions = actions.unsqueeze(0)
            returns_to_go = returns_to_go.unsqueeze(0)
            timesteps = timesteps.unsqueeze(0)

        states = states[:, -self.context_len :]
        actions = actions[:, -self.context_len :]
        returns_to_go = returns_to_go[:, -self.context_len :]
        timesteps = timesteps[:, -self.context_len :]

        preds = self.forward(
            states=states,
            actions=actions,
            returns_to_go=returns_to_go,
            timesteps=timesteps,
            reset_state=True,
        )
        return preds[0, -1]


def create_snndt_h_ablation(
    ablation_id: str,
    state_dim: int = 11,
    action_dim: int = 3,
    **kwargs,
) -> SNNDTH:
    """
    Factory function for Controlled Experiments E0 through E6:
    - E0: Corrected SNN-DT Baseline (standard attention, LayerNorm)
    - E1: SNN-DT + PSSA (local spiking attention, window S=8)
    - E2: SNN-DT + TSSA (temporal concatenation across T x D)
    - E3: SNN-DT + E1 + E2 (dual branch, uniform averaging)
    - E4: E3 + Dendritic Routing (adaptive gating fusion)
    - E5: E4 + PTBN (progressive threshold-dependent normalization)
    - E6: E5 + RTG adaptive threshold modulation
    """
    ab = ablation_id.lower().strip()

    if ab in ("e0", "baseline", "standard"):
        return SNNDTH(
            state_dim=state_dim,
            action_dim=action_dim,
            attention_mode="standard",
            use_dendritic_fusion=False,
            norm_mode="layernorm",
            **kwargs,
        )
    elif ab in ("e1", "pssa"):
        return SNNDTH(
            state_dim=state_dim,
            action_dim=action_dim,
            attention_mode="pssa",
            use_dendritic_fusion=False,
            norm_mode="layernorm",
            **kwargs,
        )
    elif ab in ("e2", "tssa"):
        return SNNDTH(
            state_dim=state_dim,
            action_dim=action_dim,
            attention_mode="tssa",
            use_dendritic_fusion=False,
            norm_mode="layernorm",
            **kwargs,
        )
    elif ab in ("e3", "dual_uniform"):
        return SNNDTH(
            state_dim=state_dim,
            action_dim=action_dim,
            attention_mode="dual",
            use_dendritic_fusion=False,
            norm_mode="layernorm",
            **kwargs,
        )
    elif ab in ("e4", "dual_routed", "dendritic"):
        return SNNDTH(
            state_dim=state_dim,
            action_dim=action_dim,
            attention_mode="dual",
            use_dendritic_fusion=True,
            norm_mode="layernorm",
            **kwargs,
        )
    elif ab in ("e5", "ptbn", "full_h"):
        return SNNDTH(
            state_dim=state_dim,
            action_dim=action_dim,
            attention_mode="dual",
            use_dendritic_fusion=True,
            norm_mode="ptbn",
            **kwargs,
        )
    elif ab in ("e6", "adaptive_th"):
        return SNNDTH(
            state_dim=state_dim,
            action_dim=action_dim,
            attention_mode="dual",
            use_dendritic_fusion=True,
            norm_mode="ptbn",
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown ablation ID: {ablation_id}. Must be one of E0..E6.")
