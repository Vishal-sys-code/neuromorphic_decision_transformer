"""
Spiking Decision Transformer for Continuous MuJoCo Control.

Incorporates:
- Vectorized Leaky Integrate-and-Fire (LIF) neurons with surrogate gradient autograd.
- Configurable biological time T (neuromorphic time axis).
- Continuous action regression head with Tanh bounded output.
- Interleaved (Return, State, Action) causal trajectory processing.
- Neuromorphic energy & spike rate monitoring for benchmark diagnostics.
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SurrogateSpike(torch.autograd.Function):
    """
    Forward: Heaviside step thresholding -> discrete spikes {0, 1}.
    Backward: Fast sigmoid surrogate gradient approximation for BPTT.
    """

    @staticmethod
    def forward(ctx, membrane_potential: torch.Tensor, v_th: float, alpha: float) -> torch.Tensor:
        ctx.save_for_backward(membrane_potential)
        ctx.v_th = float(v_th)
        ctx.alpha = float(alpha)
        return (membrane_potential >= v_th).to(membrane_potential.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (v,) = ctx.saved_tensors
        v_th = ctx.v_th
        alpha = ctx.alpha
        # Sigmoid surrogate gradient: d/dv [sigma(alpha * (v - v_th))]
        sigma = torch.sigmoid(alpha * (v - v_th))
        grad_v = alpha * sigma * (1.0 - sigma) * grad_output
        return grad_v, None, None


surrogate_spike = SurrogateSpike.apply


class VectorizedLIF(nn.Module):
    """
    Vectorized Leaky Integrate-and-Fire (LIF) dynamics over biological timesteps T.
    
    Inputs:
        current: (B, L, D) injection current
    Outputs:
        spike_rate: (B, L, D) integrated spike rate across biological time T
        spikes: (T, B, L, D) binary spike rasters across biological time T
    """

    def __init__(
        self,
        dim: int,
        timesteps: int = 4,
        tau: float = 20.0,
        v_th: float = 0.5,
        alpha: float = 25.0,
        dt: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.timesteps = timesteps
        self.tau = tau
        self.v_th = v_th
        self.alpha = alpha
        self.dt = dt
        # Membrane decay factor rho in [0, 1]
        self.decay = math.exp(-dt / max(tau, 1e-4))

    def forward(self, current: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # current: (B, L, D)
        B, L, D = current.shape
        v = torch.zeros(B, L, D, device=current.device, dtype=current.dtype)
        spikes_list = []

        for _ in range(self.timesteps):
            # Integrate membrane potential with leak
            v = self.decay * v + current
            s = surrogate_spike(v, self.v_th, self.alpha)
            # Soft/hard reset: subtract threshold * spike
            v = v - self.v_th * s
            spikes_list.append(s)

        spikes = torch.stack(spikes_list, dim=0)  # (T, B, L, D)
        spike_rate = torch.mean(spikes, dim=0)     # (B, L, D)
        return spike_rate, spikes


class SpikingTransformerBlock(nn.Module):
    """
    Transformer block with causal multi-head self-attention and
    spiking neuromorphic feed-forward network (SNN-FFN).
    """

    def __init__(
        self,
        embed_dim: int,
        n_head: int,
        dropout: float = 0.1,
        snn_timesteps: int = 4,
        tau: float = 20.0,
        v_th: float = 0.5,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_head = n_head
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            n_head,
            dropout=dropout,
            batch_first=True,
        )

        self.ln_2 = nn.LayerNorm(embed_dim)
        self.mlp_up = nn.Linear(embed_dim, 4 * embed_dim)
        self.lif = VectorizedLIF(
            dim=4 * embed_dim,
            timesteps=snn_timesteps,
            tau=tau,
            v_th=v_th,
        )
        self.mlp_down = nn.Linear(4 * embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self-Attention with pre-LayerNorm
        norm_x = self.ln_1(x)
        attn_out, _ = self.attn(
            norm_x,
            norm_x,
            norm_x,
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.dropout(attn_out)

        # Spiking Neuromorphic FFN with pre-LayerNorm
        norm_x2 = self.ln_2(x)
        current = self.mlp_up(norm_x2)
        spike_rate, spikes = self.lif(current)
        ffn_out = self.mlp_down(spike_rate)
        x = x + self.dropout(ffn_out)

        return x, spikes


class ContinuousSnnDecisionTransformer(nn.Module):
    """
    Continuous Spiking Decision Transformer for offline RL continuous control.
    Conditioned on (Returns-to-go, State, Action) interleaved sequences.
    """

    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        embed_dim: int = 128,
        n_layer: int = 3,
        n_head: int = 4,
        max_ep_len: int = 1000,
        context_len: int = 20,
        snn_timesteps: int = 4,
        tau: float = 20.0,
        v_th: float = 0.5,
        dropout: float = 0.1,
        action_tanh: bool = True,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.embed_dim = embed_dim
        self.context_len = context_len
        self.max_ep_len = max_ep_len
        self.snn_timesteps = snn_timesteps
        self.action_tanh = action_tanh

        # Modality embeddings
        self.embed_return = nn.Linear(1, embed_dim)
        self.embed_state = nn.Linear(state_dim, embed_dim)
        self.embed_action = nn.Linear(act_dim, embed_dim)

        # Positional embedding for episodic timesteps
        self.embed_timestep = nn.Embedding(max_ep_len + 1, embed_dim)

        self.embed_ln = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Spiking transformer backbone
        self.blocks = nn.ModuleList(
            [
                SpikingTransformerBlock(
                    embed_dim=embed_dim,
                    n_head=n_head,
                    dropout=dropout,
                    snn_timesteps=snn_timesteps,
                    tau=tau,
                    v_th=v_th,
                )
                for _ in range(n_layer)
            ]
        )

        self.ln_f = nn.LayerNorm(embed_dim)

        # Continuous action head (predicting action from state representation)
        self.head = nn.Linear(embed_dim, act_dim)

    def forward(
        self,
        states: torch.Tensor,           # (B, K, state_dim)
        actions: torch.Tensor,          # (B, K, act_dim)
        returns_to_go: torch.Tensor,    # (B, K, 1)
        timesteps: torch.Tensor,        # (B, K)
        attention_mask: Optional[torch.Tensor] = None,  # (B, K)
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        batch_size, seq_len = states.shape[0], states.shape[1]

        # Clamp timesteps to valid embedding range
        timesteps = torch.clamp(timesteps, 0, self.max_ep_len)
        time_embeddings = self.embed_timestep(timesteps)

        # Compute modality embeddings and add positional timestep embeddings
        returns_embeddings = self.embed_return(returns_to_go) + time_embeddings
        state_embeddings = self.embed_state(states) + time_embeddings
        action_embeddings = self.embed_action(actions) + time_embeddings

        # Interleave tokens: [R_1, s_1, a_1, R_2, s_2, a_2, ..., R_K, s_K, a_K]
        # Shape: (B, 3, K, D) -> transpose to (B, K, 3, D) -> reshape (B, 3*K, D)
        token_embeddings = (
            torch.stack([returns_embeddings, state_embeddings, action_embeddings], dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * seq_len, self.embed_dim)
        )
        x = self.embed_ln(token_embeddings)
        x = self.dropout(x)

        # Construct boolean causal attention mask (3*K, 3*K)
        # True means masked (cannot attend), False means attend
        total_tokens = 3 * seq_len
        causal_mask = torch.triu(
            torch.ones((total_tokens, total_tokens), device=states.device, dtype=torch.bool),
            diagonal=1,
        )

        # Construct key_padding_mask if attention_mask provided
        # attention_mask: (B, K) where 1 is valid, 0 is padding
        key_padding_mask = None
        if attention_mask is not None:
            # Repeat mask 3 times for (R, s, a): (B, K) -> (B, 3*K)
            interleaved_mask = (
                attention_mask.unsqueeze(2)
                .repeat(1, 1, 3)
                .reshape(batch_size, total_tokens)
            )
            # PyTorch MultiheadAttention key_padding_mask: True indicates masked out
            key_padding_mask = interleaved_mask == 0

        total_spikes = 0.0
        total_neurons = 0.0

        for block in self.blocks:
            x, spikes = block(
                x,
                causal_mask=causal_mask,
                key_padding_mask=key_padding_mask,
            )
            total_spikes += spikes.sum().item()
            total_neurons += spikes.numel()

        x = self.ln_f(x)

        # Extract predictions for actions
        # Tokens are: [R_1 (idx 0), s_1 (idx 1), a_1 (idx 2), R_2 (idx 3), s_2 (idx 4), ...]
        # State tokens are at indices: 1, 4, 7, ..., 3*k + 1
        # Predict action a_t from state representation s_t
        state_repr = x[:, 1::3, :]  # (B, K, embed_dim)
        action_preds = self.head(state_repr)

        if self.action_tanh:
            action_preds = torch.tanh(action_preds)

        mean_spike_rate = total_spikes / max(total_neurons, 1.0)
        diagnostics = {
            "mean_spike_rate": mean_spike_rate,
            "total_spikes": total_spikes,
        }

        return action_preds, diagnostics

    @torch.no_grad()
    def get_action(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate step in rollout loop. Returns the action prediction for the latest state.
        Inputs may be unbatched (K, D) or batched (1, K, D).
        """
        if states.dim() == 2:
            states = states.unsqueeze(0)
            actions = actions.unsqueeze(0)
            returns_to_go = returns_to_go.unsqueeze(0)
            timesteps = timesteps.unsqueeze(0)

        # Truncate to context_len if needed
        states = states[:, -self.context_len :]
        actions = actions[:, -self.context_len :]
        returns_to_go = returns_to_go[:, -self.context_len :]
        timesteps = timesteps[:, -self.context_len :]

        preds, _ = self.forward(
            states=states,
            actions=actions,
            returns_to_go=returns_to_go,
            timesteps=timesteps,
            attention_mask=None,
        )
        # Return action corresponding to the latest state token
        return preds[0, -1]
