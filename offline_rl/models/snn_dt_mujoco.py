"""
SNN Decision Transformer for Continuous MuJoCo Offline RL Control.

Adapts and wraps the core neuromorphic SNN-DT research backbone:
- Biological time axis T
- Vectorized Leaky Integrate-and-Fire (LIF) neurons with surrogate gradient autograd
- PhaseSpikeEncoder: Sinusoidal current modulation across biological time
- DendriticRouter: Per-head dendritic routing with learned gating alpha_h
- ThreeFactorPlasticity: Biological local plasticity with eligibility traces
- Causal (Return, State, Action) sequence processing
- Continuous action prediction head with Tanh bounding for MuJoCo control
"""

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.snn_dt import (
    DendriticRouter,
    PhaseSpikeEncoder,
    SnnDtBackbone,
    SpikingTransformerBlock,
    SurrogateSpike,
    ThreeFactorPlasticity,
    VectorizedLIF,
    surrogate_spike,
)

# Canonical MuJoCo Environment Dimension Mapping (Section 8 Automatic Ingestion)
ENV_DIM_MAP: Dict[str, Tuple[int, int]] = {
    "hopper": (11, 3),
    "hopper-v2": (11, 3),
    "hopper-v3": (11, 3),
    "hopper-v4": (11, 3),
    "hopper-medium-v2": (11, 3),
    "hopper-medium-replay-v2": (11, 3),
    "hopper-medium-expert-v2": (11, 3),
    "halfcheetah": (17, 6),
    "halfcheetah-v2": (17, 6),
    "halfcheetah-v3": (17, 6),
    "halfcheetah-v4": (17, 6),
    "halfcheetah-medium-v2": (17, 6),
    "halfcheetah-medium-replay-v2": (17, 6),
    "halfcheetah-medium-expert-v2": (17, 6),
    "walker2d": (17, 6),
    "walker2d-v2": (17, 6),
    "walker2d-v3": (17, 6),
    "walker2d-v4": (17, 6),
    "walker2d-medium-v2": (17, 6),
    "walker2d-medium-replay-v2": (17, 6),
    "walker2d-medium-expert-v2": (17, 6),
    "ant": (27, 8),
    "ant-v2": (27, 8),
    "ant-v3": (27, 8),
    "ant-v4": (27, 8),
    "ant-medium-v2": (27, 8),
    "ant-medium-replay-v2": (27, 8),
    "ant-medium-expert-v2": (27, 8),
}


def get_env_dims(env_or_name: Union[str, Any, None]) -> Tuple[int, int]:
    """
    Automatically resolve (state_dim, action_dim) from a Gymnasium environment,
    environment name string, or configuration.
    Avoids hardcoding Hopper's state_dim=11 and action_dim=3 for general scaling.
    """
    if env_or_name is None:
        return 11, 3

    if hasattr(env_or_name, "observation_space") and hasattr(env_or_name, "action_space"):
        s_dim = int(env_or_name.observation_space.shape[0])
        a_dim = int(env_or_name.action_space.shape[0])
        return s_dim, a_dim

    if isinstance(env_or_name, str):
        key = env_or_name.strip().lower()
        if key in ENV_DIM_MAP:
            return ENV_DIM_MAP[key]
        for prefix, dims in ENV_DIM_MAP.items():
            if key.startswith(prefix):
                return dims
        try:
            import gymnasium as gym
            env = gym.make(env_or_name)
            s_dim = int(env.observation_space.shape[0])
            a_dim = int(env.action_space.shape[0])
            env.close()
            return s_dim, a_dim
        except Exception:
            pass

    return 11, 3


class SNNDTMuJoCo(nn.Module):
    """
    MuJoCo continuous control adapter for the SNN-DT research backbone.

    Conceptual architecture (Section 4.1):
        e_t^R = W_R R_t + b_R + e_t^time
        e_t^s = W_s s_t + b_s + e_t^time
        e_t^a = W_a a_t + b_a + e_t^time
        z_t = SNNDT(e_t^R, e_t^s, e_t^a, t)
        \\hat{a}_t = W_o z_t + b_o
    """

    def __init__(
        self,
        backbone: Optional[nn.Module] = None,
        hidden_dim: int = 128,
        state_dim: Optional[int] = None,
        action_dim: Optional[int] = None,
        act_dim: Optional[int] = None,
        embed_dim: Optional[int] = None,
        env: Optional[Any] = None,
        max_ep_len: int = 1000,
        context_len: int = 20,
        action_tanh: bool = True,
        n_layer: int = 3,
        n_head: int = 4,
        snn_timesteps: int = 4,
        tau: float = 20.0,
        v_th: float = 0.5,
        alpha: float = 25.0,
        current_scale: float = 1.0,
        use_plasticity: bool = False,
        dropout: float = 0.1,
        use_phase_coding: bool = True,
        use_dendritic_routing: bool = True,
        **kwargs,
    ):
        super().__init__()
        # Dynamic environment dimension ingestion (Phase 8 General Scaling)
        if env is not None:
            inferred_s, inferred_a = get_env_dims(env)
            if state_dim is None:
                state_dim = inferred_s
            if action_dim is None and act_dim is None:
                action_dim = inferred_a

        # Parameter aliases for flexible instantiation
        if act_dim is not None:
            action_dim = act_dim
        if embed_dim is not None:
            hidden_dim = embed_dim

        # General defaults if still unspecified (fallback)
        if state_dim is None:
            state_dim = 11
        if action_dim is None:
            action_dim = 3

        self.hidden_dim = hidden_dim
        self.embed_dim = hidden_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.act_dim = action_dim
        self.max_ep_len = max_ep_len
        self.context_len = context_len
        self.action_tanh = action_tanh
        self.snn_timesteps = snn_timesteps
        self.use_phase_coding = use_phase_coding
        self.use_dendritic_routing = use_dendritic_routing
        self.use_plasticity = use_plasticity

        # SNN-DT research backbone preserving LIF, phase coding, dendritic routing, plasticity
        if backbone is None:
            self.backbone = SnnDtBackbone(
                hidden_dim=hidden_dim,
                n_heads=n_head,
                n_layers=n_layer,
                snn_timesteps=snn_timesteps,
                tau=tau,
                v_th=v_th,
                alpha=alpha,
                current_scale=current_scale,
                use_plasticity=use_plasticity,
                use_phase_coding=use_phase_coding,
                use_dendritic_routing=use_dendritic_routing,
                dropout=dropout,
            )
        else:
            self.backbone = backbone

        # Modality embedding layers (Section 4.1):
        # e_t^R = W_R R_t + b_R
        # e_t^s = W_s s_t + b_s
        # e_t^a = W_a a_t + b_a
        self.state_embedding = nn.Linear(state_dim, hidden_dim)
        self.action_embedding = nn.Linear(action_dim, hidden_dim)
        self.return_embedding = nn.Linear(1, hidden_dim)

        # Episodic position timestep embedding
        self.embed_timestep = nn.Embedding(max_ep_len + 1, hidden_dim)

        self.embed_ln = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Action prediction head: \\hat{a}_t = W_o z_t + b_o
        self.action_head = nn.Linear(hidden_dim, action_dim)

        # Aliases for backward compatibility
        self.head = self.action_head
        self.embed_state = self.state_embedding
        self.embed_action = self.action_embedding
        self.embed_return = self.return_embedding
        self.blocks = getattr(self.backbone, "blocks", nn.ModuleList())

        self.diagnostics: Dict[str, float] = {}

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
        reset_state: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
        """
        Forward pass conforming to required Section 4.1 interface:
        Inputs:
            states         # [B, K, state_dim]
            actions        # [B, K, action_dim]
            returns_to_go  # [B, K, 1]
            timesteps      # [B, K]
            attention_mask # [B, K] (optional)
            return_diagnostics # bool (default False)
            reset_state    # bool (default True) - resets LIF membrane state per batch
        Outputs:
            predicted_actions # [B, K, action_dim] (default)
            (or tuple (predicted_actions, diagnostics) if return_diagnostics=True)
        """
        if reset_state or self.training:
            self.reset_state()
        B, K, _ = states.shape
        device = states.device

        # Episodic position embedding
        t_clamped = torch.clamp(timesteps, 0, self.max_ep_len).long()
        time_emb = self.embed_timestep(t_clamped)  # (B, K, hidden_dim)

        # Modality embeddings + timestep embedding (Section 4.1)
        e_R = self.return_embedding(returns_to_go) + time_emb
        e_s = self.state_embedding(states) + time_emb
        e_a = self.action_embedding(actions) + time_emb

        e_R = self.embed_ln(e_R)
        e_s = self.embed_ln(e_s)
        e_a = self.embed_ln(e_a)

        e_R = self.dropout(e_R)
        e_s = self.dropout(e_s)
        e_a = self.dropout(e_a)

        # Pass through SNN-DT research backbone: z_t = SNNDT(e_t^R, e_t^s, e_t^a, t)
        if hasattr(self.backbone, "forward_backbone"):
            z = self.backbone.forward_backbone(
                returns_embeddings=e_R,
                state_embeddings=e_s,
                action_embeddings=e_a,
                timesteps=timesteps,
                attention_mask=attention_mask,
            )
        else:
            z = self.backbone(
                returns_embeddings=e_R,
                state_embeddings=e_s,
                action_embeddings=e_a,
                timesteps=timesteps,
                attention_mask=attention_mask,
            )

        # Continuous action prediction head: \\hat{a}_t = W_o z_t + b_o
        predicted_actions = self.action_head(z)  # (B, K, action_dim)

        if self.action_tanh:
            predicted_actions = torch.tanh(predicted_actions)

        # Neuromorphic diagnostics
        self.diagnostics = getattr(self.backbone, "diagnostics", {
            "mean_spike_rate": 0.0,
            "total_spikes": 0.0,
        })

        if return_diagnostics:
            return predicted_actions, self.diagnostics
        return predicted_actions

    @torch.no_grad()
    def get_action(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate step in rollout loop. Returns action prediction for latest state.
        """
        if states.dim() == 2:
            states = states.unsqueeze(0)
            actions = actions.unsqueeze(0)
            returns_to_go = returns_to_go.unsqueeze(0)
            timesteps = timesteps.unsqueeze(0)

        # Truncate context to context_len
        states = states[:, -self.context_len :]
        actions = actions[:, -self.context_len :]
        returns_to_go = returns_to_go[:, -self.context_len :]
        timesteps = timesteps[:, -self.context_len :]

        preds = self.forward(
            states=states,
            actions=actions,
            returns_to_go=returns_to_go,
            timesteps=timesteps,
            attention_mask=None,
            return_diagnostics=False,
        )
        return preds[0, -1]

    def reset_state(self):
        """
        Architecture-specific state handling (Critical Correction 3):
        Recursively resets LIF membrane potentials and eligibility traces
        to ensure one sampled trajectory does not contaminate another in offline RL.
        """
        if hasattr(self.backbone, "reset_state"):
            self.backbone.reset_state()
        for m in self.modules():
            if m is not self and hasattr(m, "reset_state"):
                m.reset_state()

    @classmethod
    def from_env(cls, env: Any, **kwargs) -> "SNNDTMuJoCo":
        """Instantiate model automatically configuring state_dim and action_dim from environment."""
        s_dim, a_dim = get_env_dims(env)
        return cls(state_dim=s_dim, action_dim=a_dim, env=env, **kwargs)

    def reset_spike_counts(self):
        if hasattr(self.backbone, "reset_spike_counts"):
            self.backbone.reset_spike_counts()

    def count_spikes(self) -> float:
        if hasattr(self.backbone, "count_spikes"):
            return self.backbone.count_spikes()
        return self.diagnostics.get("mean_spike_rate", 0.0)

    def apply_plasticity(self, reward: float):
        if hasattr(self.backbone, "apply_plasticity"):
            self.backbone.apply_plasticity(reward)


class ContinuousSnnDecisionTransformer(SNNDTMuJoCo):
    """
    Subclass for continuous SNN Decision Transformer on MuJoCo,
    providing full backward compatibility with offline_rl training & evaluation scripts.
    """

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_diagnostics: bool = True,
        reset_state: bool = True,
    ):
        return super().forward(
            states=states,
            actions=actions,
            returns_to_go=returns_to_go,
            timesteps=timesteps,
            attention_mask=attention_mask,
            return_diagnostics=return_diagnostics,
            reset_state=reset_state,
        )

    @classmethod
    def from_env(cls, env: Any, **kwargs) -> "ContinuousSnnDecisionTransformer":
        """Instantiate model automatically configuring state_dim and act_dim from environment."""
        s_dim, a_dim = get_env_dims(env)
        return cls(state_dim=s_dim, act_dim=a_dim, env=env, **kwargs)


def create_ablation_model(
    ablation_type: str = "full",
    state_dim: Optional[int] = None,
    act_dim: Optional[int] = None,
    env: Optional[Any] = None,
    **kwargs,
) -> ContinuousSnnDecisionTransformer:
    """
    Factory function for Phase 7.1 Ablation Study & Phase 8 Scaling models:
    - "full" (SNN-DT Full): Phase coding=✓, Dendritic routing=✓, Local plasticity=✓
    - "ablation_a": Phase coding=✗, Dendritic routing=✓, Local plasticity=✓
    - "ablation_b": Phase coding=✓, Dendritic routing=✗, Local plasticity=✓
    - "ablation_c": Phase coding=✓, Dendritic routing=✓, Local plasticity=✗
    """
    if env is not None:
        env_s, env_a = get_env_dims(env)
        if state_dim is None:
            state_dim = env_s
        if act_dim is None:
            act_dim = env_a

    state_dim = state_dim if state_dim is not None else 11
    act_dim = act_dim if act_dim is not None else 3

    ablation_norm = ablation_type.lower().replace("-", "_").replace(" ", "_")
    if ablation_norm in ("full", "snn_dt_full", "snndt_full"):
        phase_coding = True
        dendritic_routing = True
        plasticity = True
    elif ablation_norm in ("ablation_a", "a", "no_phase_coding"):
        phase_coding = False
        dendritic_routing = True
        plasticity = True
    elif ablation_norm in ("ablation_b", "b", "no_dendritic_routing"):
        phase_coding = True
        dendritic_routing = False
        plasticity = True
    elif ablation_norm in ("ablation_c", "c", "no_local_plasticity"):
        phase_coding = True
        dendritic_routing = True
        plasticity = False
    else:
        raise ValueError(
            f"Unknown ablation_type: '{ablation_type}'. "
            f"Must be one of: 'full', 'ablation_a', 'ablation_b', 'ablation_c'."
        )

    return ContinuousSnnDecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        use_phase_coding=phase_coding,
        use_dendritic_routing=dendritic_routing,
        use_plasticity=plasticity,
        **kwargs,
    )

