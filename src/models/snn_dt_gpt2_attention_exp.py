"""
Author: Vishal Pandey
Email: pandeyvishal.mlprof@gmail.com

SNNDecisionTransformer
----------------------
Extends the standard Decision Transformer by replacing GPT-2 attention blocks
with SpikingGPT2Attention modules. This enables neuromorphic temporal coding
and spike-based efficiency tracking while preserving the DT autoregressive
sequence modeling behavior.

Exposed APIs:
- get_spike_count(): Total spikes across all LIFNeuronLayer modules.
- reset_spike_count(): Clears spike counters (called automatically on forward).
- get_spike_breakdown(): Per-layer spike counts.
- estimate_energy(): Convert spikes to energy given per-spike cost.

Intended usage:
model = SNNDecisionTransformer(...)
out = model(input)   # automatically resets and counts spikes
spikes = model.get_spike_count()
energy = model.estimate_energy(per_spike_energy=4.6e-12)
"""

import torch
import torch.nn as nn

# ---------------------------------------------------------------------
# Try importing DecisionTransformer from external repo; fallback to src
# ---------------------------------------------------------------------
try:
    from external.decision_transformer.gym.decision_transformer.models.decision_transformer import (
        DecisionTransformer,
    )
    print("[INFO] Using DecisionTransformer from external/decision_transformer")
except ImportError:
    from src.models.dsf_models.decision_transformer import DecisionTransformer
    print("[INFO] Using fallback DecisionTransformer from src/models/dsf_models")

from src.models.spiking_gpt2_attention import SpikingGPT2Attention
from src.models.snn_lif import LIFNeuronLayer


class SNNDecisionTransformer(DecisionTransformer):
    """
    Spiking Decision Transformer.

    This class overrides the attention mechanism in GPT-2 blocks with
    SpikingGPT2Attention while keeping the rest of the Decision Transformer
    pipeline intact. It also tracks and exposes spike activity for
    neuromorphic efficiency evaluation.
    """

    def __init__(
        self,
        state_dim,
        act_dim,
        hidden_size,
        max_length=None,
        max_ep_len=4096,
        action_tanh=True,
        time_window: int = 10,
        **kwargs
    ):
        super().__init__(
            state_dim=state_dim,
            act_dim=act_dim,
            hidden_size=hidden_size,
            max_length=max_length,
            max_ep_len=max_ep_len,
            action_tanh=action_tanh,
            **kwargs
        )

        # Swap GPT2 attention for spiking attention
        for block in self.transformer.h:  # h = list of GPT2Block
            orig_attn = block.attn
            block.attn = SpikingGPT2Attention(orig_attn, time_window)

    # -----------------------------------------------------------------
    # Spike monitoring APIs
    # -----------------------------------------------------------------
    def reset_spike_count(self):
        """Reset spike counters in all LIFNeuronLayer modules to zero."""
        for module in self.modules():
            if isinstance(module, LIFNeuronLayer):
                module.reset_spike_count()

    def get_spike_count(self):
        """Return total spike count across all LIFNeuronLayer modules."""
        total_spikes = 0
        for module in self.modules():
            if isinstance(module, LIFNeuronLayer):
                total_spikes += module.spike_count
        return total_spikes

    def get_spike_breakdown(self):
        """
        Return a dict mapping layer names to spike counts.
        Useful for diagnosing which layers dominate spike activity.
        """
        breakdown = {}
        for name, module in self.named_modules():
            if isinstance(module, LIFNeuronLayer):
                breakdown[name] = module.spike_count
        return breakdown

    def estimate_energy(self, per_spike_energy=4.6e-12):
        """
        Estimate energy cost given a per-spike energy constant (Joules).
        Default: 4.6 pJ/spike (45nm process).
        """
        return self.get_spike_count() * per_spike_energy

    # -----------------------------------------------------------------
    # Override forward to auto-reset spike counters
    # -----------------------------------------------------------------
        # -----------------------------------------------------------------
    # Override forward to auto-reset spike counters + dtype safety
    # -----------------------------------------------------------------
    def forward(self, states, actions, returns_to_go, timesteps, attention_mask=None, **kwargs):
        """
        Forward pass with spike reset and automatic dtype fix for actions.

        Args:
            states: [batch, seq, state_dim]
            actions: [batch, seq, act_dim] (float one-hot) or [batch, seq] (long indices)
            returns_to_go: [batch, seq, 1]
            timesteps: [batch, seq]
            attention_mask: optional mask
        """
        # reset spike counts each forward call
        self.reset_spike_count()

        # auto-convert discrete (Long) actions to one-hot float
        if actions is not None and actions.dtype == actions.long().dtype:
            actions = torch.nn.functional.one_hot(
                actions, num_classes=self.act_dim
            ).float()

        return super().forward(
            states,
            actions,
            returns_to_go,
            timesteps,
            attention_mask=attention_mask,
            **kwargs,
        )
