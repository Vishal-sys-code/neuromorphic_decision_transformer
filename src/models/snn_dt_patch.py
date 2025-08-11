"""
Author: Vishal Pandey [X: its_vayishu]
Email: pandeyvishal.mlprof@gmail.com
"""

import torch.nn as nn
from external.decision_transformer.gym.decision_transformer.models.decision_transformer import DecisionTransformer
from src.models.spiking_gpt2_attention import SpikingGPT2Attention
from transformers.activations import ACT2FN
from transformers.modeling_utils import Conv1D

class SNNDecisionTransformer(DecisionTransformer):
    def __init__(self, *args, time_window: int = 10, **kwargs):
        super().__init__(*args, **kwargs)

        # Walk every block in the GPT2Model to make it compatible
        for block in self.transformer.h:
            # 1. Swap in the spiking attention mechanism
            orig_attn = block.attn
            block.attn = SpikingGPT2Attention(orig_attn, time_window)

            # 2. Re-structure the MLP to match the checkpoint's nn.Sequential format
            # This is necessary because the saved checkpoint has a different MLP structure
            # than the one defined in the original trajectory_gpt2.py.
            inner_dim = block.mlp.c_fc.nf
            hidden_size = block.mlp.c_proj.nf
            config = self.transformer.config
            
            # The checkpoint expects an nn.Sequential with numbered layers ("0", "2"),
            # so we build it here to match.
            block.mlp = nn.Sequential(
                Conv1D(inner_dim, hidden_size),
                ACT2FN[config.activation_function],
                Conv1D(hidden_size, inner_dim),
                nn.Dropout(config.resid_pdrop)
            )