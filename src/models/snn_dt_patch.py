"""
Author: Vishal Pandey [X: its_vayishu]
Email: pandeyvishal.mlprof@gmail.com
"""

import torch.nn as nn
from external.decision_transformer.gym.decision_transformer.models.decision_transformer import DecisionTransformer
from src.models.spiking_gpt2_attention import SpikingGPT2Attention

class SNNDecisionTransformer(DecisionTransformer):
    def __init__(self, *args, time_window: int = 10, **kwargs):
        # Patch config to add n_ctx if missing
        if 'config' in kwargs:
            config = kwargs['config']
            if not hasattr(config, 'n_ctx'):
                if hasattr(config, 'n_positions'):
                    config.n_ctx = config.n_positions
                else:
                    config.n_ctx = kwargs.get('max_length', 1024)
        super().__init__(*args, **kwargs)

        # walk every block in GPT2Model
        for block in self.transformer.h:  # `h` is the list of GPT2Block
            orig_attn = block.attn
            # swap in your spiking version
            block.attn = SpikingGPT2Attention(orig_attn, time_window)
