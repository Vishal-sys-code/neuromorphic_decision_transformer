import os, sys
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ext = os.path.join(root, "external")
if ext not in sys.path:
    sys.path.insert(0, ext)
import math
import torch
import torch.nn as nn
from external.decision_transformer.gym.decision_transformer.models.decision_transformer import DecisionTransformer
from external.decision_transformer.gym.decision_transformer.models.trajectory_gpt2 import TransformerBlock as TransformerBlock
from src.models.spiking_layers import SpikingSelfAttention

class SpikingTransformerBlock(nn.Module):
    def __init__(self, block: TransformerBlock, time_window: int):
        super().__init__()
        # reuse the original layer norms
        self.ln1 = block.ln_1
        self.ln2 = block.ln_2
        # spiking self‑attention in place of block.attn
        self.snn_attn = SpikingSelfAttention(
            embed_dim=block.ln_1.normalized_shape[0],
            num_heads=block.attn.n_head,
            time_window=time_window
        )
        # keep the original feed‑forward (FFN) as is
        self.ff = block.mlp

    def forward(self, hidden_states, *args, **kwargs):
        # Self‑Attention pass
        a = self.snn_attn(self.ln1(hidden_states))
        x = hidden_states + a               # residual
        # Feed‑forward pass
        f = self.ff(self.ln2(x))           # [B, S, E]
        hidden_states = x + f              # residual
        # Return only the hidden_states tensor to match expected output
        return hidden_states

class SNNDecisionTransformer(DecisionTransformer):
    def __init__(self, *args, time_window: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace each vanilla TransformerBlock with our Spiking version
        self.transformer.h = nn.ModuleList([
            SpikingTransformerBlock(block, time_window)
            for block in self.transformer.h
        ])
