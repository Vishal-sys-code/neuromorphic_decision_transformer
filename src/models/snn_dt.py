import math
import torch
import torch.nn as nn
from external.decision_transformer.models.decision_transformer import DecisionTransformer, TransformerBlock
from models.spiking_layers import SpikingSelfAttention

class SpikingTransformerBlock(nn.Module):
    def __init__(self, block: TransformerBlock, time_window: int):
        super().__init__()
        # reuse the original layer norms
        self.ln1 = block.ln1
        self.ln2 = block.ln2
        # spiking self‑attention in place of block.attn
        self.snn_attn = SpikingSelfAttention(
            embed_dim=block.hidden_size,
            num_heads=block.n_head,
            time_window=time_window
        )
        # keep the original feed‑forward (FFN) as is
        self.ff = block.ff

    def forward(self, x):
        # Self‑Attention pass
        a = self.snn_attn(self.ln1(x))      # [B, S, E]
        x = x + a                           # residual
        # Feed‑forward pass
        f = self.ff(self.ln2(x))           # [B, S, E]
        return x + f                       # residual

class SNNDecisionTransformer(DecisionTransformer):
    def __init__(self, *args, time_window: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace each vanilla TransformerBlock with our Spiking version
        self.transformer = nn.ModuleList([
            SpikingTransformerBlock(block, time_window)
            for block in self.transformer
        ])
