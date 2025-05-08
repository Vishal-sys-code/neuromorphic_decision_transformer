import torch
import torch.nn as nn
from src.models.spiking_layers import SpikingSelfAttention

# Spiking GPT2 Attention
class SpikingGPT2Attention(nn.Module):
    def __init__(self, orig_attn, time_window):
        """
        orig_attn: the original GPT2Attention module (we’ll re‑use its c_proj if desired)
        """
        super().__init__()
        self.time_window = time_window
        self.embed_dim = orig_attn.n_embd
        self.num_heads = orig_attn.n_head

        # your spike‑based MHA
        self.snn_attn = SpikingSelfAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            time_window=self.time_window,
        )

        # reuse the original linear that projects back to embed_dim (optional)
        self.c_proj = orig_attn.c_proj

    def forward(self, hidden_states, layer_past=None, attention_mask=None, use_cache=False, **kwargs):
        # hidden_states: [B, S, E]
        # 1) run spiking attention
        out = self.snn_attn(hidden_states)  # [B, S, E]

        # 2) optional: project through the original c_proj to match GPT2’s pipeline
        out = self.c_proj(out)

        # 3) GPT2Attention returns (attn_output, present); we don’t support caching so present=None
        return out
