import torch
import torch.nn as nn
from ...models.adaptive_attention import AdaptiveSpikingAttention

class SpikingTransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, T_max=20, lambda_reg=1e-3):
        super().__init__()
        
        # 🔥 Replace standard attention with adaptive spiking attention
        self.attention = AdaptiveSpikingAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            T_max=T_max,
            lambda_reg=lambda_reg
        )
        
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(0.1)
        )
    
    def forward(self, x, mask=None):
        # Adaptive spiking attention
        attn_out, metrics = self.attention(self.norm1(x), mask)
        x = x + attn_out
        
        # Feed-forward network
        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out
        
        return x, metrics