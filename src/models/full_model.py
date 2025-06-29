import torch
import torch.nn as nn
from .transformer_blocks import SpikingTransformerBlock

class AdaptiveSpikingTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim=512, num_heads=8, 
                 num_layers=6, T_max=20, lambda_reg=1e-3):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1000, embedding_dim))
        
        # Stack of adaptive spiking transformer blocks
        self.layers = nn.ModuleList([
            SpikingTransformerBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads, 
                T_max=T_max,
                lambda_reg=lambda_reg
            ) for _ in range(num_layers)
        ])
        
        self.output_head = nn.Linear(embedding_dim, vocab_size)
        
    def forward(self, x, mask=None):
        seq_len = x.size(1)
        
        # Embeddings + positional encoding
        x = self.embedding(x) + self.pos_encoding[:seq_len]
        
        # Collect metrics from all layers
        all_metrics = []
        
        # Pass through adaptive spiking layers
        for layer in self.layers:
            x, metrics = layer(x, mask)
            all_metrics.append(metrics)
        
        # Output projection
        logits = self.output_head(x)
        
        return logits, all_metrics