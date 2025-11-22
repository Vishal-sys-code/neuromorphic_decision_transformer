import torch
import torch.nn as nn

class DendriticRouter(nn.Module):
    def __init__(self, d_model, n_heads, hidden=64):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        # MLP gate g in R^H from concatenated per-head outputs
        # Input to MLP: (B, L, H * head_dim) = (B, L, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_heads)
        )
        
    def forward(self, y_heads):
        """
        Args:
            y_heads: (B, L, H, head_dim) - Per-head outputs
        Returns:
            y_routed: (B, L, head_dim) - Gated and summed output
            gates: (B, L, H) - Routing coefficients
        """
        batch, seq, n_heads, head_dim = y_heads.shape
        
        # Concatenate per-head outputs for MLP input
        # (B, L, H, head_dim) -> (B, L, H * head_dim)
        y_concat = y_heads.reshape(batch, seq, -1)
        
        # Compute MLP gate
        gates = self.mlp(y_concat) # (B, L, n_heads)
        
        # Apply softmax to get dendritic routing coefficients alpha_h
        alpha = torch.softmax(gates, dim=-1) # (B, L, n_heads)
        
        # Apply gating before summing heads: y_routed = Sum_h alpha_h * y_h
        # alpha: (B, L, n_heads) -> (B, L, n_heads, 1)
        y_weighted = y_heads * alpha.unsqueeze(-1)
        
        # Sum heads
        y_routed = y_weighted.sum(dim=2) # (B, L, head_dim)
        
        return y_routed, alpha
