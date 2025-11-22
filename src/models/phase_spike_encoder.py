import torch
import torch.nn as nn

class PhaseSpikeEncoder(nn.Module):
    def __init__(self, d_model, n_heads, T):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.T = T
        
        # Learnable parameters per head
        self.omega = nn.Parameter(torch.randn(n_heads))
        self.phi = nn.Parameter(torch.randn(n_heads))
        
    def forward(self, timesteps):
        """
        Args:
            timesteps: (batch, seq_len)
        Returns:
            spikes: (batch, seq_len, d_model)
        """
        # timesteps: (batch, seq_len)
        
        t = timesteps.float().unsqueeze(-1) # (B, L, 1)
        
        # omega, phi -> (1, 1, n_heads)
        w = self.omega.view(1, 1, -1)
        p = self.phi.view(1, 1, -1)
        
        # s_k(t) = (sin(omega_k * t + phi_k) > 0).float()
        wave = torch.sin(t * w + p) # (B, L, n_heads)
        
        if self.training:
             # Use sigmoid approximation for gradient flow
             spikes = torch.sigmoid(wave * 10.0) # Scale up to make it steeper
        else:
             spikes = (wave > 0).float()
        
        # Tile along feature dimension so total features = d_model
        repeats = self.d_model // self.n_heads
        spikes = spikes.repeat_interleave(repeats, dim=-1) # (B, L, n_heads * repeats)
        
        # Handle remainder if d_model is not perfectly divisible
        if spikes.shape[-1] != self.d_model:
            padding = self.d_model - spikes.shape[-1]
            if padding > 0:
                # Pad with first few features to match d_model
                spikes = torch.cat([spikes, spikes[:, :, :padding]], dim=-1)
                
        return spikes