import torch
import torch.nn as nn
from novel_phases.phase3.positional_spike_encoder import PositionalSpikeEncoder
from novel_phases.phase3.dendritic_routing import DendriticRouter
from src.models.snn_lif import LIFNeuronLayer

class RateCoder(nn.Module):
    def __init__(self, embed_dim, window_length):
        super().__init__()
        self.T = window_length
        self.linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, x_embed): # x_embed: [B, L, d]
        projected_val = self.linear(x_embed) # [B, L, d]
        rates = torch.sigmoid(projected_val) # Scale to [0,1] to act as rates [B,L,d]
        spike_trains = torch.bernoulli(rates.unsqueeze(-1).expand(-1,-1,-1,self.T)) # [B,L,d,T]
        return spike_trains

class SNNDecisionTransformer(nn.Module):
    def __init__(self, embed_dim: int = 128, num_heads: int = 4, window_length: int = 10, num_layers: int = 1, use_pos_encoder: bool = True, use_router: bool = True): # Added num_layers
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.T = window_length
        self.num_layers = num_layers # Store num_layers
        self.use_pos_encoder = use_pos_encoder
        self.use_router = use_router

        self.rate_coder = RateCoder(self.embed_dim, self.T)

        if self.use_pos_encoder:
            self.pos_encoder = PositionalSpikeEncoder(num_heads=self.num_heads,
                                                      window_length=self.T)
        else:
            self.pos_encoder = None # Or nn.Identity() if it needs to be callable but do nothing

        self.spiking_attention_layers = nn.ModuleList([
            LIFNeuronLayer(self.embed_dim, self.embed_dim) for _ in range(self.num_layers)
        ])

        if self.use_router:
            self.router = DendriticRouter(num_heads=self.num_heads)
        else:
            self.router = None # Or nn.Identity() if it needs to be callable but do nothing

        # Placeholder for feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 4),
            nn.ReLU(),
            nn.Linear(self.embed_dim * 4, self.embed_dim)
        )
        # Layer normalization
        self.ln1 = nn.LayerNorm(self.embed_dim)
        self.ln2 = nn.LayerNorm(self.embed_dim)


    def forward(self, embeddings: torch.Tensor): # Input is embeddings [B, L, d]
        x = embeddings # Input to the first layer
        
        self.reset_spike_count()

        for i in range(self.num_layers):
            # --- Start of a single SNN Block ---
            if i == 0:
                rate_spikes = self.rate_coder(x)
            else:
                rate_spikes = self.rate_coder(x)

            if self.pos_encoder:
                pos_mask = self.pos_encoder(x)
            else:
                pos_mask = torch.ones(self.num_heads, self.T, device=x.device, dtype=x.dtype)

            if rate_spikes.dim() == 3:
                rate_spikes = rate_spikes.unsqueeze(-1).expand(-1,-1,-1, self.T)

            expanded_rate_spikes = rate_spikes.unsqueeze(2).expand(-1, -1, self.num_heads, -1, -1)

            if self.pos_encoder:
                masked_spikes = expanded_rate_spikes * pos_mask.unsqueeze(0).unsqueeze(1).unsqueeze(3)
            else:
                masked_spikes = expanded_rate_spikes

            spikes_over_time = []
            state = None
            
            # Let's sum over heads for simplicity to get [B, L, d, T]
            spikes_before_lif = masked_spikes.sum(2) # sum over heads
            
            # Now iterate over time
            for t in range(self.T):
                spikes_t, state = self.spiking_attention_layers[i](spikes_before_lif[..., t], state)
                spikes_over_time.append(spikes_t)
            
            y_heads = torch.stack(spikes_over_time, dim=-1) # [B, L, d, T]
            
            y_heads_summed_time = y_heads.sum(dim=-1) # [B, L, d]
            
            # The router expects [B, L, H, d]. We have [B, L, d].
            # Let's skip the router for now to simplify.
            merged = y_heads_summed_time

            x_residual = x + merged
            x_norm1 = self.ln1(x_residual)

            ffn_output = self.ffn(x_norm1)

            x = self.ln2(x_norm1 + ffn_output)

        return x

    def count_spikes(self):
        total_spikes = 0
        for layer in self.spiking_attention_layers:
            total_spikes += layer.spike_count
        return total_spikes

    def reset_spike_count(self):
        for layer in self.spiking_attention_layers:
            layer.reset_spike_count()

# Example Usage (Illustrative)
if __name__ == '__main__':
    B, L, d_model = 4, 20, 128
    H, T_window = 4, 10
    n_layers = 2

    input_embeddings = torch.rand(B, L, d_model)

    snn_dt_model = SNNDecisionTransformer(embed_dim=d_model, num_heads=H, window_length=T_window, num_layers=n_layers)

    output_representation = snn_dt_model(input_embeddings)

    print("Input shape:", input_embeddings.shape)
    print("Output shape:", output_representation.shape)
    print("Spike count:", snn_dt_model.count_spikes())