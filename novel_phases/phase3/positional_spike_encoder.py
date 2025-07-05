import torch
import torch.nn as nn

class PositionalSpikeEncoder(nn.Module):
    """
    Given a batch of token embeddings (B*L*d) and desired window T,
    generates a binary spike tensor (B*L*H*d*T) where H is # heads.
    """

    def __init__(self, num_heads: int, window_length: int):
        super().__init__()
        self.H = num_heads
        self.T = window_length
        # learnable freq & phase per head
        self.freq  = nn.Parameter(torch.rand(self.H) * 2 * torch.pi)
        self.phase = nn.Parameter(torch.rand(self.H) * 2 * torch.pi)

    def forward(self, embeddings: torch.Tensor):
        # embeddings: [B, L, d]
        B, L, d = embeddings.shape
        device = embeddings.device

        # 1) Rate-code embeddings: (you already have this)
        #    rate_spikes: [B, L, d, T]

        # 2) Build positional masks: [H, T]
        t = torch.arange(self.T, device=device).float()  # [T]
        wave = torch.sin(self.freq.unsqueeze(1) * t + self.phase.unsqueeze(1))  # [H, T]
        pos_mask = (wave > 0).float()  # [H, T]

        # 3) Expand to [B, L, H, d, T]
        #    assume rate_spikes is computed earlier, you’ll multiply by pos_mask
        return pos_mask  # user will integrate with their rate_spikes