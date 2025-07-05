import torch
import torch.nn as nn

class DendriticRouter(nn.Module):
    """
    Given multi-head spike outputs Y: [B, L, H, d, T] or [B, L, H, d] (after summing T),
    computes gates per head and re-weights.
    """

    def __init__(self, num_heads: int):
        super().__init__()
        self.H = num_heads
        # tiny MLP: maps from H → H gates
        self.routing_mlp = nn.Sequential(
            nn.Linear(self.H, self.H),
            nn.Sigmoid()
        )

    def forward(self, y_heads: torch.Tensor):
        # y_heads: [B, L, H, d]  (assume summed over T already)
        B, L, H, d = y_heads.shape

        # 1) Summarize per head: sum over features
        summary = y_heads.sum(dim=-1)  # [B, L, H]

        # 2) Compute gates: flatten B×L into one batch for MLP
        gates = self.routing_mlp(summary.view(-1, H))  # [(B×L), H]
        gates = gates.view(B, L, H)  # [B, L, H]

        # 3) Apply gates
        gated = (gates.unsqueeze(-1) * y_heads).sum(dim=2)  # [B, L, d]
        return gated
