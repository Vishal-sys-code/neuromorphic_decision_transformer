import torch
import torch.nn as nn

class DendriticRouter(nn.Module):
    def __init__(self, num_heads: int):
        super().__init__()
        self.num_heads = num_heads

        # MLP for gating head outputs
        # Input: [B, H], Output: [B, H] (sigmoid activated)
        # Using num_heads * 2 as a simple hidden layer dimension for expressiveness
        hidden_dim_routing = self.num_heads * 2
        
        self.routing_mlp = nn.Sequential(
            nn.Linear(self.num_heads, hidden_dim_routing),
            nn.ReLU(),
            nn.Linear(hidden_dim_routing, self.num_heads),
            nn.Sigmoid()
        )

    def forward(self, out_h: torch.Tensor):
        """
        Applies learned dendritic-style routing to the attention head outputs.

        Args:
            out_h (torch.Tensor): Output of attention heads,
                                  shape [B, H, S, head_dim] (Batch, Heads, SeqLen, HeadDim).

        Returns:
            torch.Tensor: Gated head outputs, same shape as out_h.
        """
        B, H, S, D_head = out_h.shape

        # Summarize head activity: Sum over sequence length (S) and head dimension (D_head)
        # head_summary: [B, H]
        head_summary = out_h.sum(dim=[2, 3])
        
        # Get gates from MLP: gates [B, H]
        gates = self.routing_mlp(head_summary)

        # Apply gating to each head's output vector
        # gates need to be reshaped to [B, H, 1, 1] for broadcasting
        gated_out_h = gates.unsqueeze(-1).unsqueeze(-1) * out_h
        
        return gated_out_h

if __name__ == '__main__':
    # Example Usage
    B, H, S, D_head = 2, 4, 10, 16  # Batch, Heads, SeqLen, HeadDim

    router = DendriticRouter(num_heads=H)

    # Dummy input for head outputs
    dummy_out_h = torch.randn(B, H, S, D_head)
    
    gated_output = router(dummy_out_h)

    print("Original out_h shape:", dummy_out_h.shape)
    print("Gated out_h shape:", gated_output.shape)

    # Check if gating happened (some values might be scaled)
    # print("\nOriginal out_h sample (batch 0, head 0, token 0, first 4 features):", dummy_out_h[0,0,0,:4])
    # print("Gated out_h sample (batch 0, head 0, token 0, first 4 features):", gated_output[0,0,0,:4])
    
    # Inspect MLP weights (example for the first layer)
    # print("\nRouting MLP first layer weights shape:", router.routing_mlp[0].weight.shape)
    # print("Routing MLP first layer weights (sample):", router.routing_mlp[0].weight)
    
    # For a production model, ensure parameters are moved to the correct device, e.g.
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # router.to(device)
    # dummy_out_h = dummy_out_h.to(device)
    # This example assumes CPU.
    # The DendriticRouter will use the device of its parameters for calculations.
    # Input tensor out_h should be on the same device.
    pass