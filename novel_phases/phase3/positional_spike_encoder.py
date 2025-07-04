import torch
import torch.nn as nn
import math

class PositionalSpikeEncoder(nn.Module):
    def __init__(self, num_heads: int, time_window: int):
        super().__init__()
        self.num_heads = num_heads
        self.time_window = time_window

        # Learnable parameters for frequency (ωₖ) and phase (ϕₖ) per head
        self.pos_freq = nn.Parameter(torch.rand(num_heads) * 2 * math.pi)
        self.pos_phase = nn.Parameter(torch.rand(num_heads) * 2 * math.pi)

        # Pre-calculate t_range for efficiency if it doesn't change based on input device
        # However, to ensure it's on the correct device, it's better to create it in forward
        # or ensure the module is moved to the correct device and then create it.
        # For simplicity and device safety, creating in forward is often safer unless performance critical.

    def forward(self, qh: torch.Tensor, kh: torch.Tensor, vh: torch.Tensor, t: int):
        """
        Applies a learned positional spike mask to the input head projections (qh, kh, vh)
        for a specific timestep t.

        Args:
            qh (torch.Tensor): Query head projections, shape [B, H, S, head_dim].
            kh (torch.Tensor): Key head projections, shape [B, H, S, head_dim].
            vh (torch.Tensor): Value head projections, shape [B, H, S, head_dim].
            t (int): The current timestep in the window [0, time_window - 1].

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Masked qh, kh, vh.
        """
        if self.time_window <= 0: # Should not happen with valid time_window
            return qh, kh, vh

        # Ensure t_range is on the same device as parameters and of correct dtype
        # Create a single value for the current timestep 't'
        current_t_val = torch.tensor([t], device=self.pos_freq.device, dtype=self.pos_freq.dtype)

        # Wave calculation for the current timestep t for each head:
        # wave_t_h = sin(ωₖ * t + ϕₖ)
        # self.pos_freq is [H], current_t_val is [1], self.pos_phase is [H]
        # Resulting wave_at_t will be [H]
        wave_at_t = torch.sin(self.pos_freq * current_t_val + self.pos_phase)

        # Binary mask for the current timestep t for each head: (spike if wave > 0)
        # pos_domain_mask_t_h will be [H]
        pos_domain_mask_t_h = (wave_at_t > 0).float()

        # Reshape mask to [1, H, 1, 1] for broadcasting over B, S, head_dim
        # Original qh, kh, vh are [B, H, S, head_dim]
        current_pos_mask_broadcastable = pos_domain_mask_t_h.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        # This makes it [H, 1, 1], then unsqueeze(0) makes it [1, H, 1, 1]

        masked_qh = qh * current_pos_mask_broadcastable
        masked_kh = kh * current_pos_mask_broadcastable
        masked_vh = vh * current_pos_mask_broadcastable

        return masked_qh, masked_kh, masked_vh

if __name__ == '__main__':
    # Example Usage
    B, H, S, D_head = 2, 4, 10, 16  # Batch, Heads, SeqLen, HeadDim
    T_window = 20 # Time window

    encoder = PositionalSpikeEncoder(num_heads=H, time_window=T_window)

    # Dummy inputs for one timestep
    qh_t = torch.randn(B, H, S, D_head)
    kh_t = torch.randn(B, H, S, D_head)
    vh_t = torch.randn(B, H, S, D_head)
    
    current_timestep = 5 # Example timestep

    masked_qh_t, masked_kh_t, masked_vh_t = encoder(qh_t, kh_t, vh_t, current_timestep)

    print("Original qh_t shape:", qh_t.shape)
    print("Masked qh_t shape:", masked_qh_t.shape)
    
    # Check if masking happened (some values might be zero)
    # print("Original vh_t sample (head 0, item 0, token 0):", vh_t[0,0,0,:4])
    # print("Masked vh_t sample (head 0, item 0, token 0):", masked_vh_t[0,0,0,:4])

    # Verify learned parameters
    print("\nLearned Frequencies (should be random):", encoder.pos_freq)
    print("Learned Phases (should be random):", encoder.pos_phase)

    # Check mask for a specific head over time
    # This requires calling the forward multiple times or exposing the mask generation
    print(f"\nSimulating mask for head 0 over time window {T_window}:")
    for t_step in range(T_window):
        _, _, vh_example = encoder(torch.ones(1,H,1,1), torch.ones(1,H,1,1), torch.ones(1,H,1,1), t_step)
        # Check if the first element of head 0 is masked (0) or not (1)
        print(f"t={t_step:02d}: mask_effect_on_head0 = {vh_example[0,0,0,0].item()}")

    # For a production model, ensure parameters are moved to the correct device, e.g.
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # encoder.to(device)
    # qh_t = qh_t.to(device) ... etc.
    # This example assumes CPU.
    # The PositionalSpikeEncoder will use the device of its parameters for calculations.
    # Input tensors qh, kh, vh should be on the same device.
    # The current_t_val is explicitly created on self.pos_freq.device.
    pass