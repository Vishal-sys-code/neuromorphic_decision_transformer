import torch
import torch.nn as nn

class VectorizedLIF(nn.Module):
    """
    A vectorized implementation of a Leaky Integrate-and-Fire (LIF) neuron layer.
    This module replaces the inefficient `norse.LIF` layer with a simple, transparent,
    and performant implementation that is optimized for GPU execution.

    The dynamics are modeled by the discrete-time recurrence:
    v[t] = v[t-1] * (1 - dt/tau) + x[t] * (dt/tau)
    s[t] = 1 if v[t] > v_th else 0
    v[t] = v[t] * (1 - s[t])  # Reset after spike
    """
    def __init__(self, tau, v_th, dt=0.001):
        super().__init__()
        self.register_buffer("tau", torch.tensor(tau))
        self.register_buffer("v_th", torch.tensor(v_th))
        self.dt = dt
        # Pre-calculate the decay/integration factor for efficiency
        self.alpha = self.dt / self.tau

    def forward(self, x, state=None):
        """
        Forward pass through the LIF layer.

        Args:
            x (torch.Tensor): Input tensor of shape (T, B, D), where T is sequence length,
                              B is batch size, and D is the feature dimension.
            state (torch.Tensor, optional): Initial membrane potential of shape (B, D).
                                            If None, it is initialized to zeros. Defaults to None.

        Returns:
            torch.Tensor: Output spike tensor of shape (T, B, D).
            torch.Tensor: Final membrane potential of shape (B, D).
        """
        seq_len, batch_size, num_features = x.shape

        # Pre-allocate tensors for performance
        spikes = torch.zeros_like(x)
        
        # Initialize membrane potential
        if state is None:
            v_t = torch.zeros(batch_size, num_features, device=x.device)
        else:
            v_t = state

        # Loop over the time dimension
        for t in range(seq_len):
            # Leaky integration
            v_t = v_t * (1 - self.alpha) + x[t] * self.alpha
            
            # Spike generation
            spikes_t = (v_t > self.v_th).float()
            
            # Reset membrane potential after spike
            v_t = v_t * (1 - spikes_t)
            
            spikes[t] = spikes_t
        
        return spikes, v_t