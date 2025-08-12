"""
Author: Vishal Pandey
Email: pandeyvishal.mlprof@gmail.com

A Leaky Integrate-and-Fire (LIF) neuron layer with spike counting.
Uses Norse for the LIFCell dynamics.
"""

import torch
import torch.nn as nn
import norse.torch as norse


class LIFNeuronLayer(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        # Linear projection from inputs to membrane currents
        self.fc = nn.Linear(input_size, output_size)
        # The Norse LIFCell handles membrane potential integration and spiking
        self.lif = norse.LIFCell()
        # Buffer to accumulate the total number of spikes emitted during each forward
        self.register_buffer("_spike_count", torch.zeros(1, dtype=torch.long))

    def forward(self, x: torch.Tensor, state: torch.Tensor = None):
        """
        Args:
            x:      Input tensor of shape [..., input_size].
            state:  Optional previous LIFCell state.

        Returns:
            spiked: Binary spike tensor of same shape as the cell output;
            state:  Updated LIFCell state.
        """
        # Project inputs to membrane currents
        z = self.fc(x)
        # Compute spikes and next state
        spiked, new_state = self.lif(z, state)

        # Count total spikes (convert to long then sum over all dims)
        num_spikes = spiked.detach().to(torch.long).sum()
        self._spike_count += num_spikes

        return spiked, new_state

    @property
    def spike_count(self) -> int:
        """
        Returns:
            The total number of spikes emitted during the last forward call.
        """
        return int(self._spike_count.item())

    def reset_spike_count(self):
        """
        Resets the spike counter to zero.
        """
        self._spike_count.zero_()