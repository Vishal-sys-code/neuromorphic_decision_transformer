import torch
from src.models.snn_lif import LIFNeuronLayer

def test_lif_spiking_on_step_input():
    layer = LIFNeuronLayer(1, 1)
    x = torch.ones(1, 1) * 5.0  # Constant input
    state = None
    spikes = []
    for _ in range(20):
        spk, state = layer(x, state)
        spikes.append(spk.item())
    assert sum(spikes) > 0, "LIF neuron failed to spike on step input"