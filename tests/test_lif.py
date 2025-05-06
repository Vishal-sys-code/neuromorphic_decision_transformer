import torch
from src.models.snn_lif import LIFNeuronLayer

def test_lif_spiking_on_step_input():
    layer = LIFNeuronLayer(1, 1)
    # Force linear weights to 1 and bias to 0 for a strong, deterministic current
    with torch.no_grad():
        layer.fc.weight.fill_(1.0)
        layer.fc.bias.fill_(0.0)
    x = torch.ones(1, 1) * 5.0  # Constant input well above threshold
    state = None
    spikes = []
    for _ in range(20):
        spk, state = layer(x, state)
        spikes.append(spk.item())
    assert sum(spikes) > 0, "LIF neuron failed to spike on step input"