import torch
import torch.nn as nn
from src.models.snn_lif import LIFNeuronLayer

class SpikingMLP(nn.Module):
    def __init__(self, input_dim=28*28, hidden_dim=256, output_dim=10, time_window=10):
        super().__init__()
        self.time_window = time_window
        self.layer1 = LIFNeuronLayer(input_dim, hidden_dim)
        self.layer2 = LIFNeuronLayer(hidden_dim, output_dim)

    def forward(self, x):
        batch_size = x.size(0)
        outputs = torch.zeros(batch_size, 10, device=x.device)
        s1 = s2 = None

        for _ in range(self.time_window):
            out1, s1 = self.layer1(x, s1)
            out2, s2 = self.layer2(out1, s2)
            outputs += out2  # Spike accumulation

        return outputs / self.time_window  # Avg firing rate