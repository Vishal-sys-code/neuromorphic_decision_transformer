import torch
import torch.nn as nn
import norse.torch as norse

class LIFNeuronLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.lif = norse.LIFCell()

    def forward(self, x, state=None):
        z = self.fc(x)
        spiked, state = self.lif(z, state)
        return spiked, state