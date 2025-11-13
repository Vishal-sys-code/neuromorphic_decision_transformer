import torch
import torch.nn as nn

class FakeLIF(nn.Module):
    def __init__(self, v_th=1.0, decay=0.9):
        super().__init__()
        self.v_th = v_th
        self.decay = decay

    def forward(self, x, state=None):
        spikes = (x > self.v_th).float()
        v = self.decay * x
        return spikes, v