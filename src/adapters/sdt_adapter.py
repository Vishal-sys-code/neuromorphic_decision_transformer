# src/adapters/sdt_adapter.py
import torch
import time
from pathlib import Path
import importlib

# import your SDT model - adjust exactly to your repo structure
from src.models import SpikingDecisionTransformer 

class SDTWrapper:
    def __init__(self, config, device='cpu'):
        self.device = device
        self.model = SpikingDecisionTransformer(**config)
        self.model.to(device)
        self.model.eval()
        self.spike_count = 0
        self._register_spike_hooks()

    def _register_spike_hooks(self):
        for name, module in self.model.named_modules():
            cls_name = module.__class__.__name__.lower()
            if 'lif' in cls_name or 'spike' in cls_name:
                module.register_forward_hook(self._make_hook())

    def _make_hook(self):
        def hook(module, inp, out):
            try:
                out_tensor = out if isinstance(out, torch.Tensor) else out[0]
                self.spike_count += (out_tensor != 0).sum().item()
            except Exception:
                pass
        return hook

    def reset_spike_count(self):
        self.spike_count = 0

    def step(self, observation):
        self.model.eval()
        with torch.no_grad():
            t0 = time.perf_counter()
            obs_t = torch.tensor(observation).unsqueeze(0).float().to(self.device)
            action = self.model.act(obs_t)  # adapt to your SDT API
            t1 = time.perf_counter()
            latency_ms = (t1 - t0) * 1000.0
        return action.cpu().numpy(), latency_ms
