import sys
import torch
import torch.nn as nn
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'snn-dt', 'src'))

from models.iql import IQL

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        for k, v in self.items():
            if isinstance(v, dict) and not isinstance(v, AttrDict):
                self[k] = AttrDict(v)
        self.__dict__ = self
    def __getattr__(self, name):
        if name in self.__dict__: return self.__dict__[name]
        if name in self: return self[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

cfg = AttrDict()
cfg.device = 'cpu'
cfg.lr = 3e-4
cfg.env = 'CartPole-v1'

cfg.iql = AttrDict()
cfg.iql.tau = 0.005
cfg.iql.hidden_size = 256
cfg.iql.temperature = 3.0
cfg.iql.expectile = 0.7

cfg.dataset = AttrDict()
cfg.dataset.act_dim = 2
cfg.dataset.state_dim = 4
cfg.dataset.state_dim = 4
# cfg.dataset.is_discrete = True # Removed to verify the fix works without this setting

try:
    model = IQL(cfg)
    print("IQL Instantiated Successfully")
except Exception as e:
    print(f"IQL Instantiation Failed: {e}")
    sys.exit(1)
