import sys
import torch
import torch.nn as nn
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'snn-dt', 'src'))

from models.cql import CQL

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

cfg.cql = AttrDict()
cfg.cql.tau = 0.005
cfg.cql.hidden_size = 256
cfg.cql.with_lagrange = False
cfg.cql.temperature = 1.0
cfg.cql.cql_weight = 1.0
cfg.cql.target_action_gap = 10.0

cfg.dataset = AttrDict()
cfg.dataset.act_dim = 2
cfg.dataset.state_dim = 4

# Mock training subsection to verify it's NOT used (if used, it would be fine if I added it, but I want to ensure it works WITHOUT it if my fix worked)
# Actually, run_experiment.py DOES NOT have cfg.training at call time. So I should NOT add it.

try:
    model = CQL(cfg)
    print("CQL Instantiated Successfully")
except Exception as e:
    print(f"CQL Instantiation Failed: {e}")
    sys.exit(1)
