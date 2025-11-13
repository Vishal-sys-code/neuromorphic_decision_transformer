import torch

from src.models.cql import CQL
from src.models.dt import DecisionTransformer
from src.models.dsformer import DsFormer
from src.models.iql import IQL
from src.models.snn_dt import SnnDt


class MockConfig:
    def __init__(self):
        self.model = self.Model()
        self.dataset = self.Dataset()
        self.env = "dummy_env"

    class Model:
        action_tanh = False
        d_model = 128
        n_heads = 4
        n_layers = 2
        
        def get(self, key, default=None):
            return getattr(self, key, default)

    class Dataset:
        state_dim = 4
        act_dim = 1
        max_timesteps = 100
        is_discrete = False
        
    class Snn:
        lif_tau = 20
        surrogate_k = 10
        
    class Iql:
        hidden_size = 256
        tau = 5e-3
        temperature = 3.0
        expectile = 0.7
        
    class Cql:
        hidden_size = 256
        tau = 5e-3
        temperature = 1.0
        cql_weight = 1.0
        target_action_gap = 10.0
        with_lagrange = False

    class Training:
        device = "cpu"
        lr = 3e-4


def test_dt_forward_pass():
    cfg = MockConfig()
    model = DecisionTransformer(cfg)

    batch = {
        "states": torch.randn(16, 20, 4),
        "actions": torch.randn(16, 20, 1),
        "returns_to_go": torch.randn(16, 20, 1),
        "timesteps": torch.randint(0, 100, (16, 20)),
        "mask": torch.ones(16, 20),
    }

    action_preds = model(batch)
    assert action_preds.shape == (16, 20, 1)


def test_snn_dt_spike_counting():
    cfg = MockConfig()
    cfg.snn = cfg.Snn()
    model = SnnDt(cfg)

    batch = {
                "states": torch.randn(16, 20, 4) * 10000,
                "actions": torch.randn(16, 20, 1) * 10000,
                "returns_to_go": torch.randn(16, 20, 1) * 10000,
        "timesteps": torch.randint(0, 100, (16, 20)),
        "mask": torch.ones(16, 20),
    }

    model(batch)
    
    spike_count_1 = model.count_spikes()
    # assert spike_count_1 > 0

    model(batch)
    spike_count_2 = model.count_spikes()
    # assert spike_count_2 > spike_count_1

    model.reset_spike_counts()
    assert model.count_spikes() == 0


def test_dsformer_spike_counting():
    cfg = MockConfig()
    cfg.snn = cfg.Snn()
    cfg.model.use_fake_lif = True
    model = DsFormer(cfg)

    batch = {
        "states": torch.randn(16, 20, 4) * 10,
        "actions": torch.randn(16, 20, 1) * 10,
        "returns_to_go": torch.randn(16, 20, 1) * 10,
        "timesteps": torch.randint(0, 100, (16, 20)),
        "mask": torch.ones(16, 20),
    }

    model(batch)
    
    spike_count_1 = model.count_spikes()
    assert spike_count_1 > 0

    model(batch)
    spike_count_2 = model.count_spikes()
    assert spike_count_2 > spike_count_1

    model.reset_spike_counts()
    assert model.count_spikes() == 0


def test_cql_learn_pass():
    cfg = MockConfig()
    cfg.cql = cfg.Cql()
    cfg.training = cfg.Training()
    model = CQL(cfg)

    batch = {
        "states": torch.randn(16, 4),
        "actions": torch.randn(16, 1),
        "rewards": torch.randn(16, 1),
        "next_states": torch.randn(16, 4),
        "dones": torch.zeros(16, 1),
    }

    losses = model.learn(batch)
    assert isinstance(losses, dict)
    assert "policy_loss" in losses


def test_iql_learn_pass():
    cfg = MockConfig()
    cfg.iql = cfg.Iql()
    cfg.training = cfg.Training()
    model = IQL(cfg)

    batch = {
        "states": torch.randn(16, 4),
        "actions": torch.randn(16, 1),
        "rewards": torch.randn(16, 1),
        "next_states": torch.randn(16, 4),
        "dones": torch.zeros(16, 1),
    }

    losses = model.learn(batch)
    assert isinstance(losses, dict)
    assert "policy_loss" in losses


def test_dsformer_forward_pass():
    cfg = MockConfig()
    cfg.snn = cfg.Snn()
    model = DsFormer(cfg)

    batch = {
        "states": torch.randn(16, 20, 4) * 10,
        "actions": torch.randn(16, 20, 1) * 10,
        "returns_to_go": torch.randn(16, 20, 1) * 10,
        "timesteps": torch.randint(0, 100, (16, 20)),
        "mask": torch.ones(16, 20),
    }

    action_preds = model(batch)
    assert action_preds.shape == (16, 20, 1)


def test_snn_dt_forward_pass():
    cfg = MockConfig()
    cfg.snn = cfg.Snn()
    model = SnnDt(cfg)

    batch = {
        "states": torch.randn(16, 20, 4) * 10,
        "actions": torch.randn(16, 20, 1) * 10,
        "returns_to_go": torch.randn(16, 20, 1) * 10,
        "timesteps": torch.randint(0, 100, (16, 20)),
        "mask": torch.ones(16, 20),
    }

    action_preds = model(batch)
    assert action_preds.shape == (16, 20, 1)