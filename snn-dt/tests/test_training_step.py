import torch
import pytest
from src.models.snn_dt import SnnDt

class MockConfig:
    class ModelConfig:
        d_model = 64
        n_heads = 4
        n_layers = 1
        seq_len = 5
    class DatasetConfig:
        state_dim = 10
        act_dim = 2
        max_timesteps = 20
        is_discrete = False
    class SnnConfig:
        lif_tau = 10.0
        surrogate_k = 1.0
        v_th = 0.5
        current_scale = 1.0
        use_plasticity = True
        eta_local = 0.01
        lambda_decay = 0.95

    model = ModelConfig()
    dataset = DatasetConfig()
    snn = SnnConfig()

def test_training_step_loss_decrease():
    torch.manual_seed(42)
    cfg = MockConfig()
    model = SnnDt(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()
    
    batch_size = 4
    seq_len = 5
    
    # Fixed target
    target_actions = torch.randn(batch_size, seq_len, cfg.dataset.act_dim)
    
    losses = []
    
    for _ in range(5):
        optimizer.zero_grad()
        model.reset_spike_counts()
        
        batch = {
            "states": torch.randn(batch_size, seq_len, cfg.dataset.state_dim),
            "actions": torch.randn(batch_size, seq_len, cfg.dataset.act_dim),
            "returns_to_go": torch.randn(batch_size, seq_len, 1),
            "timesteps": torch.randint(0, cfg.dataset.max_timesteps, (batch_size, seq_len))
        }
        
        preds = model(batch)
        loss = criterion(preds, target_actions)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        # Apply plasticity (simulated reward)
        model.apply_plasticity(reward=-loss.item()) # Negative loss as reward? Or just descent.
        
    # Check if loss generally decreases
    assert losses[-1] < losses[0]