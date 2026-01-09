
import sys
import torch
from pathlib import Path
import yaml

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))


try:
    from src.models.snn_dt import SnnDt
    import src.models.snn_dt as snn_dt_module
except ImportError:
    try:
        from models.snn_dt import SnnDt
        import models.snn_dt as snn_dt_module
    except ImportError as e:
        print(f"Import failed. Path: {sys.path}")
        raise e

def test_snn_dt():
    print(f"Imported SnnDt from: {snn_dt_module.__file__}")

    
    # Mock Config
    cfg = AttrDict({
        "model": AttrDict({
            "name": "snn_dt",
            "d_model": 128,
            "n_heads": 4,
            "n_layers": 2,
            "seq_len": 20
        }),
        "snn": AttrDict({
            "lif_tau": 5.0,
            "v_th": 0.5,
            "current_scale": 1.0,
            "surrogate_k": 10,
        }),
        "dataset": AttrDict({
             "state_dim": 4,
             "act_dim": 1,
             "max_timesteps": 100,
             "is_discrete": True
        })
    })

    model = SnnDt(cfg)
    print(f"Model created. Type: {type(model)}")
    print(f"Has last_logs? {hasattr(model, 'last_logs')}")
    
    # Mock Batch
    batch = {
        "states": torch.randn(2, 20, 4),
        "actions": torch.zeros(2, 20, 1),
        "returns_to_go": torch.randn(2, 20, 1),
        "timesteps": torch.arange(20).unsqueeze(0).repeat(2, 1)
    }
    
    print("Running forward pass...")
    output = model(batch)
    print("Forward pass complete.")
    
    print(f"Has last_logs? {hasattr(model, 'last_logs')}")
    if hasattr(model, 'last_logs'):
        print(f"Last Logs: {model.last_logs}")
    
    print(f"Count Spikes: {model.count_spikes()}")

if __name__ == "__main__":
    test_snn_dt()
