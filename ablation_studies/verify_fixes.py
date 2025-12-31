import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from ablation_studies.run_experiment import get_model, AttrDict
import numpy as np

# Mock Config
cfg = AttrDict({
    "model": {"name": "ablation_dsformer"},
    "hidden_dim_d": 128,
    "num_layers_L": 2,
    "num_heads_H": 4,
    "dataset": {"state_dim": 4, "act_dim": 2, "max_timesteps": 500},
    "phase_encoder": {"enabled": False},
    "routing": {"enabled": False}, # check no_routing path
    "local_plasticity": {"enabled": False},
    "surrogate_slope_k": 10
})

print("Instantiating Model...")
model = get_model(cfg)
print(f"Model instantiated: {type(model).__name__}")

# Check 1: predict_action existence
has_predict = hasattr(model, 'predict_action')
print(f"hasattr(model, 'predict_action'): {has_predict}")
if has_predict:
    print("FAIL: Model should NOT have predict_action")
else:
    print("PASS: Model correctly missing predict_action (will use sequence generation loop)")

# Check 2: Forward pass (Fixing RuntimeError: view size ...)
print("Running Forward Pass...")
batch_size = 2
seq_len = 20
batch = {
    "states": torch.randn(batch_size, seq_len, 4),
    "actions": torch.randn(batch_size, seq_len, 2),
    "returns_to_go": torch.randn(batch_size, seq_len, 1),
    "timesteps": torch.zeros(batch_size, seq_len, dtype=torch.long)
}

try:
    preds, _ = model(batch)
    print(f"PASS: Forward pass successful. Output shape: {preds.shape}")
except Exception as e:
    print(f"FAIL: Forward pass failed. Error: {e}")

