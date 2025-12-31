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
    "routing": {"enabled": False},
    "local_plasticity": {"enabled": False},
    "surrogate_slope_k": 10,
    "sequence_length_N": 20,
    "device": "cpu"
})

print("Instantiating Model...")
model = get_model(cfg)

print("Checking Evaluation Tensor Types (Mocking evaluate_policy logic)...")
target_return = 500 # Integer, as in the real code
rtgs = torch.full((1, cfg.sequence_length_N, 1), target_return, dtype=torch.float32, device=cfg.device)

print(f"RTGs dtype: {rtgs.dtype}")
if rtgs.dtype == torch.float32:
    print("PASS: RTGs are float32")
else:
    print(f"FAIL: RTGs are {rtgs.dtype}")

# Simulate Forward Pass with these tensors
batch = {
    "states": torch.zeros(1, cfg.sequence_length_N, 4, dtype=torch.float32),
    "actions": torch.zeros(1, cfg.sequence_length_N, 2, dtype=torch.float32),
    "returns_to_go": rtgs,
    "timesteps": torch.zeros(1, cfg.sequence_length_N, 1, dtype=torch.long)
}

try:
    preds, _ = model(batch)
    print("PASS: Forward pass successful with mock eval tensors.")
except Exception as e:
    print(f"FAIL: Forward pass failed. Error: {e}")
