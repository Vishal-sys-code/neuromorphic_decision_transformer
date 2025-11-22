
import sys
import os
import torch
import numpy as np
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.path.abspath("snn-dt"))
sys.path.append(os.path.abspath("."))

from src.models.snn_dt import SnnDt
from src.utils.config import AttrDict

def check_dataset(dataset_path):
    print(f"Checking dataset at {dataset_path}...")
    if not os.path.exists(dataset_path):
        print("Dataset not found.")
        return

    data = np.load(dataset_path)
    for key in data.files:
        arr = data[key]
        if np.isnan(arr).any():
            print(f"WARNING: NaN found in {key}")
        if np.isinf(arr).any():
            print(f"WARNING: Inf found in {key}")
        print(f"{key}: shape={arr.shape}, min={arr.min()}, max={arr.max()}, mean={arr.mean()}")

def test_model_forward():
    print("\nTesting model forward pass...")
    cfg = AttrDict({
        "model": {
            "d_model": 128,
            "n_heads": 4,
            "n_layers": 2,
        },
        "dataset": {
            "state_dim": 4,
            "act_dim": 2,
            "max_timesteps": 1000,
            "is_discrete": True
        },
        "snn": {
            "lif_tau": 20.0,
            "surrogate_k": 25.0,
            "v_th": 0.05,
            "current_scale": 5.0,
            "use_plasticity": False
        }
    })

    model = SnnDt(cfg)
    model.eval()

    batch_size = 2
    seq_len = 10
    
    states = torch.randn(batch_size, seq_len, cfg.dataset.state_dim)
    actions = torch.randint(0, cfg.dataset.act_dim, (batch_size, seq_len, 1)).float()
    returns_to_go = torch.randn(batch_size, seq_len, 1)
    timesteps = torch.randint(0, cfg.dataset.max_timesteps, (batch_size, seq_len))
    
    batch = {
        "states": states,
        "actions": actions,
        "returns_to_go": returns_to_go,
        "timesteps": timesteps
    }

    print("Running forward pass...")
    try:
        output = model(batch)
        print("Output shape:", output.shape)
        print("Output has NaNs:", torch.isnan(output).any().item())
        print("Spike count:", model.count_spikes())
    except Exception as e:
        print("Forward pass failed:", e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    dataset_path = "data/CartPole-v1/dataset.npz"
    check_dataset(dataset_path)
    test_model_forward()