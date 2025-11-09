# snn-dt/scripts/run_minimal.py
import sys
from pathlib import Path
import torch
import yaml
import numpy as np

# Add project root and snn-dt directory to sys.path
snn_dt_root = Path(__file__).resolve().parent.parent
project_root = snn_dt_root.parent
sys.path.append(str(snn_dt_root))
sys.path.append(str(project_root))

from src.utils.config import AttrDict
from src.utils.models import get_model

def main():
    """
    Performs a one-batch forward pass to check the integrity of the model and data pipeline.
    """
    print("--- Starting Minimal Integrity Check ---")

    # --- Configuration ---
    config_path = project_root / "configs" / "dsformer_cartpole_debug.yaml"
    with open(config_path, "r") as f:
        cfg_raw = yaml.safe_load(f)

    cfg = {
        "model": {
            "name": "dsformer",
            "seq_len": cfg_raw.get("seq_len", 10),
            "d_model": cfg_raw.get("hidden_dim", 64),
            "n_heads": cfg_raw.get("n_heads", 4),
            "n_layers": cfg_raw.get("n_layers", 2),
            "action_tanh": False,
        },
        "dataset": {
            "state_dim": 4,  # CartPole-v1
            "act_dim": 2,    # CartPole-v1
            "max_timesteps": 500, # CartPole-v1
            "is_discrete": True,
        },
        "snn": {
            "lif_tau": cfg_raw.get("lif_tau", 10.0),
            "surrogate_k": cfg_raw.get("surrogate_k", 25.0),
        },
        "training": {
            "device": "cpu",
        }
    }
    cfg = AttrDict(cfg)

    print(f"--- Checkpoint: Configuration loaded from {config_path} ---")

    # --- Model Initialization ---
    model = get_model(cfg).to(cfg.training.device)
    print(f"--- Checkpoint: Model '{cfg.model.name}' initialized on device '{cfg.training.device}' ---")

    # --- Dummy Data ---
    batch_size = 32
    seq_len = cfg.model.seq_len
    state_dim = cfg.dataset.state_dim
    act_dim = cfg.dataset.act_dim

    batch = {
        "states": torch.randn(batch_size, seq_len, state_dim),
        "actions": torch.randint(0, act_dim, (batch_size, seq_len, 1)),
        "returns_to_go": torch.randn(batch_size, seq_len, 1),
        "timesteps": torch.randint(0, cfg.dataset.max_timesteps, (batch_size, seq_len, 1)),
        "mask": torch.ones(batch_size, seq_len),
    }
    print("--- Checkpoint: Dummy data batch created ---")

    # --- Forward Pass ---
    try:
        action_preds = model(batch)
        print("--- Checkpoint: Forward pass successful ---")
        print(f"Output shape: {action_preds.shape}")
        assert action_preds.shape == (batch_size, seq_len, act_dim)
        
        spikes = model.count_spikes()
        print(f"Spike count: {spikes}")
        assert spikes > 0, "Spike count should be > 0"

        print("--- Minimal Integrity Check PASSED ---")
    except Exception as e:
        print(f"--- Minimal Integrity Check FAILED: {e} ---")
        raise e

if __name__ == "__main__":
    main()