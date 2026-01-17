import os
import yaml
from pathlib import Path

# Constants
MODELS = ["dt", "dsformer", "snn_dt", "iql", "cql"]
ENVS = [
    "hopper-medium-v2",
    "hopper-medium-expert-v2",
    "walker2d-medium-v2",
    "walker2d-medium-expert-v2",
    "halfcheetah-medium-v2",
    "halfcheetah-medium-expert-v2",
]
EPOCHS = 50
BATCH_SIZE = 64
SEQ_LEN = 20
HIDDEN_DIM = 256 # Standard for D4RL
N_HEADS = 4
N_LAYERS = 3

# Base Config Structure
BASE_CONFIG = {
    "training": {
        "epochs": EPOCHS,
        "batches_per_epoch": 1000,
        "batch_size": BATCH_SIZE,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "eval_every": 5,
        "checkpoint_every": 50, # Save at end
        "num_workers": 4,
        "device": "cuda",
        "simulator_available": False, # D4RL usually doesn't need sim during train, but eval needs it. 
                                     # Train script handles eval if sim is present.
    },
    "model": {
        "seq_len": SEQ_LEN,
        "d_model": HIDDEN_DIM,
        "n_heads": N_HEADS,
        "n_layers": N_LAYERS,
    },
    "hyperparameters": {
        "eval_episodes": 10
    },
    "snn": {
        "lif_tau": 20.0,
        "v_th": 1.0,
        "surrogate_k": 25.0,
        "current_scale": 0.2, # As per paper/previous configs
        "use_plasticity": False
    },
    "iql": {
        "tau": 0.005,
        "temperature": 3.0,
        "expectile": 0.7,
        "hidden_size": HIDDEN_DIM
    },
    "cql": {
        "tau": 0.005,
        "temperature": 1.0,
        "hidden_size": HIDDEN_DIM,
        "with_lagrange": False,
        "cql_weight": 1.0,
        "target_action_gap": 10.0
    }
}

def generate_configs():
    out_dir = Path("configs/phase0")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for model in MODELS:
        for env in ENVS:
            config = BASE_CONFIG.copy()
            # Deep copy to safely modify
            import copy
            config = copy.deepcopy(BASE_CONFIG)
            
            # Specific Overrides
            config["model"]["name"] = model
            config["env"] = env
            
            # SNN-DT specific tweaks?
            if model == "snn_dt":
                # Maybe scale down for efficiency? Keeping standard for now.
                pass
            
            # DSFormer specific tweaks?
            if model == "dsformer":
                # Ensure patch size / embedding fits
                pass
            
            # Save
            filename = f"{model}_{env.replace('-', '_')}.yaml"
            with open(out_dir / filename, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
                
    print(f"Generated {len(MODELS) * len(ENVS)} configs in {out_dir}")

if __name__ == "__main__":
    generate_configs()
