import torch

# General Hyperparameters
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Environment Specific
ENV_NAME = "MountainCar-v0"

# SNN-DT Training Hyperparameters
time_window    = 5
max_length     = 20
steps_per_epoch= 5000
gamma          = 0.99
batch_size     = 1

# Offline data collection
offline_steps = 5000

# Offline DT training
dt_batch_size   = 64
dt_epochs    = 100
lr = 1e-4

# Decision Transformer model config
dt_config = {
    "hidden_size": 128,
    "n_layer": 2,
    "n_head": 1,
    "n_inner": 256,
    "activation_function": "relu",
    "n_positions": 1024,
    "resid_pdrop": 0.1,
    "attn_pdrop": 0.1,
}
