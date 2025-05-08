"""
Author: Vishal Pandey
Email: pandeyvishal.mlprof@gmail.com
"""
ENVIRONMENTS = [
    "CartPole-v1",
    "MountainCar-v0",
    "LunarLander-v2",
]

SEED = 42
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"

# Baseline training hyperparameters
epochs = 50
steps_per_epoch = 1000
lr = 1e-4

# Decision Transformer settings (will override state_dim & act_dim per env)
dt_config = {
    "state_dim": 4,
    "act_dim": 2,
    "hidden_size": 128,
    "max_length": 100,
    "n_layer": 2,
    "n_head": 1,
    "n_inner": 256,
}

# === SNN-DT Training Hyperparameters ===
time_window    = 5       # how many timesteps for your spiking attention
max_length     = 20      # context window length for DT (<= your embedder max_length)
steps_per_epoch= 5000    # env steps per epoch
epochs         = 20      # number of epochs
gamma          = 0.99    # discount for return-to-go
batch_size     = 1       # online, so batch of 1