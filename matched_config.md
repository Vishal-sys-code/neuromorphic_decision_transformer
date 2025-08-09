# Matched Hyperparameters for SNN-DT and DSF-DT Comparison

## Shared Hyperparameters

```python
ENV_NAME = "CartPole-v1"
MAX_LENGTH = 20  # sequence length
BATCH_SIZE = 64
GAMMA = 0.99  # discount factor
LEARNING_RATE = 1e-4
OFFLINE_STEPS = 5000
TRAINING_EPOCHS = 10
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

## Model-Specific Hyperparameters

### DSF-DT Configuration

```python
dsf_config = {
    "state_dim": 4,  # CartPole-v1 state dimension
    "act_dim": 2,    # CartPole-v1 action dimension
    "hidden_size": 128,
    "max_length": MAX_LENGTH,
    "n_layer": 2,
    "n_head": 1,
    "T": 4,          # Spiking time steps
    "norm_type": 1,  # Layer normalization
    "warmup_ratio": 0.1,
    "window_size": 8,
    "num_training_steps": TRAINING_EPOCHS * (OFFLINE_STEPS // BATCH_SIZE)
}
```

### SNN-DT Configuration

```python
snn_config = {
    "state_dim": 4,  # CartPole-v1 state dimension
    "act_dim": 2,    # CartPole-v1 action dimension
    "hidden_size": 128,
    "max_length": MAX_LENGTH,
    "n_layer": 2,
    "n_head": 1,
    "time_window": 5,  # Spiking time steps
    "n_inner": 256
}
```

## Hyperparameter Matching Notes

1. **Hidden Size**: Both models use 128 dimensions
2. **Layers**: Both models use 2 transformer layers
3. **Heads**: Both models use 1 attention head
4. **Sequence Length**: Both models use 20 context length
5. **Time Steps**: SNN-DT uses 5, DSF-DT uses 4 (closest match)
6. **Batch Size**: Both use 64
7. **Training Epochs**: Both use 10 epochs
8. **Offline Steps**: Both use 5000 steps for dataset collection

## Training Configuration

```python
training_config = {
    "env_name": ENV_NAME,
    "max_length": MAX_LENGTH,
    "batch_size": BATCH_SIZE,
    "gamma": GAMMA,
    "lr": LEARNING_RATE,
    "offline_steps": OFFLINE_STEPS,
    "dt_epochs": TRAINING_EPOCHS,
    "seed": SEED,
    "device": DEVICE
}
```

## Evaluation Configuration

```python
evaluation_config = {
    "num_eval_episodes": 10,
    "env_name": ENV_NAME,
    "seed": SEED
}