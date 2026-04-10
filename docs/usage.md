# Experimental Workflows

The framework partitions naturally into offline preprocessing, gradient-optimized spiking transformer training, and neuromorphic deployment environments.

## 1. Offline Dataset Instantiation

Before initializing SNN-DT surrogate algorithms, offline trajectories require sequencing structurally padded with corresponding reward-to-go scalars.

```python
from snn_dt.data import get_mixed_trajectory_loader

# Pre-compile the structured Return/State/Action inputs (50% Expert / 50% Random)
train_loader = get_mixed_trajectory_loader(
    env_name="Acrobot-v1",
    num_steps=10000, 
    seq_length=20
)
```

## 2. Model Initialization

```python
from snn_dt.models import SpikingDecisionTransformer

model = SpikingDecisionTransformer(
    env_dim=6, 
    action_dim=1, 
    d_model=128, 
    n_heads=4,
    n_layers=2, 
    lif_tau=20.0
)
```

## 3. Training Paradigm

During optimization, the localized Three-Factor eligibility updates happen simultaneously while Surrogate Gradients optimize dense representations.

```python
from snn_dt.trainer import train_offline_snn_dt

# Launches hybrid backpropagation with localized trace tracking.
train_offline_snn_dt(
    model, 
    train_loader, 
    epochs=50, 
    batch_size=64, 
    lr=3e-4, 
    eta_local=0.05
)
```