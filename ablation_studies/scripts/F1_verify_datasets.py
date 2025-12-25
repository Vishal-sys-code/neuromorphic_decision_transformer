import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import sys

# --- Add Project Root to sys.path ---
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# --- Local Imports ---
from ablation_studies.src.models.ablation_dsformer import AblationDsFormer
from ablation_studies.src.datasets import OfflineSequenceDataset

# --- Configuration ---
PROCESSED_DIR = Path(__file__).parent.parent / "datasets/processed"
PLOTS_DIR = Path(__file__).parent.parent / "datasets/verification_plots"
ENVS = ["CartPole-v1", "Acrobot-v1", "Pendulum-v1"]

# --- Plotting Style ---
sns.set_theme(style="whitegrid", context="paper")

# --- Verification Functions ---
def plot_distributions(env_name):
    """Plots the return and RTG distributions for both datasets."""
    plt.figure(figsize=(12, 5))
    
    # Stratified Dataset
    strat_data = np.load(PROCESSED_DIR / env_name / "stratified_dataset.npz")
    plt.subplot(1, 2, 1)
    sns.histplot(strat_data['returns_to_go'].flatten(), bins=50, kde=True)
    plt.title(f"{env_name} - Stratified Dataset RTG Distribution")
    
    # Random-Heavy Dataset
    random_data = np.load(PROCESSED_DIR / env_name / "random_heavy_dataset.npz")
    plt.subplot(1, 2, 2)
    sns.histplot(random_data['returns_to_go'].flatten(), bins=50, kde=True)
    plt.title(f"{env_name} - Random-Heavy Dataset RTG Distribution")
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{env_name}_distributions.png")
    plt.close()
    print(f"Saved distribution plots for {env_name}.")

def spike_sanity_check(env_name):
    """Performs a spike sanity check on both datasets."""
    # Dummy config for the model
    class DummyCfg:
        class model:
            name = 'ablation_dsformer'
            hidden_dim_d = 128
            num_heads_H = 4
            num_layers_L = 2
            surrogate_slope_k = 10
            class routing: enabled = True
            class phase_encoder: enabled = True
            class local_plasticity: enabled = False
        class dataset:
            max_timesteps = 1000
        device = 'cpu'

    cfg = DummyCfg()
    
    # Load model
    model = AblationDsFormer(cfg)
    
    # Run forward pass on a batch from each dataset
    strat_dataset = OfflineSequenceDataset(str(PROCESSED_DIR / env_name / "stratified_dataset.npz"), 20)
    random_dataset = OfflineSequenceDataset(str(PROCESSED_DIR / env_name / "random_heavy_dataset.npz"), 20)
    
    strat_batch = next(iter(torch.utils.data.DataLoader(strat_dataset, batch_size=32)))
    random_batch = next(iter(torch.utils.data.DataLoader(random_dataset, batch_size=32)))
    
    model.reset_spike_counts()
    model(strat_batch)
    strat_spikes = model.count_spikes()
    
    model.reset_spike_counts()
    model(random_batch)
    random_spikes = model.count_spikes()
    
    print(f"Spike Sanity Check for {env_name}:")
    print(f"  - Stratified Dataset: {strat_spikes:.2f} spikes")
    print(f"  - Random-Heavy Dataset: {random_spikes:.2f} spikes")
    print("  - Expected: Stratified < Random-Heavy")

# --- Main Script ---
def main():
    PLOTS_DIR.mkdir(exist_ok=True)
    for env_name in ENVS:
        plot_distributions(env_name)
        spike_sanity_check(env_name)

if __name__ == "__main__":
    main()