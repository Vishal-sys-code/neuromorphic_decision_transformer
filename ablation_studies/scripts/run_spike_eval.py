
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import json
import random

# --- Add Project Root to sys.path ---
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from ablation_studies.src.models.ablation_dsformer import AblationDsFormer
from ablation_studies.src.datasets import OfflineSequenceDataset

# --- Configuration ---
FROZEN_DIR = Path(__file__).parent.parent / "datasets/frozen_v1"
OUTPUT_DIR = Path(__file__).parent.parent / "spike_eval"
ENVS = ["CartPole-v1", "Acrobot-v1", "Pendulum-v1", "MountainCar-v0"]
NUM_RUNS = 50
BATCH_SIZE = 1 # We want per-inference precise stats, but batching boosts speed. 
               # User said "spikes_per_inference", which usually implies normalized by batch size.
               # Let's use a small batch size for stability.

class EvaluationConfig:
    # Standard config used in experiments
    hidden_dim_d = 128
    num_heads_H = 4
    num_layers_L = 3 # Common default, will verify
    surrogate_slope_k = 10.0
    local_lr_eta_local = 0.0
    
    class model:
        name = 'ablation_dsformer'
    class routing: enabled = True
    class phase_encoder: enabled = True
    class local_plasticity: enabled = False
    class dataset:
        max_timesteps = 1000
        state_dim = 0
        act_dim = 0
    device = 'cpu' # Use CPU for deterministic repro or GPU if available

def get_config_for_env(env_name, state_dim, act_dim):
    cfg = EvaluationConfig()
    cfg.dataset.state_dim = state_dim
    cfg.dataset.act_dim = act_dim
    # Adjust specific params if known, otherwise use defaults
    # Based on AblationDsFormer defaults in papers
    return cfg

# --- detailed spike recording ---
class SpikeRecorder:
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.reset_storage()
        self._register_hooks()
        
    def reset_storage(self):
        # Storage: [layer_idx][head_idx] -> scalar count
        # Storage: [timestep] -> scalar count
        self.layer_spikes = {} 
        self.head_spikes = {} # Not fully trivial since heads are fused in linear, but we can reshape q/k
        self.time_spikes = {}
        self.total_spikes = 0
        self.total_inferences = 0

    def _register_hooks(self):
        # We hook into SpikingAttentionBlock.lif 
        # But lif is a lambda in the code... 
        # Instead, let's hook into the block itself or q/k proj?
        # The block code: spikes_q, spikes_k = self.lif(q), self.lif(k)
        # We can hook 'ln1' to get input, but easier to monkey patch the lif of each block?
        # Or better: Forward Hook on the block module.
        # But block returns (x, attn) and spikes are internal local vars.
        # However, block has `self.spike_count`. 
        # But `self.spike_count` is a scalar sum.
        
        # Strategy: Define a strict forward hook on the LIF function if possible? 
        # No, LIF is lambda.
        # Strategy: Monkey patch the `lif` method of each block instance.
        pass

    def capture_pass(self, batch):
        # We will wrap the model's LIF functions temporarily
        
        # Accumulators for this specific pass
        pass_metrics = {
            'total': 0,
            'per_layer': [],
            'per_head': [], 
            'per_timestep': [] 
        }

        # We need to access internal activations. 
        # We will use a custom forward function for the blocks during eval.
        pass

# Re-implementing Recorder with a cleaner approach: wrapping the blocks
def run_evaluation(env_name):
    print(f"--- Evaluating {env_name} ---")
    
    env_out_dir = OUTPUT_DIR / env_name
    env_out_dir.mkdir(parents=True, exist_ok=True)

    # Load Normalization Info & Check Env Dims
    # We can just load the npz to get dims
    strat_path = FROZEN_DIR / env_name / "stratified_dataset.npz"
    random_path = FROZEN_DIR / env_name / "random_heavy_dataset.npz"
    
    if not strat_path.exists():
        print(f"Skipping {env_name}, missing files.")
        return

    # Load one to get dims
    tmp = np.load(strat_path)
    state_dim = tmp['states'].shape[-1]
    act_dim = tmp['actions'].shape[-1]
    
    cfg = get_config_for_env(env_name, state_dim, act_dim)
    model = AblationDsFormer(cfg)
    model.eval()
    
    # --- Prepare Monkey Patching for Detailed Log ---
    # We replace the 'lif' lambda in each block with a logging version
    
    for layer_idx, block in enumerate(model.blocks):
        # Current lif: lambda x: (torch.rand_like(x) < torch.sigmoid(x - k)).float()
        # We need to capture the output of this.
        
        # We'll attach a 'recorder' to the block
        block.recorder_id = layer_idx
        block.recorded_spikes = []
        
        old_lif = block.lif
        
        def make_hooked_lif(b, original_lif):
            def hooked_lif(x):
                spikes = original_lif(x) # (B, L, D)
                b.recorded_spikes.append(spikes.detach().cpu())
                return spikes
            return hooked_lif
            
        block.lif = make_hooked_lif(block, old_lif)

    datasets = {
        "stratified": strat_path,
        "random_heavy": random_path
    }

    for ds_name, ds_path in datasets.items():
        print(f"  > Dataset: {ds_name}")
        
        dataset = OfflineSequenceDataset(str(ds_path), seq_len=20)
        loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        results = []
        
        for run_id in range(NUM_RUNS):
            # Seed everything
            seed = 42 + run_id
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            
            # Get one batch
            try:
                batch = next(iter(loader))
            except StopIteration:
                loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
                batch = next(iter(loader))
                
            # Clear recordings
            for block in model.blocks:
                block.recorded_spikes = []
                
            # Forward
            with torch.no_grad():
                model(batch)
            
            # Process Recordings
            # Each block called LIF twice (q, k). So recorded_spikes has 2 entries per forward.
            # Shape: (B, L, D)
            
            total_spikes = 0
            layer_counts = []
            head_counts = np.zeros(cfg.num_heads_H)
            timestep_counts = np.zeros(20) # seq_len=20
            
            for layer_idx, block in enumerate(model.blocks):
                # q_spikes, k_spikes
                q_s = block.recorded_spikes[0] # (B, L, D)
                k_s = block.recorded_spikes[1]
                
                layer_sum = q_s.sum().item() + k_s.sum().item()
                layer_counts.append(layer_sum)
                total_spikes += layer_sum
                
                # Per Head
                # D = H * D_head
                # reshape (B, L, H, D_h)
                B, L, D = q_s.shape
                H = cfg.num_heads_H
                Dh = D // H
                
                q_h = q_s.view(B, L, H, Dh).sum(dim=(0, 1, 3)) # sum over batch, time, dim
                k_h = k_s.view(B, L, H, Dh).sum(dim=(0, 1, 3))
                head_counts += (q_h.numpy() + k_h.numpy())
                
                # Per Timestep
                q_t = q_s.sum(dim=(0, 2)) # sum over batch, dim -> (L,)
                k_t = k_s.sum(dim=(0, 2))
                # Ensure length matches
                valid_len = min(len(q_t), len(timestep_counts))
                timestep_counts[:valid_len] += (q_t[:valid_len].numpy() + k_t[:valid_len].numpy())

            # Normalize "per inference" usually means per sample in batch? 
            # Or just raw count for this specific input?
            # Metric: spikes_per_inference.
            spikes_per_inf = total_spikes / BATCH_SIZE
            
            # Extract Return info (Target Return for this sequence)
            # batch['returns_to_go'] is (B, L, 1). We take the start of the sequence.
            avg_rtg = batch['returns_to_go'][:, 0, 0].mean().item()

            # Log
            results.append({
                "run_id": run_id,
                "spikes_per_inference": spikes_per_inf,
                "avg_rtg": avg_rtg,
                "spikes_per_layer": str(layer_counts), # Store as string to simplify CSV
                "spikes_per_head": str(head_counts.tolist()),
                "spikes_per_timestep": str(timestep_counts.tolist())
            })
            
            if (run_id+1) % 10 == 0:
                print(f"    Run {run_id+1}/{NUM_RUNS} done.")

        # Save to CSV
        df = pd.DataFrame(results)
        out_csv = env_out_dir / f"{ds_name}.csv"
        df.to_csv(out_csv, index=False)
        print(f"    Saved {out_csv}")

def main():
    if not FROZEN_DIR.exists():
        print("Error: Frozen datasets not found. Run Step 1 first.")
        return
        
    for env in ENVS:
        run_evaluation(env)

if __name__ == "__main__":
    main()
