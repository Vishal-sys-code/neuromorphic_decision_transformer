
import numpy as np
from pathlib import Path

try:
    # Correct path
    base_dir = Path("d:/Github/neuromorphic_decision_transformer/ablation_studies/datasets/processed")
    # Find first npz file
    npz_files = list(base_dir.rglob("*.npz"))
    
    if not npz_files:
        print("No NPZ files found.")
    else:
        target_file = npz_files[0]
        print(f"Inspecting: {target_file}")
        data = np.load(target_file)
        print("Keys:", list(data.keys()))
        
        # Check for normalization stats
        if 'state_mean' in data:
            print("state_mean found")
        if 'state_std' in data:
            print("state_std found")
            
        # Check for trajectory info
        if 'timesteps' in data:
             print("timesteps found")

except Exception as e:
    print(f"Error: {e}")
