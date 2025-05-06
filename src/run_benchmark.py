import os, sys
current_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

import torch
import time
# adjust import if needed for your project structure
from src.models.snn_mlp import SpikingMLP  

# device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# initialize model on chosen device
model = SpikingMLP().to(device)

# sample input on the same device
x = torch.rand(64, 28*28).to(device)

# benchmark forward pass
start = time.time()
out = model(x)
# if using CUDA, synchronize the device to get accurate timing
if device.type == "cuda":
    torch.cuda.synchronize()
end = time.time()

print(f"SNN Forward time: {(end - start)*1000:.2f} ms")
print(f"Output shape: {out.shape}")