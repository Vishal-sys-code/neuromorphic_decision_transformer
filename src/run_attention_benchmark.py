import time
import torch
import torch.nn as nn
from models.spiking_layers import SpikingSelfAttention

# Config
B, S, E, H, T = 32, 50, 64, 4, 10
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Prepare data
x = torch.randn(B, S, E, device=device)

# 1) Spiking Self‑Attention
ssa = SpikingSelfAttention(embed_dim=E, num_heads=H, time_window=T).to(device)
start = time.time()
out_snn = ssa(x)
torch.cuda.synchronize() if device=='cuda' else None
snn_time = (time.time() - start) * 1000

# 2) Standard MultiheadAttention
mha = nn.MultiheadAttention(E, H).to(device)
# nn.MultiheadAttention expects [S, B, E]
x_t = x.transpose(0,1)
start = time.time()
out2, _ = mha(x_t, x_t, x_t)
torch.cuda.synchronize() if device=='cuda' else None
ann_time = (time.time() - start) * 1000

print(f"SNN‑SA forward time: {snn_time:.2f} ms")
print(f"ANN‑MHA forward time: {ann_time:.2f} ms")
print("Output shapes:", out_snn.shape, out2.transpose(0,1).shape)