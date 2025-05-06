import torch
import time
from src.models.snn_mlp import SpikingMLP

model = SpikingMLP().cuda()
x = torch.rand(64, 28*28).cuda()

start = time.time()
out = model(x)
torch.cuda.synchronize()
end = time.time()

print(f"SNN Forward time: {(end - start)*1000:.2f} ms")
print(f"Output shape: {out.shape}")