import time
import torch
import numpy as np

# import your model and config
from src.config import DEVICE, dt_config, max_length
from src.models.snn_dt_patch import SNNDecisionTransformer

def build_dummy_inputs(model):
    """
    Construct one batch of random inputs matching what your model.forward expects:
        states:       [batch=1, L, state_dim]
        actions_in:   [1, L, act_dim] or [1, 0, act_dim]
        returns:      [1, L, 1]
        timesteps:    [1, L]
    """
    cfg = model
    B, L = 1, max_length
    state_dim = model.state_dim
    act_dim   = model.act_dim

    states = torch.randn(B, L, state_dim, device=DEVICE)
    # for discrete: make random one‑hot
    actions_idx = torch.randint(0, act_dim, (B, L), device=DEVICE)
    actions_in  = torch.nn.functional.one_hot(actions_idx, act_dim).to(torch.float32)
    returns     = torch.randn(B, L, 1, device=DEVICE)
    timesteps   = torch.arange(L, device=DEVICE).unsqueeze(0).repeat(B,1)

    return (states, actions_in, None, returns, timesteps)

def measure_spikes_and_latency(model, dummy_inputs, runs=100):
    # 1) Warm‑up
    for _ in range(10):
        _ = model(*dummy_inputs)

    # 2) Zero all spike counters
    for m in model.modules():
        if hasattr(m, "_spike_count"):
            m._spike_count.zero_()

    # 3) Time the forward passes
    t0 = time.perf_counter()
    for _ in range(runs):
        _ = model(*dummy_inputs)
    t1 = time.perf_counter()

    # 4) Sum spikes over all layers
    total_spikes = sum(
        m.spike_count for m in model.modules()
        if hasattr(m, "spike_count")
    )
    avg_spikes = total_spikes / runs
    avg_ms     = (t1 - t0) / runs * 1000

    return avg_spikes, avg_ms

if __name__ == "__main__":
    # 5) Instantiate your trained SNN-DT (you can load a checkpoint here)
    cfg = dt_config.copy()
    # set dims...
    # e.g.:
    # cfg.update(state_dim=..., act_dim=..., max_length=max_length)
    model = SNNDecisionTransformer(**cfg).to(DEVICE).eval()
    # Optionally load weights:
    # ckpt = torch.load("checkpoints/offline_dt_CartPole-v1_9.pt", map_location=DEVICE)
    # model.load_state_dict(ckpt["model_state"])

    dummy = build_dummy_inputs(model)
    spikes, latency = measure_spikes_and_latency(model, dummy, runs=100)
    print(f"Avg spikes/forward: {spikes:.1f}, Avg latency: {latency:.3f} ms")