# src/benchmark_snn_vs_ann.py

import time
import torch
import numpy as np
from src.config import DEVICE, dt_config, max_length
from src.models.snn_dt_patch import SNNDecisionTransformer
from external.decision_transformer.gym.decision_transformer.models.decision_transformer import DecisionTransformer as GymDecisionTransformer

def build_dummy_inputs(model):
    """
    Return a tuple (states, actions_in, None, returns, timesteps)
    matching model.get_action / forward signature.
    """
    B, L = 1, max_length
    state_dim = model.state_dim
    act_dim   = model.act_dim

    states = torch.randn(B, L, state_dim, device=DEVICE)
    # one‐hot for discrete
    actions_idx = torch.randint(0, act_dim, (B, L), device=DEVICE)
    actions_in  = torch.nn.functional.one_hot(actions_idx, act_dim).to(torch.float32)
    returns     = torch.randn(B, L, 1, device=DEVICE)
    timesteps   = torch.arange(L, device=DEVICE).unsqueeze(0).repeat(B,1)
    return (states, actions_in, None, returns, timesteps)

def measure_snn(model, dummy, runs=100):
    # warm‐up
    for _ in range(10): _ = model(*dummy)

    # zero spike counters
    for m in model.modules():
        if hasattr(m, "_spike_count"):
            m._spike_count.zero_()

    t0 = time.perf_counter()
    for _ in range(runs):
        _ = model(*dummy)
    t1 = time.perf_counter()

    total_spikes = sum(
        m.spike_count for m in model.modules()
        if hasattr(m, "spike_count")
    )
    return total_spikes/runs, (t1-t0)/runs*1000

def measure_ann(model, dummy, runs=100):
    # warm‐up
    for _ in range(10): _ = model(*dummy)

    t0 = time.perf_counter()
    for _ in range(runs):
        _ = model(*dummy)
    t1 = time.perf_counter()

    return (t1-t0)/runs*1000

if __name__=="__main__":
    # Build and load your SNN‐DT
    snn_cfg = dt_config.copy()
    snn_cfg.update(state_dim=4, act_dim=2, max_length=max_length)  # e.g. CartPole dims
    snn_model = SNNDecisionTransformer(**snn_cfg).to(DEVICE).eval()
    # Optionally load weights:
    # ckpt = torch.load("checkpoints/offline_dt_CartPole-v1_9.pt", map_location=DEVICE)
    # snn_model.load_state_dict(ckpt["model_state"])

    # Build ANN‐DT
    ann_cfg = dt_config.copy()
    ann_cfg.update(state_dim=4, act_dim=2, max_length=max_length)
    ann_model = GymDecisionTransformer(**ann_cfg).to(DEVICE).eval()

    dummy = build_dummy_inputs(snn_model)

    snn_spikes, snn_ms = measure_snn(snn_model, dummy)
    ann_ms            = measure_ann(ann_model, dummy)

    print(f"{'Model':<10} {'Spikes/forward':>15}   {'Latency (ms)':>12}")
    print("-"*40)
    print(f"{'SNN‑DT':<10} {snn_spikes:15.1f}   {snn_ms:12.3f}")
    print(f"{'ANN‑DT':<10} {'—':>15}   {ann_ms:12.3f}")