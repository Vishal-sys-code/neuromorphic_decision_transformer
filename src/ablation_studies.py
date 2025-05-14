# src/ablation_studies.py
import os, time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import DEVICE, dt_config
from src.models.snn_dt_patch import SNNDecisionTransformer

# Hyperparameter grids
TIME_WINDOWS = [5, 10, 20, 40]
MAX_LENGTHS  = [20, 50, 100]

# Number of forward runs per measurement
RUNS = 100

def build_dummy_inputs(model):
    B = 1
    L = model.max_length
    state_dim = model.state_dim
    act_dim   = model.act_dim

    states = torch.randn(B, L, state_dim, device=DEVICE)
    idx    = torch.randint(0, act_dim, (B, L), device=DEVICE)
    actions = torch.nn.functional.one_hot(idx, act_dim).to(torch.float32)
    returns = torch.randn(B, L, 1, device=DEVICE)
    timesteps = torch.arange(L, device=DEVICE).unsqueeze(0).repeat(B,1)
    return (states, actions, None, returns, timesteps)

def measure_spikes_and_latency(model, dummy_inputs, runs=RUNS):
    # Warm‑up
    for _ in range(10):
        _ = model(*dummy_inputs)

    # Zero spike counters
    for m in model.modules():
        if hasattr(m, "_spike_count"):
            m._spike_count.zero_()

    t0 = time.perf_counter()
    for _ in range(runs):
        _ = model(*dummy_inputs)
    t1 = time.perf_counter()

    total_spikes = sum(
        m.spike_count for m in model.modules()
        if hasattr(m, "spike_count")
    )
    avg_spikes = total_spikes / runs
    avg_ms     = (t1 - t0) / runs * 1000
    return avg_spikes, avg_ms

def main():
    records = []
    for max_len in MAX_LENGTHS:
        for tw in TIME_WINDOWS:
            # Build model instance
            cfg = dt_config.copy()
            # we'll need state_dim and act_dim: pick CartPole defaults or read from config
            # here assume CartPole for profiling; adjust if you need different env dims
            cfg.update(
                state_dim=4,
                act_dim=2,
                max_length=max_len,
                time_window=tw
            )
            model = SNNDecisionTransformer(**cfg).to(DEVICE).eval()

            dummy = build_dummy_inputs(model)
            spikes, ms = measure_spikes_and_latency(model, dummy)

            records.append({
                "max_length": max_len,
                "time_window": tw,
                "avg_spikes": spikes,
                "latency_ms": ms
            })
            print(f"  max_len={max_len}, tw={tw} → spikes={spikes:.1f}, lat={ms:.2f} ms")

    df = pd.DataFrame(records)
    os.makedirs("results", exist_ok=True)
    csv_path = os.path.join("results", "ablation_spikes_latency.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved results to {csv_path}")

    # Plot 1: spikes vs. time_window
    plt.figure()
    for max_len in MAX_LENGTHS:
        sub = df[df.max_length==max_len]
        plt.plot(sub.time_window, sub.avg_spikes, marker='o', label=f"max_len={max_len}")
    plt.xlabel("Time Window"); plt.ylabel("Avg Spikes"); plt.title("Spikes vs. Time Window")
    plt.legend(); plt.tight_layout()
    plt.savefig("results/ablation_spikes_vs_tw.png")
    print("Saved results/ablation_spikes_vs_tw.png")

    # Plot 2: latency vs. time_window
    plt.figure()
    for max_len in MAX_LENGTHS:
        sub = df[df.max_length==max_len]
        plt.plot(sub.time_window, sub.latency_ms, marker='o', label=f"max_len={max_len}")
    plt.xlabel("Time Window"); plt.ylabel("Latency (ms)"); plt.title("Latency vs. Time Window")
    plt.legend(); plt.tight_layout()
    plt.savefig("results/ablation_latency_vs_tw.png")
    print("Saved results/ablation_latency_vs_tw.png")

if __name__ == "__main__":
    main()