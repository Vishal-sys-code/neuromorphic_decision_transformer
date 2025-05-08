"""
Author: Vishal Pandey
Email: pandeyvishal.mlprof@gmail.com
"""
import numpy as np

def compute_returns_to_go(rewards, gamma=0.99):
    rtg = np.zeros_like(rewards, dtype=float)
    running = 0.0
    for i in reversed(range(len(rewards))):
        running = rewards[i] + gamma * running
        rtg[i] = running
    return rtg

def simple_logger(log_dict, step):
    print(f"Step {step}: " + ", ".join(f"{k}={v:.3f}" for k,v in log_dict.items()))