import numpy as np
import torch


def compute_returns_to_go(rewards, gamma=0.99):
    rtg = np.zeros_like(rewards, dtype=float)
    running = 0.0
    for i in reversed(range(len(rewards))):
        running = rewards[i] + gamma * running
        rtg[i] = running
    return rtg


def simple_logger(log_dict, step):
    entries = []
    for k, v in log_dict.items():
        if isinstance(v, float): entries.append(f"{k}={v:.3f}")
        else: entries.append(f"{k}={v}")
    print(f"[Step {step}] " + ", ".join(entries))


def save_checkpoint(model, optimizer, path):
    torch.save({
        'model_state': model.state_dict(),
        'optim_state': optimizer.state_dict(),
    }, path)