import time
import torch
import gym
import numpy as np

# Import your SNN-DT model
from external.DecisionSpikeFormer.models.snn_dt import SNNDecisionTransformer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(checkpoint_path, env_name, device=DEVICE):
    """Load SNN-DT from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = SNNDecisionTransformer(env_name=env_name).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

def evaluate(model, env_name, episodes=50, max_len=200, per_spike_energy=1e-9):
    env = gym.make(env_name)
    returns, losses, spikes_all, latencies = [], [], [], []

    for ep in range(episodes):
        state, done = env.reset(), False
        ep_return, ep_loss, ep_spikes, ep_steps = 0, 0, 0, 0

        while not done:
            state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)

            # Measure latency
            start_t = time.time()
            with torch.no_grad():
                action, extra = model.forward_with_stats(state_t)
            latency = (time.time() - start_t) * 1000  # ms

            # Extra dict contains spike count, internal loss etc.
            spike_count = extra.get("spikes", 0)
            step_loss = extra.get("loss", 0.0)

            state, reward, done, _ = env.step(action)
            ep_return += reward
            ep_loss += step_loss
            ep_spikes += spike_count
            ep_steps += 1
            latencies.append(latency)

        returns.append(ep_return)
        losses.append(ep_loss / max(ep_steps, 1))
        spikes_all.append(ep_spikes / max(ep_steps, 1))  # normalize per step

    env.close()

    # Aggregate stats
    result = {
        "avg_return": np.mean(returns),
        "std_return": np.std(returns),
        "avg_val_loss": np.mean(losses),
        "avg_spikes_per_step": np.mean(spikes_all),
        "avg_latency_ms": np.mean(latencies),
        "estimated_energy_J": np.mean(spikes_all) * per_spike_energy
    }
    return result

if __name__ == "__main__":
    # Example run (CartPole)
    checkpoint = "./logs/CartPole-v1_snn-dt_seed42_20250905_113833/checkpoint_epoch99.pt"
    env_name = "CartPole-v1"

    model = load_model(checkpoint, env_name)
    result = evaluate(model, env_name, episodes=50)

    print(f"[EVAL RESULT] {env_name}")
    for k, v in result.items():
        print(f"{k}: {v}")

