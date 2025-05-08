"""
Author: Vishal Pandey
Email: pandeyvishal.mlprof@gmail.com
"""
import torch, os
from src.models.snn_dt import SNNDecisionTransformer
from src.utils.trajectory_buffer import TrajectoryBuffer
from src.utils.helpers import compute_returns_to_go, simple_logger

device = "cuda" if torch.cuda.is_available() else "cpu"

def smoke():
    # toy config
    cfg = dict(
        state_dim=4, act_dim=2,
        hidden_size=32, max_length=10,
        n_layer=1, n_head=1, n_inner=64,
        time_window=5
    )
    model = SNNDecisionTransformer(**cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # build one dummy trajectory
    buf = TrajectoryBuffer()
    obs = torch.randn(cfg['state_dim']).numpy()
    for t in range(cfg['max_length']):
        buf.add(obs, t % cfg['act_dim'], float(t))
    traj = buf.get_trajectory()

    # batch‑ify
    states = torch.tensor(traj['states']).unsqueeze(0).to(device).float()
    actions = torch.tensor(traj['actions']).unsqueeze(0).unsqueeze(-1).to(device).long()
    rtg = torch.tensor(compute_returns_to_go(traj['rewards'])).unsqueeze(0).unsqueeze(-1).to(device).float()
    timesteps = torch.arange(cfg['max_length']).unsqueeze(0).to(device)

    # forward & backward
    out = model(states, actions, rtg, timesteps)
    loss = out.mean()
    loss.backward()
    simple_logger({'smoke_loss': loss.item()}, 0)
    print("✅ SNN-DT smoke pass complete.")

if __name__ == "__main__":
    smoke()