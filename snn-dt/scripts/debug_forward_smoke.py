# scripts/debug_forward_smoke.py
import torch, yaml, sys, os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # repo root

from src.utils.config import AttrDict
from src.utils.models import get_model
import numpy as np

# load config (small)
cfg_path = Path("configs/sdt_mountaincar.yaml")
with open(cfg_path) as f:
    raw = yaml.safe_load(f)

cfg = {
  "model": {"name":"snn_dt", "d_model": raw.get("hidden_dim",128), "n_heads": raw.get("n_heads",2), "n_layers": raw.get("n_layers",2), "seq_len": raw.get("seq_len", 20)},
  "dataset": {"max_timesteps": raw.get("seq_len",20), "state_dim": 2, "act_dim": 3, "is_discrete": True},
  "snn": {"debug_use_fake_lif": True, "v_th": 0.05, "current_scale": 10.0},
  "training": {"device": "cuda" if torch.cuda.is_available() else "cpu"}
}
cfg = AttrDict(cfg)

model = get_model(cfg).to(cfg.training.device)
model.eval()

# build a synthetic batch with larger magnitudes
B, S = 8, raw.get("seq_len",20)
states = torch.randn(B, S, cfg.dataset.state_dim) * 5.0
actions = torch.zeros(B, S, cfg.dataset.act_dim)  # dummy
rtg = torch.randn(B, S, 1) * 5.0
timesteps = torch.arange(S).unsqueeze(0).repeat(B,1)

batch = {"states": states.to(cfg.training.device),
         "actions": actions.to(cfg.training.device),
         "returns_to_go": rtg.to(cfg.training.device),
         "timesteps": timesteps.to(cfg.training.device),
         "mask": torch.ones(B, S).to(cfg.training.device)}

with torch.no_grad():
    out = model(batch)

print("Forward ok. Action preds shape:", out.shape)
# if model has last_logs/diagnostics print them
if hasattr(model, "last_diagnostics"):
    print("last_diagnostics keys:", model.last_diagnostics.keys())
    import json
    print(json.dumps({k: float(v) if isinstance(v, (int,float)) else str(type(v)) for k,v in model.last_diagnostics.items()}, indent=2))
    
# check spike counts
if hasattr(model, "count_spikes"):
    print("Normalized spikes:", model.count_spikes())
else:
    print("model.count_spikes not present")