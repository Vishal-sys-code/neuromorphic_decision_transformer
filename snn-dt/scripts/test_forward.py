import sys
from pathlib import Path
snn_dt_root = Path(__file__).resolve().parent
project_root = snn_dt_root.parent
sys.path.append(str(snn_dt_root))
sys.path.append(str(project_root))

import numpy as np
import torch
from src.utils.config import AttrDict
from src.utils.models import get_model

npz_path = project_root / '..' / 'data' / 'CartPole-v1' / 'dataset.npz'
# Normalize path in case project layout differs
npz_path = (project_root / 'data' / 'CartPole-v1' / 'dataset.npz')
print('Loading dataset from', npz_path)
data = np.load(str(npz_path))

# Build a minimal cfg using dataset shapes (avoid relying on metadata format)
state_dim = data['states'].shape[2]
# actions stored as shape (N, T, 1) -- infer number of discrete actions
act_dim = int(data['actions'].max()) + 1 if data['actions'].size > 0 else 1
max_timesteps = data['timesteps'].shape[1]

cfg = {
    'model': {'name': 'snn_dt', 'd_model': 128, 'n_heads': 4, 'n_layers': 4},
    'training': {'device': 'cpu'},
    'dataset': {'path': '', 'state_dim': state_dim, 'act_dim': act_dim, 'max_timesteps': max_timesteps, 'is_discrete': True},
    'env': 'CartPole-v1', 'seed': 42, 'save_dir': 'results/tmp',
    'snn': {'lif_tau': 20.0, 'surrogate_k': 25.0, 'use_plasticity': False}
}
cfg = AttrDict(cfg)
model = get_model(cfg)
model.eval()

batch = {
    'states': torch.from_numpy(data['states'][:2]).float(),
    'actions': torch.from_numpy(data['actions'][:2]).long(),
    'returns_to_go': torch.from_numpy(data['returns_to_go'][:2]).float(),
    'timesteps': torch.from_numpy(data['timesteps'][:2]).long(),
    'mask': torch.from_numpy(data['mask'][:2]).float()
}

out = model(batch)
print('forward OK, output shape', out.shape)