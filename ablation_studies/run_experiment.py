import warnings
warnings.filterwarnings("ignore")
import argparse
import logging
import os
import sys
import time
import json
import subprocess
import pickle
from pathlib import Path
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
import gymnasium as gym

# --- Add Project Root to sys.path ---
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
# Add snn-dt/src to sys.path to allow for model imports
snn_dt_src_path = project_root / 'snn-dt' / 'src'
if snn_dt_src_path.exists():
    sys.path.insert(0, str(snn_dt_src_path))

# --- Local Imports ---
from ablation_studies.src.datasets import OfflineSequenceDataset, OfflineTransitionDataset
from ablation_studies.src.models.ablation_dsformer import AblationDsFormer, BasePolicy

# --- Configuration Management ---
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        for k, v in self.items():
            if isinstance(v, dict) and not isinstance(v, AttrDict):
                self[k] = AttrDict(v)
        self.__dict__ = self
    def __getattr__(self, name):
        if name in self.__dict__: return self.__dict__[name]
        if name in self: return self[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

def load_config(contract_path, variant_path):
    with open(contract_path, 'r') as f: base_cfg = yaml.safe_load(f)
    if variant_path.exists():
        with open(variant_path, 'r') as f: variant_cfg = yaml.safe_load(f)
        for key, value in variant_cfg.items():
            if isinstance(value, dict) and key in base_cfg and isinstance(base_cfg[key], dict):
                base_cfg[key].update(value)
            else:
                base_cfg[key] = value
    base_cfg['dataset'] = {}
    return AttrDict(base_cfg)

# --- Model Factory ---
def get_model(cfg):
    model_name_map = {
        'dt': ('models.dt', 'DecisionTransformer'),
        'snn_dt': ('models.snn_dt', 'SnnDt'),
        'iql': ('models.iql', 'IQL'),
        'cql': ('models.cql', 'CQL'),
        'ablation_dsformer': ('ablation_studies.src.models.ablation_dsformer', 'AblationDsFormer'),
    }
    
    if cfg.model.name not in model_name_map:
        raise NotImplementedError(f"Model {cfg.model.name} not supported.")

    module_name, class_name = model_name_map[cfg.model.name]
    module = __import__(module_name, fromlist=[class_name])
    model_class = getattr(module, class_name)
    
    model = model_class(cfg)
    
    # Monkey-patch save/load for non-BasePolicy models
    if not isinstance(model, BasePolicy):
        setattr(model, 'save', lambda path: torch.save(model.state_dict(), path))
        setattr(model, 'load', lambda path, device='cpu': model.load_state_dict(torch.load(path, map_location=device)))
        
    return model

# --- Evaluation ---
@torch.no_grad()
def evaluate_policy(model, env_name, cfg):
    model.eval()
    if hasattr(model, 'reset_spike_counts'): model.reset_spike_counts()
    env = gym.make(env_name)
    
    target_returns = {'CartPole-v1': 500, 'Acrobot-v1': -100, 'Pendulum-v1': -120}
    target_return = target_returns.get(env_name, 0)

    total_rewards = []
    for _ in range(cfg.eval_rollouts):
        if hasattr(model, 'reset_state'):
            model.reset_state()
            
        state, _ = env.reset()
        done, episode_return, t = False, 0, 0
        
        # Decide evaluation loop type based on model type
        is_transition_model = cfg.model.name in ['iql', 'cql']
        
        if is_transition_model: # For IQL, CQL
            while not done:
                action = model.predict_action(state)
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_return += reward
        else: # For DT, SNN-DT, AblationDsFormer
            state_dim = cfg.dataset.state_dim
            # For discrete actions, we want shape (B, L, 1) containing indices
            # For continuous, we want (B, L, act_dim)
            is_discrete = getattr(cfg.dataset, 'is_discrete', False)
            act_dim = 1 if is_discrete else cfg.dataset.act_dim
            max_len = cfg.sequence_length_N
            device = cfg.device
            
            # Initialize context buffers (fixed size N)
            states = torch.zeros(1, max_len, state_dim, dtype=torch.float32, device=device)
            actions = torch.zeros(1, max_len, act_dim, dtype=torch.float32, device=device)
            rtgs = torch.full((1, max_len, 1), target_return, dtype=torch.float32, device=device)
            timesteps = torch.zeros(1, max_len, 1, dtype=torch.long, device=device)
            
            # Set initial state
            states[0, 0] = torch.from_numpy(state).to(device=device, dtype=torch.float32)
            timesteps[0, 0] = 0
            
            while not done:
                # Use the valid index for prediction (t if t < N, else N-1)
                valid_idx = min(t, max_len - 1)
                
                batch = {
                    "states": states, 
                    "actions": actions, 
                    "returns_to_go": rtgs, 
                    "timesteps": timesteps.squeeze(-1),
                    "mask": torch.ones_like(states[:, :, 0])
                }
                
                output = model(batch)
                if isinstance(output, tuple):
                    action_pred = output[0]
                else:
                    action_pred = output
                action = action_pred[0, valid_idx].cpu().numpy()
                
                if isinstance(env.action_space, gym.spaces.Discrete):
                    if cfg.dataset.act_dim == 1:
                        # Clamp action to valid range [0, n-1]
                        raw_action = int(np.round(action.item()))
                        action = max(0, min(raw_action, env.action_space.n - 1))
                    else:
                        action = int(np.argmax(action))
                
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated # Respect env termination
                episode_return += reward
                
                if not done:
                    if t < max_len - 1:
                        # Append to buffer
                        actions[0, t] = torch.tensor(action, device=device)
                        states[0, t+1] = torch.from_numpy(next_state).to(device)
                        rtgs[0, t+1] = rtgs[0, t] - reward
                        timesteps[0, t+1] = t + 1
                    else:
                        # Sliding window shift
                        actions[0, -1] = torch.tensor(action, device=device) # Store taken action
                        
                        states = torch.roll(states, shifts=-1, dims=1)
                        actions = torch.roll(actions, shifts=-1, dims=1)
                        rtgs = torch.roll(rtgs, shifts=-1, dims=1)
                        timesteps = torch.roll(timesteps, shifts=-1, dims=1)
                        
                        states[0, -1] = torch.from_numpy(next_state).to(device)
                        actions[0, -1] = 0.0 # Placeholder for next action
                        rtgs[0, -1] = rtgs[0, -2] - reward
                        timesteps[0, -1] = t + 1
                        
                t += 1
        total_rewards.append(episode_return)

    env.close()
    return {"val/mean_return": np.mean(total_rewards), "val/std_return": np.std(total_rewards)}

# --- Main Training Loop ---
def train(cfg, logger):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_path = project_root / f"ablation_studies/datasets/processed/{cfg.env}/stratified_dataset.npz"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset for {cfg.env} not found at {dataset_path}.")
    
    with np.load(dataset_path, allow_pickle=True) as data:
        meta_item = data['metadata'].item()
        if isinstance(meta_item, str):
            metadata = json.loads(meta_item)
        else:
            metadata = pickle.loads(meta_item)
        cfg.dataset.update(metadata)

        if 'state_dim' not in cfg.dataset:
            if 'states' in data: cfg.dataset.state_dim = data['states'].shape[-1]
            elif 'state_mean' in data: cfg.dataset.state_dim = data['state_mean'].shape[-1]
    
        if 'act_dim' not in cfg.dataset:
            if 'actions' in data: cfg.dataset.act_dim = data['actions'].shape[-1]
            elif 'action_dim' in metadata: cfg.dataset.act_dim = metadata['action_dim']
    
    is_transition_model = cfg.model.name in ['iql', 'cql']
    DatasetClass = OfflineTransitionDataset if is_transition_model else OfflineSequenceDataset
    dataset_args = {'path': str(dataset_path)}
    if not is_transition_model: dataset_args['seq_len'] = cfg.sequence_length_N
    dataset = DatasetClass(**dataset_args)
    # NOTE: num_workers is set to 0 to avoid a hanging issue with multiprocessing.
    train_loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    
    model = get_model(cfg).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay)) if list(model.parameters()) else None
    loss_fn = torch.nn.MSELoss()

    logger.info(json.dumps({"train/param_count": sum(p.numel() for p in model.parameters())}))

    # Create save directory if it doesn't exist
    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}", file=sys.stderr)):
            batch = {k: v.to(cfg.device) for k, v in batch.items()}
            
            if is_transition_model:
                loss = model.learn(batch)['value_loss']
            else:
                optimizer.zero_grad()
                output = model(batch)
                if isinstance(output, tuple):
                    action_preds = output[0]
                else:
                    action_preds = output
                loss = loss_fn(action_preds, batch["actions"])
                loss.backward()
                optimizer.step()
            
            if (epoch * len(train_loader) + batch_idx) % cfg.log_interval_steps == 0:
                logger.info(json.dumps({"train/step": epoch * len(train_loader) + batch_idx, "train/loss": loss.item()}))
        
        if epoch % cfg.checkpoint_interval_epochs == 0:
            val_metrics = evaluate_policy(model, cfg.env, cfg)
            logger.info(json.dumps({"epoch": epoch, **val_metrics}))
            model.save(Path(cfg.save_dir) / f"ckpt_epoch_{epoch}.pt")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True)
    parser.add_argument("--env", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--contract", default="experiment_contract.yaml", help="Path to experiment contract yaml")
    args = parser.parse_args()

    variant_path = Path(__file__).parent / f"configs/phase2/{args.variant}.yaml"
    cfg = load_config(Path(__file__).parent / args.contract, variant_path)
    cfg.env, cfg.seed = args.env, args.seed
    model_name = cfg.get('model', {}).get('name', args.variant if args.variant in ['dt', 'snn_dt', 'iql', 'cql'] else 'ablation_dsformer')
    if 'model' not in cfg: cfg['model'] = AttrDict()
    elif isinstance(cfg['model'], dict) and not isinstance(cfg['model'], AttrDict): cfg['model'] = AttrDict(cfg['model'])
    
    cfg.model.name = model_name
    
    run_name = cfg.model.name if cfg.model.name != 'ablation_dsformer' else args.variant

    # --- Compatibility Bridge (Contract -> snn-dt models) ---
    # Map top-level contract content to cfg.model and cfg.snn where expected
    
    # Model params
    if 'd_model' not in cfg.model and 'hidden_dim_d' in cfg:
        cfg.model.d_model = cfg.hidden_dim_d
    if 'n_heads' not in cfg.model and 'num_heads_H' in cfg:
        cfg.model.n_heads = cfg.num_heads_H
    if 'n_layers' not in cfg.model and 'num_layers_L' in cfg:
        cfg.model.n_layers = cfg.num_layers_L
    if 'seq_len' not in cfg.model and 'sequence_length_N' in cfg:
        cfg.model.seq_len = cfg.sequence_length_N
    if 'action_tanh' not in cfg.model:
        cfg.model.action_tanh = False

    # SNN params
    if 'snn' not in cfg:
        cfg['snn'] = AttrDict()
    elif isinstance(cfg['snn'], dict) and not isinstance(cfg['snn'], AttrDict):
        cfg['snn'] = AttrDict(cfg['snn'])
        
    if 'biological_time' not in cfg.snn and 'spiking_window_T' in cfg:
        cfg.snn.biological_time = cfg.spiking_window_T
    if 'surrogate_k' not in cfg.snn and 'surrogate_slope_k' in cfg:
        cfg.snn.surrogate_k = cfg.surrogate_slope_k
    if 'eta_local' not in cfg.snn and 'local_lr_eta_local' in cfg:
        cfg.snn.eta_local = cfg.local_lr_eta_local
    
    # Check environment for discreteness
    import gymnasium as gym
    act_dim_override = None
    try:
        tmp_env = gym.make(cfg.env)
        is_discrete = isinstance(tmp_env.action_space, gym.spaces.Discrete)
        if is_discrete and hasattr(tmp_env.action_space, 'n'):
            act_dim_override = tmp_env.action_space.n
        tmp_env.close()
    except Exception as e:
        print(f"Warning: Could not determine discreteness from env {cfg.env}: {e}")
        is_discrete = False # Default

    if 'dataset' not in cfg:
        cfg['dataset'] = AttrDict()
    elif isinstance(cfg['dataset'], dict) and not isinstance(cfg['dataset'], AttrDict):
        cfg['dataset'] = AttrDict(cfg['dataset'])
    
    if 'is_discrete' not in cfg.dataset:
        cfg.dataset.is_discrete = is_discrete
    
    if act_dim_override is not None:
        cfg.dataset.act_dim = act_dim_override
    # ----------------------------------------------------
    
    save_dir = Path(__file__).parent / f"runs/{run_name}/seed_{args.seed}/{cfg.env}"
    save_dir.mkdir(parents=True, exist_ok=True)
    cfg.save_dir = str(save_dir)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = [logging.StreamHandler(), logging.FileHandler(save_dir / "metrics.jsonl")]
    for handler in logger.handlers: handler.setFormatter(logging.Formatter('%(message)s'))

    with open(save_dir / "run_info.txt", "w") as f:
        f.write(f"Command: {' '.join(sys.argv)}\nGit Hash: {subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')}\n")

    train(cfg, logger)

if __name__ == "__main__":
    main()