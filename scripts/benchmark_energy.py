import os
import sys
import argparse
import time
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import gym
try:
    import d4rl
except ImportError:
    print("Warning: d4rl not found. MuJoCo environments won't be registered.")

# Setup paths
repo_root = Path(__file__).resolve().parent.parent
if (repo_root / 'snn-dt').exists():
    sys.path.insert(0, str(repo_root / 'snn-dt'))
sys.path.append(str(repo_root))

from scripts.energy_logger import EnergyTimeLogger
from src.utils.config import AttrDict
from src.utils.seed import seed_everything
from scripts.eval import evaluate_policy
from src.utils.evaluation import create_env
from src.utils.d4rl_dataset import D4RLSequenceDataset

def get_model(cfg):
    if cfg.model.name == 'dt':
        from src.models.dt import DecisionTransformer
        return DecisionTransformer(cfg)
    elif cfg.model.name == 'dsformer':
        from src.models.dsformer import DsFormer
        return DsFormer(cfg)
    elif cfg.model.name == 'snn_dt':
        from src.models.snn_dt import SnnDt
        return SnnDt(cfg)
    else:
        raise ValueError(f"Unknown model: {cfg.model.name}")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def build_cfg(env_name, model_name, seed):
    cfg_raw = {
        "model": {
            "name": model_name,
            "seq_len": 20,
            "d_model": 128,
            "n_heads": 4,
            "n_layers": 4,
            "action_tanh": False,
        },
        "training": {
            "batch_size": 64,
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "epochs": 1,
            "batches_per_epoch": 200,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "simulator_available": True,
        },
        "dataset": {
            "mode": "d4rl_direct",
            "path": str(repo_root / "data/d4rl_raw" / f"{env_name.replace('-medium-', '_medium-')}.hdf5"),
        },
        "hyperparameters": {
             "eval_episodes": 10
        },
        "env": env_name,
        "seed": seed,
        "snn": {
            "lif_tau": 20.0,
            "surrogate_k": 25.0,
            "v_th": 1.0,
            "current_scale": 0.2,
            "use_plasticity": False
        }
    }
    return AttrDict(cfg_raw)

def run_benchmark():
    envs = ["hopper-medium-v2", "walker2d-medium-v2", "halfcheetah-medium-v2"]
    models = ["dt", "dsformer", "snn_dt"]
    seeds = [42, 123, 999]
    
    results = []
    
    # Pre-calculate baseline DT parameter count
    dummy_cfg = build_cfg("hopper-medium-v2", "dt", 42)
    dummy_cfg.dataset.state_dim = 11
    dummy_cfg.dataset.act_dim = 3
    dummy_cfg.dataset.max_timesteps = 1000
    dummy_cfg.dataset.is_discrete = False
    dummy_dt = get_model(dummy_cfg)
    target_params = count_parameters(dummy_dt)
    print(f"Target parameter count (DT): {target_params}")
    
    for env in envs:
        for seed in seeds:
            for model_name in models:
                print(f"--- Running {model_name} on {env} (Seed {seed}) ---")
                
                cfg = build_cfg(env, model_name, seed)
                seed_everything(cfg.seed)
                
                # Load dataset
                if not os.path.exists(cfg.dataset.path):
                    print(f"Dataset {cfg.dataset.path} missing. Skipping.")
                    continue
                
                dataset = D4RLSequenceDataset(cfg.env, data_dir=str(repo_root / "data/d4rl_raw"), seq_len=cfg.model.seq_len)
                cfg.dataset.state_dim = dataset.state_dim
                cfg.dataset.act_dim = dataset.act_dim
                cfg.dataset.max_timesteps = 1000
                cfg.dataset.is_discrete = False
                
                # Model balancing
                if model_name == "dsformer":
                    # Simple heuristic: adjust d_model slightly to match target params
                    # We can brute force the d_model around 128 to get close
                    best_diff = float('inf')
                    best_d_model = 128
                    best_n_heads = 4
                    for test_d_model in range(110, 150):
                        for test_n_heads in [2, 4]:
                            if test_d_model % test_n_heads != 0: continue
                            cfg.model.d_model = test_d_model
                            cfg.model.n_heads = test_n_heads
                            temp_model = get_model(cfg)
                            p_count = count_parameters(temp_model)
                            diff = abs(p_count - target_params)
                            if diff < best_diff:
                                best_diff = diff
                                best_d_model = test_d_model
                                best_n_heads = test_n_heads
                    
                    cfg.model.d_model = best_d_model
                    cfg.model.n_heads = best_n_heads
                    print(f"Adjusted DSFormer to d_model={best_d_model}, n_heads={best_n_heads}")
                elif model_name == "snn_dt":
                    pass # Similar check could be added if snn_dt deviates from DT
                
                model = get_model(cfg).to(cfg.training.device)
                actual_params = count_parameters(model)
                print(f"Model {model_name} params: {actual_params}")
                
                optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.training.lr), weight_decay=float(cfg.training.weight_decay))
                loss_fn = torch.nn.MSELoss()
                
                from torch.utils.data import DataLoader
                train_loader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=True)
                
                # --- TRAINING PHASE ---
                train_iter = iter(train_loader)
                epoch_losses = []
                print("Starting training (200 steps)...")
                with EnergyTimeLogger() as train_logger:
                    model.train()
                    for i in tqdm(range(cfg.training.batches_per_epoch)):
                        try:
                            batch = next(train_iter)
                        except StopIteration:
                            train_iter = iter(train_loader)
                            batch = next(train_iter)
                            
                        for k, v in batch.items():
                            batch[k] = v.to(cfg.training.device)
                            
                        optimizer.zero_grad()
                        
                        # Forward pass based on model architecture
                        if hasattr(model, 'forward'):
                            # Handling diff return styles
                            action_preds = model(batch)
                            if isinstance(action_preds, tuple):
                                action_preds = action_preds[1] # e.g. for DSF or standard DT that returns (s, a, r)
                        else:
                            action_preds = model(batch)
                            
                        action_targets = batch["actions"]
                        
                        # Alignment
                        if action_preds.shape[1] > action_targets.shape[1]:
                             action_preds = action_preds[:, :action_targets.shape[1]]
                             
                        mask = batch["mask"][:, :action_targets.shape[1]].reshape(-1).bool()
                        action_preds = action_preds.reshape(-1, cfg.dataset.act_dim)[mask]
                        action_targets = action_targets.reshape(-1, cfg.dataset.act_dim)[mask]
                        
                        loss = loss_fn(action_preds, action_targets)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        epoch_losses.append(loss.item())

                train_energy = train_logger.total_energy_joules
                train_time = train_logger.total_time
                print(f"Training Energy: {train_energy:.2f} J | Training Time: {train_time:.2f} s")
                
                # --- EVALUATION PHASE ---
                print("Starting evaluation...")
                env_obj = create_env(cfg.env, simulator_available=True, dataset_path=cfg.dataset.path)
                with EnergyTimeLogger() as eval_logger:
                    eval_results = evaluate_policy(
                        model,
                        env_obj,
                        cfg,
                        episodes=cfg.hyperparameters.eval_episodes,
                        state_mean=dataset.state_mean,
                        state_std=dataset.state_std
                    )
                
                eval_energy = eval_logger.total_energy_joules
                eval_time = eval_logger.total_time
                
                # Calculate metrics
                energy_per_step = train_energy / cfg.training.batches_per_epoch
                energy_per_eval_episode = eval_energy / cfg.hyperparameters.eval_episodes
                eval_steps = eval_results.get("total_steps", cfg.dataset.max_timesteps * cfg.hyperparameters.eval_episodes)
                energy_per_eval_step = eval_energy / max(1, eval_steps)
                
                print(f"Eval Return: {eval_results['return_mean']:.2f} | Eval Energy: {eval_energy:.2f} J")
                
                results.append({
                    "env": env,
                    "model": model_name,
                    "seed": seed,
                    "params": actual_params,
                    "train_loss": np.mean(epoch_losses),
                    "train_energy_joules": train_energy,
                    "train_time_s": train_time,
                    "energy_per_train_step": energy_per_step,
                    "eval_return_mean": eval_results['return_mean'],
                    "eval_return_std": eval_results.get('return_std', 0.0),
                    "eval_energy_joules": eval_energy,
                    "eval_time_s": eval_time,
                    "energy_per_eval_episode": energy_per_eval_episode,
                    "energy_per_eval_step": energy_per_eval_step
                })
                
    # Save results
    results_dir = repo_root / "results"
    results_dir.mkdir(exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(results_dir / "energy_benchmark_results.csv", index=False)
    print("Benchmark complete. Results saved to energy_benchmark_results.csv")

if __name__ == "__main__":
    run_benchmark()
