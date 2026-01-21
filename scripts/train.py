import argparse
import logging
import os
import sys
import time
import tempfile
import atexit
import shutil
import csv
from pathlib import Path
from datetime import datetime

# Add project root and snn-dt directory to sys.path
snn_dt_root = Path(__file__).resolve().parent.parent
project_root = snn_dt_root.parent

# Prioritize snn-dt/src because it contains the complete source (cql.py, config.py) 
# which are missing from root/src
if (snn_dt_root / 'snn-dt').exists():
    sys.path.insert(0, str(snn_dt_root / 'snn-dt'))

sys.path.append(str(snn_dt_root))
sys.path.append(str(project_root))

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from src.utils.config import AttrDict
# from src.utils.models import get_model # Imported below now or verified
# from src.utils.seed import seed_everything
from src.utils.seed import seed_everything
from src.utils.evaluation import create_env
from scripts.eval import evaluate_policy
from src.utils.d4rl_dataset import D4RLSequenceDataset, D4RLTransitionDataset

# Lazy Imports for Models to avoid norse/tensorflow crash on Python 3.13
def get_model_class(name):
    if name == 'cql':
        from src.models.cql import CQL
        return CQL
    elif name == 'dt':
        from src.models.dt import DecisionTransformer
        return DecisionTransformer
    elif name == 'dsformer':
        from src.models.dsformer import DsFormer
        return DsFormer
    elif name == 'iql':
        from src.models.iql import IQL
        return IQL
    elif name == 'snn_dt':
        from src.models.snn_dt import SnnDt
        return SnnDt
    else:
        raise ValueError(f"Unknown model: {name}")

# Re-implement get_model here
def get_model(cfg):
    model_class = get_model_class(cfg.model.name)
    # All models (CQL, DT, SnnDt, etc.) accept cfg as the single argument
    return model_class(cfg)



class OfflineDataset(Dataset):
    def __init__(self, dataset_path):
        # Load with mmap_mode='r' to keep data on disk
        self.data = np.load(dataset_path, mmap_mode='r')
        self.states = self.data["states"]
        self.actions = self.data["actions"]
        self.returns_to_go = self.data["returns_to_go"]
        self.timesteps = self.data["timesteps"]
        self.mask = self.data["mask"]

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        # Convert to tensor only when accessed
        return {
            "states": torch.as_tensor(self.states[idx], dtype=torch.float32),
            "actions": torch.as_tensor(self.actions[idx], dtype=torch.float32),
            "returns_to_go": torch.as_tensor(self.returns_to_go[idx], dtype=torch.float32),
            "timesteps": torch.as_tensor(self.timesteps[idx], dtype=torch.long),
            "mask": torch.as_tensor(self.mask[idx], dtype=torch.float32),
        }


class OfflineTransitionDataset(Dataset):
    def __init__(self, dataset_path):
        data = np.load(dataset_path, mmap_mode='r')
        
        total_transitions = 0
        for i in range(data['mask'].shape[0]):
            total_transitions += int(data['mask'][i].sum())

        state_dim = data['states'].shape[2]
        action_dim = data['actions'].shape[2] if 'actions' in data.keys() and data['actions'].shape[0] > 0 else 0

        # Pre-allocate arrays in memory
        self.states = np.empty((total_transitions, state_dim), dtype=np.float32)
        self.actions = np.empty((total_transitions, action_dim), dtype=np.int64) # This might be float for continuous? IQL usually continuous.
        # Check action dtype from data
        if data['actions'].dtype.kind == 'f':
             self.actions = np.empty((total_transitions, action_dim), dtype=np.float32)

        self.rewards = np.empty((total_transitions, 1), dtype=np.float32)
        self.next_states = np.empty((total_transitions, state_dim), dtype=np.float32)
        self.dones = np.empty((total_transitions, 1), dtype=np.float32)
        
        current_idx = 0
        has_rewards_key = 'rewards' in data.keys()

        for i in range(data['states'].shape[0]):
            mask = data['mask'][i]
            clip_len = int(mask.sum())
            
            if clip_len == 0:
                continue

            traj_states = data['states'][i, :clip_len]
            traj_rtg = data['returns_to_go'][i, :clip_len]
            
            # Actions
            traj_actions = data['actions'][i, :clip_len]
            # If dataset has padding or specific structure, handle here. 
            # D4RL conversions are already clipped.
            # Usually transition dataset needs s, a, r, s', d
            # For a clip of length T, we have T-1 transitions usually if we rely on next state in clip
            # But if the clip is a segment of a trajectory, s[t+1] is next state.
            
            # We can use T transitions if we have s[T] (next state for last step)?
            # Usually we only have states up to T.
            # If done=True at T, then next_state is terminal (often same as s[T] or 0s).
            # If done=False, s[T] is just the state at T.
            
            # Current implementation logic:
            # rewards[:-1], next_states[:-1] = traj_states[1:]
            
            # Let's stick to the existing logic which extracts transitions FROM the sequence.
            # This means from T steps, we get T-1 transitions? Or T if we handle the last one?
            # The original code:
            # rewards = np.zeros((clip_len, 1))
            # next_states = np.zeros_like(traj_states)
            # dones = np.zeros(...)
            # rewards[:-1] = ...
            # next_states[:-1] = traj_states[1:]
            # rewards[-1] = traj_rtg[-1] -- This is a hack for the last step?
            # dones[-1] = 1.0 -- forcing done at end of clip? That seems wrong for non-terminal clips.
            
            # Improvement: Use 'dones' from data if available.
            
            traj_rewards = np.zeros((clip_len, 1), dtype=np.float32)
            traj_next_states = np.zeros_like(traj_states)
            traj_dones = np.zeros((clip_len, 1), dtype=np.float32)
            
            # Next states
            if clip_len > 1:
                traj_next_states[:-1] = traj_states[1:]
            # For the last step, we don't have next state in this clip unless we peek?
            # But we don't have access to next clip.
            # We will pad next_state of last step with itself or zeros, and trust 'dones'.
            traj_next_states[-1] = traj_states[-1] # fallback
            
            # Rewards
            if has_rewards_key:
                 traj_rewards = data['rewards'][i, :clip_len].reshape(-1, 1)
            else:
                if clip_len > 1:
                    traj_rewards[:-1] = (traj_rtg[:-1] - traj_rtg[1:]).reshape(-1, 1)
                traj_rewards[-1] = traj_rtg[-1] # Fallback
            
            # Dones
            if 'dones' in data.keys():
                traj_dones = data['dones'][i, :clip_len].reshape(-1, 1)
            else:
                traj_dones[-1] = 1.0 # Original fallback
            
            self.states[current_idx:current_idx+clip_len] = traj_states
            self.actions[current_idx:current_idx+clip_len] = traj_actions
            self.rewards[current_idx:current_idx+clip_len] = traj_rewards
            self.next_states[current_idx:current_idx+clip_len] = traj_next_states
            self.dones[current_idx:current_idx+clip_len] = traj_dones
            
            current_idx += clip_len
        
        self.states = torch.from_numpy(self.states).float()
        self.actions = torch.from_numpy(self.actions).float()
        self.rewards = torch.from_numpy(self.rewards).float()
        self.next_states = torch.from_numpy(self.next_states).float()
        self.dones = torch.from_numpy(self.dones).float()

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return {
            "states": self.states[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_states": self.next_states[idx],
            "dones": self.dones[idx],
        }


def train(cfg, logger):
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass # For older torch versions
    seed_everything(cfg.seed)

    if cfg.model.name in ["snn_dt", "dsformer"]:
        cfg.training.batches_per_epoch = min(cfg.training.batches_per_epoch, 100)
    
    # Create save directory
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"--- Checkpoint: Save directory created at {save_dir} ---")

    # Load data and metadata
    state_mean = None
    state_std = None

    if cfg.dataset.mode == 'd4rl_direct':
        if os.path.isfile(cfg.dataset.path):
            data_dir = os.path.dirname(cfg.dataset.path)
        else:
            data_dir = cfg.dataset.path

        if cfg.model.name in ['iql', 'cql']:
            dataset = D4RLTransitionDataset(cfg.env, data_dir=data_dir)
        else:
            dataset = D4RLSequenceDataset(cfg.env, data_dir=data_dir, seq_len=cfg.model.seq_len)
            
        state_mean = dataset.state_mean
        state_std = dataset.state_std
        
        cfg.dataset.state_dim = dataset.states.shape[1] if hasattr(dataset, 'states') else dataset.state_dim
        cfg.dataset.act_dim = dataset.act_dim if hasattr(dataset, 'act_dim') else dataset.actions.shape[1]
        cfg.dataset.max_timesteps = 1000 # Standard MuJoCo
        
    else:
        # Legacy
        if cfg.model.name in ['iql', 'cql']:
            dataset = OfflineTransitionDataset(cfg.dataset.path)
        else:
            dataset = OfflineDataset(cfg.dataset.path)
            
        with np.load(cfg.dataset.path, allow_pickle=True) as data:
            metadata = data["metadata"].item()
            if isinstance(metadata, str):
                metadata = yaml.safe_load(metadata)
        
        cfg.dataset.state_dim = metadata["state_dim"]
        cfg.dataset.act_dim = metadata["act_dim"]
        cfg.dataset.max_timesteps = metadata["max_timesteps"]

    if len(dataset) == 0:
        logger.error(f"Dataset at {cfg.dataset.path} is empty! Aborting training.")
        sys.exit(1)
    logger.info(f"Dataset size: {len(dataset)} items")
    
    # Lazily import gymnasium to avoid potential C-extension conflicts at startup
    import gymnasium as gym
    try:
        # Try to make the env to check discrete/continuous, but handle if simulator missing
        temp_env = gym.make(cfg.env)
        cfg.dataset.is_discrete = isinstance(temp_env.action_space, gym.spaces.Discrete)
        temp_env.close()
    except Exception:
        # Fallback if simulator not available: assume continuous (common for D4RL/MuJoCo)
        # Or check dataset action shape?
        logger.warning("Could not create env to check action space. Assuming Continuous (False).")
        cfg.dataset.is_discrete = False

    from torch.utils.data import DataLoader
    num_workers = cfg.training.num_workers
    pin_memory = cfg.training.pin_memory
    persistent_workers = cfg.training.persistent_workers

    train_loader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    logger.info(f"DataLoader created with num_workers={num_workers}, pin_memory={pin_memory}, persistent_workers={persistent_workers}.")

    # Initialize model and optimizer
    model = get_model(cfg).to(cfg.training.device)
    logger.info(f"--- Checkpoint: Model '{cfg.model.name}' initialized on device '{cfg.training.device}' ---")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay),
    )
    
    if cfg.dataset.is_discrete:
        loss_fn = torch.nn.CrossEntropyLoss()
    else:
        loss_fn = torch.nn.MSELoss()

    # Training loop
    metrics = []
    best_eval_return = -np.inf
    
    # Lazily initialize the environment
    env = None

    logger.info("--- Checkpoint: Starting main training loop ---")
    for epoch in range(cfg.training.epochs):
        start_time = time.time()
        epoch_losses = []

        if hasattr(model, "reset_spike_counts"):
            model.reset_spike_counts()
        
        train_iter = iter(train_loader)
        pbar = tqdm(range(cfg.training.batches_per_epoch), desc=f"Epoch {epoch+1}/{cfg.training.epochs}")

        for i in pbar:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            
            model.train()

            for k, v in batch.items():
                batch[k] = v.to(cfg.training.device)
            
            if cfg.model.name in ['iql', 'cql']:
                losses = model.learn(batch)
                epoch_losses.append(losses['value_loss'])
                pbar.set_postfix(loss=f"{np.mean(epoch_losses):.4f}")
            else:
                optimizer.zero_grad()
                action_preds = model(batch)
                action_targets = batch["actions"]

                # Align sequence lengths and filter padded tokens
                action_preds = action_preds[:, :action_targets.shape[1]]
                mask = batch["mask"][:, :action_targets.shape[1]].reshape(-1).bool()
                
                action_preds = action_preds.reshape(-1, cfg.dataset.act_dim)[mask]
                
                if cfg.dataset.is_discrete:
                    action_targets = action_targets.reshape(-1)[mask].long()
                else:
                    action_targets = action_targets.reshape(-1, cfg.dataset.act_dim)[mask]

                loss = loss_fn(action_preds, action_targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_losses.append(loss.item())
                if i % 10 == 0:
                    pbar.set_postfix(loss=f"{np.mean(epoch_losses):.4f}")

        # Evaluation, Checkpointing, and Logging
        if (epoch + 1) % cfg.training.eval_every == 0:
            if env is None:
                env = create_env(cfg.env, simulator_available=cfg.training.simulator_available, dataset_path=cfg.dataset.path)

            eval_results = evaluate_policy(
                model, 
                env, 
                cfg, 
                episodes=cfg.hyperparameters.eval_episodes,
                state_mean=state_mean,
                state_std=state_std
            )
            epoch_time = time.time() - start_time
            avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
            
            # Simplified Log String
            log_items = [f"Epoch {epoch+1}/{cfg.training.epochs}"]
            log_items.append(f"Loss: {avg_loss:.4f}")
            log_items.append(f"Return: {eval_results['return_mean']:.2f}")
            
            # Spike counting for SNN models
            if hasattr(model, "count_spikes"):
                spikes = model.count_spikes()
                log_items.append(f"Spikes: {spikes:.4f}")
                eval_results["spikes"] = spikes
            else:
                eval_results["spikes"] = 0.0
            
            if hasattr(model, "get_max_attn_score"):
                max_attn = model.get_max_attn_score()
                eval_results["max_attn"] = max_attn

            metrics.append({"epoch": epoch + 1, "loss": avg_loss, **eval_results, "time_s": epoch_time})
            
            logger.info(" | ".join(log_items))
            
            if eval_results['return_mean'] > best_eval_return:
                best_eval_return = eval_results['return_mean']
                torch.save(model.state_dict(), save_dir / "best.pt")
                logger.info(f"New best eval return: {best_eval_return:.2f}. Saved best model.")
            
            # Apply plasticity for SNN models
            if isinstance(model, SnnDt) and model.use_plasticity:
                model.apply_plasticity(eval_results["return_mean"])

        # Periodic checkpointing
        if (epoch + 1) % cfg.training.checkpoint_every == 0:
            torch.save(model.state_dict(), save_dir / f"ckpt_epoch_{epoch+1}.pt")

    # Save metrics
    df = pd.DataFrame(metrics)
    df.to_csv(save_dir / "metrics.csv", index=False)
    
    # Update summary
    if not df.empty and "return_mean" in df.columns:
        summary_path = Path(cfg.save_dir).parent.parent / "summary.csv"
        summary_df = pd.DataFrame([{"model": cfg.model.name, "env": cfg.env, "seed": cfg.seed, "return_mean": df["return_mean"].max()}])
        if summary_path.exists():
            summary_df.to_csv(summary_path, mode="a", header=False, index=False)
        else:
            summary_df.to_csv(summary_path, index=False)
    else:
        logger.warning("No evaluation metrics found. Skipping summary generation.")
        
    # Manifest Logging
    try:
        manifest_path = Path("runs/manifest.csv")
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_entry = {
            "timestamp": datetime.now().isoformat(),
            "config_path": cfg.get("config_path", "N/A"),
            "dataset_path": cfg.dataset.path,
            "git_commit": "N/A", # Could fetch via subprocess if needed
            "checkpoint_path": str(save_dir / "best.pt"),
            "status": "success"
        }
        
        file_exists = manifest_path.exists()
        with open(manifest_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=manifest_entry.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(manifest_entry)
        logger.info(f"Run logged to {manifest_path}")
    except Exception as e:
        logger.error(f"Failed to write to manifest: {e}")

    logger.info("Training complete.")
        

def main():
    # --- Robust Crash Logging ---
    import faulthandler
    faulthandler.enable()
    
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logging.getLogger().critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to config file.")
    parser.add_argument("--model", type=str, required=True, help="Name of the model to train (dt, snn_dt, dsformer, iql, cql).")
    parser.add_argument("--env", type=str, required=True, help="Environment name (e.g., CartPole-v1).")
    parser.add_argument("--save-dir", type=str, default="results/run", help="Directory to save results.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--dataset-path", type=str, default=None, help="Explicit path to dataset file or directory.")
    parser.add_argument("--simulator-available", action="store_true", help="Set if a real simulator is available for eval.")
    parser.add_argument("--dataset-mode", type=str, default="d4rl_direct", help="Dataset mode: 'legacy' (npz) or 'd4rl_direct' (hdf5).")
    args = parser.parse_args()
    
    # Configure logging
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(save_dir / "training.log"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger()

    # Determine config file path
    if args.config is None:
        model_abbr = {"dt": "dt", "snn_dt": "sdt", "dsformer": "dsf", "iql": "iql", "cql": "cql"}
        # Simple heuristic mapping, user can override with --config
        # We don't have default configs for D4RL envs yet, but we will create light ones.
        # Fallback to existing logic
        env_abbr = {"CartPole-v1": "cartpole", "Acrobot-v1": "acrobot", "MountainCar-v0": "mountaincar", "Pendulum-v1": "pendulum"}
        
        # Mapping for D4RL
        if "hopper" in args.env: env_abbr[args.env] = "hopper"
        if "walker" in args.env: env_abbr[args.env] = "walker"
        if "cheetah" in args.env: env_abbr[args.env] = "halfcheetah"
        
        if args.model in model_abbr and args.env in env_abbr:
            config_name = f"{model_abbr[args.model]}_{env_abbr[args.env]}.yaml"
            args.config = str(snn_dt_root / "configs" / config_name)
        else:
             # Just try a generic name if above fails
            config_name = f"{args.model}_{args.env}.yaml"
            if (snn_dt_root / "configs" / config_name).exists():
                 args.config = str(snn_dt_root / "configs" / config_name)
            else:
                 # Last resort: use a default?
                 pass

    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            cfg_raw = yaml.safe_load(f)
    else:
        logger.warning(f"Config file {args.config} not found. Using minimal defaults.")
        cfg_raw = {}
    
    # Create structured config with defaults
    cfg = {
        "config_path": args.config,
        "model": {
            "name": args.model,
            "seq_len": cfg_raw.get("seq_len", 20),
            "d_model": cfg_raw.get("hidden_dim", 128),
            "n_heads": cfg_raw.get("n_heads", 4),
            "n_layers": cfg_raw.get("n_layers", 4),
            "action_tanh": False,
        },
        "training": {
            "batch_size": cfg_raw.get("batch_size", 64),
            "lr": cfg_raw.get("learning_rate", 1e-4),
            "weight_decay": cfg_raw.get("weight_decay", 0.0),
            "epochs": cfg_raw.get("epochs", 1000),
            "eval_every": cfg_raw.get("eval_every", 10),
            "checkpoint_every": cfg_raw.get("checkpoint_every", 50),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "batches_per_epoch": cfg_raw.get("batches_per_epoch", 1000),
            "num_workers": cfg_raw.get("num_workers", 0),
            "pin_memory": cfg_raw.get("pin_memory", False),
            "persistent_workers": cfg_raw.get("persistent_workers", False),
            "simulator_available": args.simulator_available,
        },
        "dataset": {
            "path": cfg_raw.get("dataset", {}).get("path") if isinstance(cfg_raw.get("dataset"), dict) else cfg_raw.get("dataset", None),
            "state_dim": None,  # Will be set from metadata
            "act_dim": None,    # Will be set from metadata
            "max_timesteps": None  # Will be set from metadata
        },
        "hyperparameters": {
             "eval_episodes": cfg_raw.get("eval_episodes", 10)
        },
        "env": args.env,
        "seed": args.seed,
        "save_dir": args.save_dir,
        "snn": {
            "lif_tau": cfg_raw.get("snn", {}).get("lif_tau", 20.0),
            "surrogate_k": cfg_raw.get("snn", {}).get("surrogate_k", 25.0),
            "v_th": cfg_raw.get("snn", {}).get("v_th", 1.0),
            "current_scale": cfg_raw.get("snn", {}).get("current_scale", 0.2),
            "use_plasticity": cfg_raw.get("snn", {}).get("use_plasticity", False)
        },
        "iql": {
            "tau": cfg_raw.get("tau", 0.005),
            "temperature": cfg_raw.get("temperature", 3.0),
            "expectile": cfg_raw.get("expectile", 0.7),
            "hidden_size": cfg_raw.get("hidden_size", 256)
        },
        "cql": {
            "tau": cfg_raw.get("tau", 0.005),
            "temperature": cfg_raw.get("temperature", 1.0),
            "hidden_size": cfg_raw.get("hidden_size", 256),
            "with_lagrange": cfg_raw.get("with_lagrange", False),
            "cql_weight": cfg_raw.get("cql_weight", 1.0),
            "target_action_gap": cfg_raw.get("target_action_gap", 10.0)
        }
    }
    
    # Convert to AttrDict for easy access
    cfg = AttrDict(cfg)

    if cfg.model.name == "snn_dt":
        logger.info(f"SNN Config: {cfg.snn}")

    # Dataset path priority: Args > Config > Default
    # Config for dataset mode
    cfg.dataset.mode = args.dataset_mode
    
    if args.dataset_path:
        cfg.dataset.path = args.dataset_path
    elif cfg.dataset.path is None:
        if args.dataset_mode == 'd4rl_direct':
             cfg.dataset.path = str(snn_dt_root / "data/d4rl_raw")
        else:
             cfg.dataset.path = str(snn_dt_root / f"data/{args.env}/dataset.npz")
    
    # Check if dataset exists (folder or file)
    if not os.path.exists(cfg.dataset.path):
         logger.warning(f"Dataset not found at {cfg.dataset.path}. Training will likely fail if data isn't generated.")
    else:
        logger.info(f"Using dataset: {cfg.dataset.path}")

    logger.info("--- Checkpoint: Starting training ---")
    try:
        train(cfg, logger)
    except Exception as e:
        logger.exception("Exception during training:")
        raise e


if __name__ == "__main__":
    main()