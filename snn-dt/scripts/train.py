import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Add project root and snn-dt directory to sys.path
snn_dt_root = Path(__file__).resolve().parent.parent
project_root = snn_dt_root.parent
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

# from src.models.cql import CQL
# from src.models.dt import DecisionTransformer
# from src.models.dsformer import DsFormer
# from src.models.iql import IQL
from src.models.snn_dt import SnnDt
from src.utils.config import AttrDict
from src.utils.models import get_model
from src.utils.seed import seed_everything
from scripts.eval import evaluate_policy


class OfflineDataset(Dataset):
    def __init__(self, dataset_path):
        data = np.load(dataset_path)
        self.states = torch.from_numpy(data["states"]).float()
        self.actions = torch.from_numpy(data["actions"]).float()
        self.returns_to_go = torch.from_numpy(data["returns_to_go"]).float()
        self.timesteps = torch.from_numpy(data["timesteps"]).long()
        self.mask = torch.from_numpy(data["mask"]).float()

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return {
            "states": self.states[idx],
            "actions": self.actions[idx],
            "returns_to_go": self.returns_to_go[idx],
            "timesteps": self.timesteps[idx],
            "mask": self.mask[idx],
        }


class OfflineTransitionDataset(Dataset):
    def __init__(self, dataset_path):
        data = np.load(dataset_path)
        
        states, actions, rewards, next_states, dones = [], [], [], [], []
        
        for i in range(data["states"].shape[0]):
            mask = data["mask"][i]
            clip_len = int(mask.sum())
            
            for t in range(clip_len - 1):
                states.append(data["states"][i, t])
                actions.append(data["actions"][i, t])
                rewards.append(data["returns_to_go"][i, t] - data["returns_to_go"][i, t+1])
                next_states.append(data["states"][i, t+1])
                dones.append(0.0)
            
            # Add final transition
            states.append(data["states"][i, clip_len-1])
            actions.append(data["actions"][i, clip_len-1])
            rewards.append(data["returns_to_go"][i, clip_len-1])
            next_states.append(np.zeros_like(data["states"][i, clip_len-1])) # Placeholder
            dones.append(1.0)

        self.states = torch.from_numpy(np.array(states)).float()
        self.actions = torch.from_numpy(np.array(actions)).float()
        self.rewards = torch.from_numpy(np.array(rewards)).float().unsqueeze(-1)
        self.next_states = torch.from_numpy(np.array(next_states)).float()
        self.dones = torch.from_numpy(np.array(dones)).float().unsqueeze(-1)

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
    seed_everything(cfg.seed)
    
    # Create save directory
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load data and metadata
    dataset = OfflineDataset(cfg.dataset.path)
    if len(dataset) == 0:
        logger.error(f"Dataset at {cfg.dataset.path} is empty! Aborting training.")
        sys.exit(1)
    logger.info(f"Dataset size: {len(dataset)} clips")

    with np.load(cfg.dataset.path, allow_pickle=True) as data:
        metadata = data["metadata"].item()
        if isinstance(metadata, str):
            metadata = yaml.safe_load(metadata)
    
    # Update config with dataset metadata
    cfg.dataset.state_dim = metadata["state_dim"]
    cfg.dataset.act_dim = metadata["act_dim"]
    cfg.dataset.max_timesteps = metadata["max_timesteps"]
    
    import gymnasium as gym
    temp_env = gym.make(cfg.env)
    cfg.dataset.is_discrete = isinstance(temp_env.action_space, gym.spaces.Discrete)
    temp_env.close()

    # OS-aware num_workers
    num_workers = 0
    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    logger.info(f"DataLoader created with num_workers={num_workers} and pin_memory=False.")

    # Initialize model and optimizer
    model = get_model(cfg).to(cfg.training.device)
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

    logger.info("Starting training loop...")
    for epoch in range(cfg.training.epochs):
        start_time = time.time()
        epoch_losses = []
        
        batch_iter = tqdm(
            enumerate(train_loader), 
            total=len(train_loader),
            desc=f"Epoch {epoch+1}/{cfg.training.epochs}"
        )

        for i, batch in batch_iter:
            model.train()

            for k, v in batch.items():
                batch[k] = v.to(cfg.training.device)
            
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
            batch_iter.set_postfix(loss=f"{np.mean(epoch_losses):.4f}")

        # Evaluation, Checkpointing, and Logging
        if (epoch + 1) % cfg.training.eval_every == 0:
            if env is None:
                import gymnasium as gym
                env = gym.make(cfg.env)
            eval_results = evaluate_policy(model, env, cfg, episodes=10)
            epoch_time = time.time() - start_time
            avg_loss = np.mean(epoch_losses)
            
            log_str = f"Epoch {epoch+1}/{cfg.training.epochs} | Time: {epoch_time:.2f}s | Loss: {avg_loss:.4f}"
            
            # Spike counting for SNN models
            if isinstance(model, SnnDt):
                spikes = model.count_spikes()
                log_str += f" | Spikes: {spikes}"
                eval_results["spikes"] = spikes

            metrics.append({"epoch": epoch + 1, "loss": avg_loss, **eval_results, "time_s": epoch_time})
            log_str += f" | Eval Return: {eval_results['return_mean']:.2f}"
            logger.info(log_str)
            
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
    summary_path = Path(cfg.save_dir).parent.parent / "summary.csv"
    summary_df = pd.DataFrame([{"model": cfg.model.name, "env": cfg.env, "seed": cfg.seed, "return_mean": df["return_mean"].max()}])
    if summary_path.exists():
        summary_df.to_csv(summary_path, mode="a", header=False, index=False)
    else:
        summary_df.to_csv(summary_path, index=False)

    logger.info("Training complete.")
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to config file.")
    parser.add_argument("--model", type=str, required=True, help="Name of the model to train (dt, snn_dt, dsformer, iql, cql).")
    parser.add_argument("--env", type=str, required=True, help="Environment name (e.g., CartPole-v1).")
    parser.add_argument("--save-dir", type=str, default="results/run", help="Directory to save results.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
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
        env_abbr = {"CartPole-v1": "cartpole", "Acrobot-v1": "acrobot", "MountainCar-v0": "mountaincar", "Pendulum-v1": "pendulum"}
        
        if args.model in model_abbr and args.env in env_abbr:
            config_name = f"{model_abbr[args.model]}_{env_abbr[args.env]}.yaml"
            args.config = str(project_root / "configs" / config_name)
        else:
            raise ValueError(f"Could not automatically determine config for model '{args.model}' and env '{args.env}'. Please specify with --config.")

    # Load config
    with open(args.config, "r") as f:
        cfg_raw = yaml.safe_load(f)
    
    # Create structured config with defaults
    cfg = {
        "model": {
            "name": args.model,
            "seq_len": cfg_raw.get("seq_len"),
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
        },
        "dataset": {
            "path": cfg_raw.get("dataset", None),
            "state_dim": None,  # Will be set from metadata
            "act_dim": None,    # Will be set from metadata
            "max_timesteps": None  # Will be set from metadata
        },
        "env": args.env,
        "seed": args.seed,
        "save_dir": args.save_dir,
        "snn": {
            "lif_tau": cfg_raw.get("lif_tau", 20.0),
            "surrogate_k": cfg_raw.get("surrogate_k", 25.0),
            "use_plasticity": cfg_raw.get("use_plasticity", False)
        }
    }
    
    # Convert to AttrDict for easy access
    cfg = AttrDict(cfg)
    
    # Construct dataset path from env name, relative to project root
    cfg.dataset.path = str(project_root / f"data/{args.env}/dataset.npz")
    
    # Create data directory if it doesn't exist
    data_dir = project_root / "data" / args.env
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate dataset if it doesn't exist
    logger.info("Checking for dataset...")
    if not os.path.exists(cfg.dataset.path):
        from scripts.generate_dataset import generate_random_trajectories, process_trajectories
        logger.info(f"Dataset not found at {cfg.dataset.path}. Generating new dataset...")
        trajectories = generate_random_trajectories(args.env, num_trajectories=1000)
        dataset = process_trajectories(trajectories, env_name=args.env)
        np.savez_compressed(cfg.dataset.path, **dataset)
        logger.info(f"Dataset generated and saved to {cfg.dataset.path}")
    else:
        logger.info(f"Dataset found at {cfg.dataset.path}.")

    logger.info("Starting training...")
    train(cfg, logger)


if __name__ == "__main__":
    main()