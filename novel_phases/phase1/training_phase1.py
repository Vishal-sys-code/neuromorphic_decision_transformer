import argparse
import os
import sys
import pickle
import random
import numpy as np
import torch
import torch.nn as nn # Added for loss functions
import torch.optim as optim # Added for optimizer
from torch.utils.data import Dataset, DataLoader # Added for clarity
import gym

# It's good practice to add the project root to the Python path
# for consistent module resolution.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from phase1_integration_with_sdt import (
    SpikingDecisionTransformer,
    compute_spiking_loss,
    get_default_config
)

# Basic constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_SEED = 42
DEFAULT_ENV_NAME = "CartPole-v1"
DEFAULT_OFFLINE_STEPS = 10000
DEFAULT_MAX_EPISODE_LENGTH = 200 # Max steps per episode during data collection
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 1e-4
DEFAULT_GAMMA = 0.99
DEFAULT_MAX_LENGTH = 20 # Max sequence length for DT input (context window)
DEFAULT_CHECKPOINT_DIR = "checkpoints_phase1"


ENV_MAP = {
    "1": "CartPole-v1",
    "2": "MountainCar-v0",
    # "3": "LunarLander-v2", # Requires Box2D
    "4": "Acrobot-v1",
    "5": "Pendulum-v1",
}

def get_args():
    parser = argparse.ArgumentParser(description="Train Spiking Decision Transformer (Phase 1 Integration)")

    parser.add_argument("--env_name", type=str, default=None,
                        help=f"Name of the Gym environment (e.g., CartPole-v1). Default: {DEFAULT_ENV_NAME}")
    parser.add_argument("--offline_steps", type=int, default=DEFAULT_OFFLINE_STEPS,
                        help=f"Number of steps to collect for the offline dataset. Default: {DEFAULT_OFFLINE_STEPS}")
    parser.add_argument("--max_episode_length", type=int, default=DEFAULT_MAX_EPISODE_LENGTH,
                        help=f"Maximum steps per episode during data collection. Default: {DEFAULT_MAX_EPISODE_LENGTH}")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help=f"Number of training epochs. Default: {DEFAULT_EPOCHS}")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Batch size for training. Default: {DEFAULT_BATCH_SIZE}")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR,
                        help=f"Learning rate. Default: {DEFAULT_LR}")
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA,
                        help=f"Discount factor for returns-to-go. Default: {DEFAULT_GAMMA}")
    parser.add_argument("--max_length", type=int, default=DEFAULT_MAX_LENGTH,
                        help=f"Maximum context length for the transformer (K). Default: {DEFAULT_MAX_LENGTH}")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Random seed. Default: {DEFAULT_SEED}")
    parser.add_argument("--checkpoint_dir", type=str, default=DEFAULT_CHECKPOINT_DIR,
                        help=f"Directory to save model checkpoints. Default: {DEFAULT_CHECKPOINT_DIR}")
    parser.add_argument("--log_interval", type=int, default=1, help="Epoch interval for logging.")

    # Model specific arguments from get_default_config()
    model_config = get_default_config()
    parser.add_argument("--embedding_dim", type=int, default=model_config['embedding_dim'])
    parser.add_argument("--num_layers", type=int, default=model_config['num_layers'])
    parser.add_argument("--num_heads", type=int, default=model_config['num_heads'])
    parser.add_argument("--T_max", type=int, default=model_config['T_max'],
                        help="Maximum time window for adaptive spiking attention.")
    # max_episode_length is already an arg for data collection, use it for model config too
    # parser.add_argument("--max_episode_len_model", type=int, default=model_config['max_episode_len'],
    # help="Maximum episode length for timestep embeddings in the model.") # Renamed to avoid clash if different needed
    parser.add_argument("--dropout", type=float, default=model_config['dropout'])
    parser.add_argument("--lambda_reg", type=float, default=model_config['lambda_reg'],
                        help="Regularization weight for adaptive window mechanism.")
    
    args = parser.parse_args()

    # Handle interactive environment selection if env_name is not provided
    if args.env_name is None:
        print("\nSelect environment:")
        for k, v in ENV_MAP.items():
            print(f"  {k}. {v}")
        while True:
            sel = input(f"Enter choice (1-{len(ENV_MAP)}) or environment name directly: ").strip()
            if sel in ENV_MAP:
                args.env_name = ENV_MAP[sel]
                break
            elif sel in ENV_MAP.values(): # Allow direct name input
                args.env_name = sel
                break
            else:
                try: # check if user typed a valid gym env name
                    gym.spec(sel) # test if env exists
                    args.env_name = sel
                    print(f"Using environment: {args.env_name}")
                    break
                except gym.error.NameNotFound: # Use gym.error.Error for broader compatibility if needed
                    print("Invalid selection or environment name. Please try again.")
    
    return args

# Helper functions
def set_seed(seed_val: int):
    """Sets random seeds for reproducibility."""
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(seed_val)
    print(f"Set seed to {seed_val}")

def simple_logger(log_dict: dict, epoch: int, args):
    """Simple console logger."""
    if epoch % args.log_interval == 0:
        log_string = f"Epoch: {epoch}"
        for k, v in log_dict.items():
            if isinstance(v, float):
                log_string += f" | {k}: {v:.4f}"
            else:
                log_string += f" | {k}: {v}"
        print(log_string)

# Data collection and preparation
def compute_returns_to_go(rewards: list, gamma: float) -> np.ndarray:
    """
    Computes the returns-to-go for a sequence of rewards.
    rtg_i = sum_{j=i}^{T} gamma^(j-i) * r_j
    """
    n = len(rewards)
    rtgs = np.zeros_like(rewards, dtype=np.float32)
    current_rtg = 0.0
    for i in reversed(range(n)):
        current_rtg = rewards[i] + gamma * current_rtg
        rtgs[i] = current_rtg
    return rtgs

class TrajectoryBuffer:
    """
    A simple buffer to store parts of a trajectory before it's complete.
    """
    def __init__(self, max_len: int, state_dim: int, act_dim: int, act_type: str):
        self.max_len = max_len
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.act_type = act_type
        self.reset()

    def add(self, state, action, reward):
        if len(self.states) < self.max_len:
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)

    def get_trajectory(self) -> dict:
        # Convert lists to numpy arrays before returning
        return {
            "states": np.array(self.states, dtype=np.float32),
            "actions": np.array(self.actions, dtype=np.int64 if self.act_type == "discrete" else np.float32),
            "rewards": np.array(self.rewards, dtype=np.float32),
        }

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
    
    def __len__(self):
        return len(self.rewards)

def collect_trajectories(env_name: str, state_dim: int, act_dim: int, act_type: str,
                         offline_steps: int, max_episode_len: int, gamma: float, seed: int):
    """Collects offline_steps env steps with a random policy."""
    env = gym.make(env_name)
    # It's good practice to seed the env for reproducibility if it supports it, though random policy diminishes this
    # env.seed(seed) # Deprecated, use env.reset(seed=seed)
    
    trajectories = []
    buf = TrajectoryBuffer(max_episode_len, state_dim, act_dim, act_type)
    
    print(f"Collecting {offline_steps} steps from {env_name} using a random policy...")
    
    current_steps = 0
    obs, _ = env.reset(seed=seed) # Seed on reset
    
    while current_steps < offline_steps:
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        buf.add(obs.astype(np.float32), action, reward)
        
        obs = next_obs
        current_steps += 1
        
        if done or len(buf) == max_episode_len:
            trajectories.append(buf.get_trajectory())
            buf.reset()
            obs, _ = env.reset() # No need to re-seed here for subsequent episodes with random policy
            if current_steps % (offline_steps // 10) == 0 and offline_steps > 0: # Log progress
                 print(f"  Collected {current_steps}/{offline_steps} steps...")
    
    env.close()
    print(f"Finished collecting {len(trajectories)} trajectories ({current_steps} total steps).")
    return trajectories


class TrajectoryDataset(torch.utils.data.Dataset):
    def __init__(self, trajectories: list, args, act_type: str):
        self.sequences = []
        self.act_type = act_type
        self.max_length = args.max_length # DT context window K
        self.gamma = args.gamma

        for traj in trajectories:
            states  = traj["states"]      # [EpisodeLen, S]
            actions = traj["actions"]     # [EpisodeLen,] or [EpisodeLen, Adim]
            rewards = traj["rewards"]     # [EpisodeLen,]
            
            if len(states) == 0: continue # Skip empty trajectories

            returns_to_go = compute_returns_to_go(rewards, self.gamma).reshape(-1, 1)
            timesteps = np.arange(len(states)).reshape(-1, 1)
            
            episode_len = len(states)

            # Create subsequences of length at most self.max_length
            for i in range(episode_len):
                start_idx = max(0, i - self.max_length + 1)
                end_idx = i + 1
                
                self.sequences.append({
                    "states":    states[start_idx:end_idx],
                    "actions":   actions[start_idx:end_idx],
                    "returns_to_go": returns_to_go[start_idx:end_idx],
                    "timesteps": timesteps[start_idx:end_idx],
                })

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        seq = self.sequences[idx]
        current_seq_len = len(seq["states"])
        pad_len = self.max_length - current_seq_len

        # Pad states: [K, S]
        state_dim = seq["states"].shape[1]
        padded_states = np.concatenate([
            np.zeros((pad_len, state_dim)),
            seq["states"]
        ], axis=0).astype(np.float32)

        # Pad actions: [K, Adim] or [K, 1] for discrete
        if self.act_type == "discrete":
            # Discrete actions are usually longs/ints, padding with 0 (or a specific pad token if available)
            # For DT, actions are often one-hot encoded later or used as indices.
            # Here, we'll keep them as longs and pad.
            padded_actions = np.concatenate([
                np.zeros(pad_len), # Assuming 0 is a safe padding for discrete actions not used in loss
                seq["actions"]
            ], axis=0).astype(np.int64).reshape(-1, 1)
        else: # Continuous
            act_dim = seq["actions"].shape[1]
            padded_actions = np.concatenate([
                np.zeros((pad_len, act_dim)),
                seq["actions"]
            ], axis=0).astype(np.float32)

        # Pad returns_to_go: [K, 1]
        padded_rtg = np.concatenate([
            np.zeros((pad_len, 1)),
            seq["returns_to_go"]
        ], axis=0).astype(np.float32)

        # Pad timesteps: [K, 1] -> then squeeze to [K] for nn.Embedding
        padded_timesteps = np.concatenate([
            np.zeros((pad_len, 1)), # Pad with 0, assuming timesteps are non-negative
            seq["timesteps"]
        ], axis=0).astype(np.int64)

        return {
            "states": torch.from_numpy(padded_states).to(DEVICE),
            "actions": torch.from_numpy(padded_actions).to(DEVICE),
            "returns_to_go": torch.from_numpy(padded_rtg).to(DEVICE),
            "timesteps": torch.from_numpy(padded_timesteps.squeeze(-1)).to(DEVICE), # Squeeze last dim
            "mask": torch.cat([torch.zeros(pad_len), torch.ones(current_seq_len)]).to(DEVICE) # For attention masking
        }


if __name__ == '__main__':
    # Patch numpy for Gym's checker if needed (often for older gym versions)
    if not hasattr(np, "bool8"): # Or np.bool_ based on common practice
        np.bool8 = np.bool_
    if not hasattr(np, "float_"):
        np.float_ = np.float64

    os.environ["GYM_DISABLE_ENV_CHECKER"] = "true"


    args = get_args()
    set_seed(args.seed) # Set seed after parsing args

    print("Selected Configuration:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    
    print(f"\nDevice: {DEVICE}")
    # Ensure checkpoint directory exists
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # --- Initialize Environment ---
    env = gym.make(args.env_name)
    state_dim = env.observation_space.shape[0]
    if hasattr(env.action_space, "n"):
        act_dim = env.action_space.n
        act_type = "discrete"
        loss_fn_action = torch.nn.CrossEntropyLoss()
    else:
        act_dim = env.action_space.shape[0]
        act_type = "continuous"
        loss_fn_action = torch.nn.MSELoss()
    
    print(f"Environment: {args.env_name}, State Dim: {state_dim}, Action Dim: {act_dim}, Action Type: {act_type}")

    # --- Collect Trajectories ---
    trajectories = collect_trajectories(
        env_name=args.env_name,
        state_dim=state_dim,
        act_dim=act_dim,
        act_type=act_type,
        offline_steps=args.offline_steps,
        max_episode_len=args.max_episode_length,
        gamma=args.gamma,
        seed=args.seed
    )
    
    # --- Create Dataset and DataLoader ---
    dataset = TrajectoryDataset(trajectories, args, act_type)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True # Important for fixed sequence length models
    )
    print(f"Created dataset with {len(dataset)} subsequences. DataLoader ready.")

    # --- Initialize Model ---
    model_config_args = {
        'state_dim': state_dim,
        'action_dim': act_dim, # For discrete, this is num_classes; for continuous, feature dim
        'embedding_dim': args.embedding_dim,
        'num_layers': args.num_layers,
        'num_heads': args.num_heads,
        'T_max': args.T_max,
        'max_length': args.max_length, # This is DT's K (context window)
        'max_episode_len': args.max_episode_length, # Pass the existing arg for data collection
        'dropout': args.dropout,
    }
    model = SpikingDecisionTransformer(**model_config_args).to(DEVICE)
    print(f"Spiking Decision Transformer model initialized with T_max={args.T_max}.")
    print(f"Model total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # --- Initialize Optimizer ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # --- Training Loop ---
    print(f"\nStarting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train()
        total_epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            states = batch["states"]          # [B, K, S]
            actions_target = batch["actions"] # [B, K, Adim] or [B, K, 1] (for discrete, these are class indices)
            returns_to_go = batch["returns_to_go"] # [B, K, 1]
            timesteps = batch["timesteps"]    # [B, K]
            # mask = batch["mask"] # Optional: if your model uses it explicitly for padding

            # For DT, actions input to the model are typically shifted by one timestep
            # or can be the same as targets if predicting the action that led to current state.
            # Here, we predict action[t] given s[t], rtg[t].
            # The `actions` argument to SpikingDecisionTransformer.forward is the *previous* action.
            # For training, we can feed actions_target directly but shift it for actual generation.
            # For now, let's assume the model's forward pass handles this appropriately or
            # we are predicting a_t from (s_t, r_t, ...).
            # The provided SDT model takes (states, actions, returns_to_go, timesteps)
            # where 'actions' are a_0, ..., a_{T-1}. It predicts a_1, ..., a_T.
            # Let's use a zero action for the first step as is common.
            
            # Create a placeholder for previous actions.
            # For the first action in a sequence, a zero action is often used.
            # The target actions are `actions_target`.
            # The input actions to the model should be a_{t-1} to predict a_t.
            # So, actions_input = [zero_action, actions_target[:, :-1]]
            
            if act_type == "discrete":
                # For discrete, input actions to model are one-hot encoded.
                # Target actions are class indices for CrossEntropyLoss.
                
                # Initialize action_input_tensor with shape [B, K, act_dim]
                # B = states.shape[0], K = states.shape[1] (args.max_length)
                action_input_tensor = torch.zeros(
                    states.shape[0], states.shape[1], act_dim, 
                    dtype=torch.float, device=DEVICE
                )
                
                # Prepare one-hot encoded actions for input a_{t-1}
                # actions_target is [B, K, 1] (indices)
                # actions_target[:, :-1] is [B, K-1, 1]
                # actions_target[:, :-1].squeeze(-1) is [B, K-1] (indices for one_hot)
                if actions_target.shape[1] > 1: # Ensure there is a "previous action"
                    one_hot_prev_actions = torch.nn.functional.one_hot(
                        actions_target[:, :-1].squeeze(-1), num_classes=act_dim
                    ).float()
                    # Assign to the input tensor from the second timestep onwards
                    action_input_tensor[:, 1:] = one_hot_prev_actions
                # The first action input action_input_tensor[:, 0] remains zeros, which is a common choice.
                
                # Target actions for loss are class indices [B, K]
                if actions_target.ndim == 3 and actions_target.shape[-1] == 1:
                    actions_target_for_loss = actions_target.squeeze(-1) 
                else:
                    # This case implies actions_target is already [B, K]
                    actions_target_for_loss = actions_target 
            else: # Continuous
                # For continuous, action_input_tensor is [B, K, act_dim]
                action_input_tensor = torch.zeros_like(actions_target, dtype=torch.float, device=DEVICE)
                if actions_target.shape[1] > 1:
                    action_input_tensor[:, 1:] = actions_target[:, :-1]
                actions_target_for_loss = actions_target # [B, K, Adim]


            model_output = model(
                states=states,
                actions=action_input_tensor, # These are a_{t-1}
                returns_to_go=returns_to_go,
                timesteps=timesteps
            )
            
            action_predictions = model_output['action_predictions'] # These are predicted a_t
            model_metrics = model_output['metrics']

            # Calculate action loss
            if act_type == "discrete":
                # Reshape for CrossEntropyLoss: preds [B*K, num_classes], target [B*K]
                action_loss = loss_fn_action(
                    action_predictions.reshape(-1, act_dim),
                    actions_target_for_loss.reshape(-1)
                )
            else: # Continuous
                action_loss = loss_fn_action(action_predictions, actions_target_for_loss)
            
            # Calculate total loss using compute_spiking_loss
            total_loss = compute_spiking_loss(
                action_loss=action_loss,        # Pass the pre-computed action loss
                metrics=model_metrics,          # Pass all metrics from model
                reg_weight=args.lambda_reg      # Pass regularization weight
            )

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient clipping
            optimizer.step()

            total_epoch_loss += total_loss.item()
            num_batches += 1

        avg_epoch_loss = total_epoch_loss / num_batches
        
        log_data = {
            "avg_loss": avg_epoch_loss,
            "action_loss": action_loss.item(), # Log last batch's action loss
            "reg_loss": model_metrics.get('avg_reg_loss', 0.0), # Log last batch's avg reg loss
            "T_mean": model_metrics.get('avg_T_mean', 0.0), # Log last batch's avg T_mean
            "T_efficiency": model_metrics.get('avg_T_efficiency', 0.0)
        }
        simple_logger(log_data, epoch, args)

        # Save checkpoint
        if (epoch + 1) % args.log_interval == 0 or epoch == args.epochs - 1: # Save more frequently or at end
            checkpoint_path = os.path.join(args.checkpoint_dir, f"sdt_phase1_{args.env_name}_epoch{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
                'args': vars(args),
                'model_config': model_config_args
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    print("\nTraining finished.")
    # Example of loading a checkpoint:
    # checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # print(f"Loaded model from {checkpoint_path}")