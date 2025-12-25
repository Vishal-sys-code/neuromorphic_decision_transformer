import torch
from torch.utils.data import Dataset
import numpy as np

class OfflineSequenceDataset(Dataset):
    """
    Dataset for sequence-based models like Decision Transformer.
    """
    def __init__(self, path, seq_len):
        data = np.load(path, mmap_mode='r')
        self.states = torch.from_numpy(data["states"]).float()
        self.actions = torch.from_numpy(data["actions"]).float()
        self.returns_to_go = torch.from_numpy(data["returns_to_go"]).float()
        self.timesteps = torch.from_numpy(data["timesteps"]).long()
        self.masks = torch.from_numpy(data["mask"]).float()
        self.seq_len = seq_len

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return {
            "states": self.states[idx, :self.seq_len],
            "actions": self.actions[idx, :self.seq_len],
            "returns_to_go": self.returns_to_go[idx, :self.seq_len],
            "timesteps": self.timesteps[idx, :self.seq_len],
            "mask": self.masks[idx, :self.seq_len],
        }

class OfflineTransitionDataset(Dataset):
    """
    Dataset for transition-based models like IQL and CQL.
    Processes trajectories into individual (s, a, r, s', d) transitions.
    """
    def __init__(self, dataset_path):
        data = np.load(dataset_path, mmap_mode='r')
        
        # Calculate total number of transitions
        total_transitions = int(np.sum(data['mask'])) - data['mask'].shape[0]

        state_dim = data['states'].shape[2]
        action_dim = data['actions'].shape[2]

        # Pre-allocate memory for efficiency
        self.states = np.zeros((total_transitions, state_dim), dtype=np.float32)
        self.actions = np.zeros((total_transitions, action_dim), dtype=np.float32)
        self.rewards = np.zeros((total_transitions, 1), dtype=np.float32)
        self.next_states = np.zeros((total_transitions, state_dim), dtype=np.float32)
        self.dones = np.zeros((total_transitions, 1), dtype=np.float32)
        
        current_idx = 0
        for i in range(data['states'].shape[0]):
            traj_len = int(data['mask'][i].sum())
            if traj_len <= 1:
                continue

            traj_states = data['states'][i, :traj_len-1]
            traj_actions = data['actions'][i, :traj_len-1]
            traj_rtg = data['returns_to_go'][i, :traj_len-1]
            next_traj_rtg = data['returns_to_go'][i, 1:traj_len]
            
            self.states[current_idx : current_idx + traj_len - 1] = traj_states
            self.actions[current_idx : current_idx + traj_len - 1] = traj_actions
            self.rewards[current_idx : current_idx + traj_len - 1] = (traj_rtg - next_traj_rtg).reshape(-1, 1)
            self.next_states[current_idx : current_idx + traj_len - 1] = data['states'][i, 1:traj_len]
            self.dones[current_idx + traj_len - 2] = 1.0 # Mark the last transition as done
            
            current_idx += traj_len - 1
            
        # Convert to tensors
        self.states = torch.from_numpy(self.states)
        self.actions = torch.from_numpy(self.actions)
        self.rewards = torch.from_numpy(self.rewards)
        self.next_states = torch.from_numpy(self.next_states)
        self.dones = torch.from_numpy(self.dones)

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