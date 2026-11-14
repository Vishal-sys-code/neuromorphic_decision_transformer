"""
Trajectory Sequence Dataset for Decision Transformer offline training.

Samples sub-trajectories of context length K from episodic datasets,
applies state normalization and return scaling, and formats tensors
for causal autoregressive conditioning: (R_t, s_t, a_t).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from offline_rl.datasets.d4rl_loader import Trajectory


class TrajectorySequenceDataset(Dataset):
    """
    Sub-sequence dataset for Decision Transformer training.
    Produces batches of shape:
        states: (K, state_dim)
        actions: (K, act_dim)
        returns_to_go: (K, 1)
        timesteps: (K,)
        mask: (K,)
    """

    def __init__(
        self,
        trajectories: List[Trajectory],
        context_len: int = 20,
        scale: float = 1000.0,
        state_mean: Optional[np.ndarray] = None,
        state_std: Optional[np.ndarray] = None,
        max_ep_len: int = 1000,
    ):
        self.trajectories = [t for t in trajectories if t.length > 0]
        if not self.trajectories:
            raise ValueError("TrajectorySequenceDataset received an empty trajectory list.")

        self.context_len = context_len
        self.scale = scale
        self.max_ep_len = max_ep_len

        # Compute or store normalization stats
        if state_mean is None or state_std is None:
            all_obs = np.concatenate([t.observations for t in self.trajectories], axis=0)
            self.state_mean = np.mean(all_obs, axis=0).astype(np.float32)
            self.state_std = (np.std(all_obs, axis=0) + 1e-6).astype(np.float32)
        else:
            self.state_mean = state_mean.astype(np.float32)
            self.state_std = state_std.astype(np.float32)

        # Sampling probabilities proportional to trajectory length
        self.traj_lengths = np.array([t.length for t in self.trajectories], dtype=np.int64)
        self.sampling_probs = self.traj_lengths / np.sum(self.traj_lengths)

        self.state_dim = self.trajectories[0].observations.shape[1]
        self.act_dim = self.trajectories[0].actions.shape[1]

    def __len__(self) -> int:
        # Standard: return total number of transitions or arbitrary large epoch size
        return int(np.sum(self.traj_lengths))

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        # Sample trajectory with probability proportional to its length
        traj_idx = np.random.choice(len(self.trajectories), p=self.sampling_probs)
        traj = self.trajectories[traj_idx]

        # Sample a start timestep within the trajectory
        si = np.random.randint(0, traj.length)

        # Extract sequence [si : si + context_len]
        s = traj.observations[si : si + self.context_len]
        a = traj.actions[si : si + self.context_len]
        r = traj.returns_to_go[si : si + self.context_len]
        timesteps = traj.timesteps[si : si + self.context_len]

        tlen = s.shape[0]

        # Normalize states
        s = (s - self.state_mean) / self.state_std

        # Scale returns to go
        r = r / self.scale

        # Pad left if sequence length < context_len
        if tlen < self.context_len:
            pad_len = self.context_len - tlen
            s = np.concatenate([np.zeros((pad_len, self.state_dim), dtype=np.float32), s], axis=0)
            a = np.concatenate([np.zeros((pad_len, self.act_dim), dtype=np.float32), a], axis=0)
            r = np.concatenate([np.zeros((pad_len,), dtype=np.float32), r], axis=0)
            timesteps = np.concatenate([np.zeros((pad_len,), dtype=np.int64), timesteps], axis=0)
            mask = np.concatenate([np.zeros((pad_len,), dtype=np.float32), np.ones((tlen,), dtype=np.float32)], axis=0)
        else:
            mask = np.ones((self.context_len,), dtype=np.float32)

        # Reshape returns-to-go to (K, 1)
        r = r[:, None]

        return {
            "states": torch.from_numpy(s).float(),
            "actions": torch.from_numpy(a).float(),
            "returns_to_go": torch.from_numpy(r).float(),
            "timesteps": torch.from_numpy(timesteps).long(),
            "attention_mask": torch.from_numpy(mask).float(),
        }
