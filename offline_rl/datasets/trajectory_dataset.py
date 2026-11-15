"""
Decision Transformer Trajectory Dataset and Sequence Sampler.

Phase 3 / Gate 3: Batch shapes are correct.
Sub-sequence dataset for Decision Transformer (DT) and Spiking Decision Transformer (SNN-DT).
Samples sub-trajectories of context length K: (R_t, s_t, a_t, t)
with episode sampling probability proportional to trajectory length:
    P(tau_i) = T_i / sum_j T_j
making each timestep approximately equally likely to be the sampled start location.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    """
    Sub-sequence dataset for Decision Transformer training.
    Produces batches of shape:
        states: (K, state_dim)
        actions: (K, act_dim)
        returns_to_go: (K, 1)
        timesteps: (K,)
        attention_mask: (K,)
    """

    def __init__(
        self,
        episodes: Optional[List[Any]] = None,
        context_length: int = 20,
        state_mean: Optional[np.ndarray] = None,
        state_std: Optional[np.ndarray] = None,
        scale: float = 1.0,
        max_ep_len: int = 1000,
        trajectories: Optional[List[Any]] = None,
        context_len: Optional[int] = None,
    ):
        if episodes is None and trajectories is not None:
            episodes = trajectories
        if episodes is None:
            raise ValueError("TrajectoryDataset requires an 'episodes' or 'trajectories' list.")

        if context_len is not None:
            context_length = context_len

        # Filter out empty episodes if any
        self.episodes = [
            ep
            for ep in episodes
            if (len(ep["observations"]) if isinstance(ep, dict) or hasattr(ep, "__getitem__") else len(ep.observations)) > 0
        ]
        if not self.episodes:
            raise ValueError("TrajectoryDataset received an empty episode list.")

        self.context_length = context_length
        self.scale = float(scale)
        self.max_ep_len = max_ep_len

        # Compute or store normalization stats
        states = np.concatenate(
            [
                (ep["observations"] if isinstance(ep, dict) or hasattr(ep, "__getitem__") else ep.observations)
                for ep in self.episodes
            ],
            axis=0,
        )
        if state_mean is None:
            state_mean = states.mean(axis=0)
        if state_std is None:
            state_std = states.std(axis=0) + 1e-6
        self.state_mean = np.asarray(state_mean, dtype=np.float32)
        self.state_std = np.asarray(state_std, dtype=np.float32)
        self.episode_lengths = np.array(
            [
                len(ep["observations"] if isinstance(ep, dict) or hasattr(ep, "__getitem__") else ep.observations)
                for ep in self.episodes
            ],
            dtype=np.float64,
        )

        # Sampling probabilities proportional to trajectory length: P(tau_i) = T_i / sum_j T_j
        total_steps = float(np.sum(self.episode_lengths))
        if total_steps > 0:
            self.sampling_probs = self.episode_lengths / total_steps
        else:
            self.sampling_probs = np.ones(len(self.episodes), dtype=np.float64) / len(self.episodes)
        # Ensure probabilities sum to exactly 1.0
        self.sampling_probs = self.sampling_probs / np.sum(self.sampling_probs)

        first_ep = self.episodes[0]
        first_obs = first_ep["observations"] if isinstance(first_ep, dict) or hasattr(first_ep, "__getitem__") else first_ep.observations
        first_act = first_ep["actions"] if isinstance(first_ep, dict) or hasattr(first_ep, "__getitem__") else first_ep.actions
        self.state_dim = first_obs.shape[1] if first_obs.ndim > 1 else 1
        self.act_dim = first_act.shape[1] if first_act.ndim > 1 else 1

    @property
    def context_len(self) -> int:
        return self.context_length

    @property
    def traj_lengths(self) -> np.ndarray:
        return self.episode_lengths.astype(np.int64)

    @property
    def trajectories(self) -> List[Any]:
        return self.episodes

    def __len__(self) -> int:
        return int(np.sum(self.episode_lengths))

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        # Sample trajectory with probability proportional to its length
        ep_idx = int(np.random.choice(len(self.episodes), p=self.sampling_probs))
        ep = self.episodes[ep_idx]

        obs = ep["observations"] if isinstance(ep, dict) or hasattr(ep, "__getitem__") else ep.observations
        actions = ep["actions"] if isinstance(ep, dict) or hasattr(ep, "__getitem__") else ep.actions

        if (isinstance(ep, dict) and "returns_to_go" in ep) or hasattr(ep, "returns_to_go"):
            returns_to_go = ep["returns_to_go"] if isinstance(ep, dict) else ep.returns_to_go
        elif (isinstance(ep, dict) and "rewards" in ep) or hasattr(ep, "rewards"):
            rewards = ep["rewards"] if isinstance(ep, dict) else ep.rewards
            returns_to_go = np.flip(np.cumsum(np.flip(rewards))).copy()
        else:
            raise KeyError("Episode missing 'returns_to_go' or 'rewards'.")

        if (isinstance(ep, dict) and "timesteps" in ep) or hasattr(ep, "timesteps"):
            timesteps = ep["timesteps"] if isinstance(ep, dict) else ep.timesteps
        else:
            timesteps = np.arange(len(obs), dtype=np.int64)

        traj_len = len(obs)
        # Sample a start timestep within the trajectory uniformly
        si = int(np.random.randint(0, traj_len)) if traj_len > 0 else 0

        # Extract sequence [si : si + context_length]
        s = obs[si : si + self.context_length]
        a = actions[si : si + self.context_length]
        r = returns_to_go[si : si + self.context_length]
        t = timesteps[si : si + self.context_length]

        tlen = len(s)

        # Normalize states
        s = (s.astype(np.float32) - self.state_mean) / self.state_std

        # Scale returns to go
        if self.scale != 1.0:
            r = r.astype(np.float32) / self.scale
        else:
            r = r.astype(np.float32)

        a = a.astype(np.float32)
        if a.ndim == 1:
            a = a[:, None]

        if r.ndim == 1:
            r = r[:, None]

        # Pad left if sequence length < context_length
        if tlen < self.context_length:
            pad_len = self.context_length - tlen
            s = np.concatenate([np.zeros((pad_len, self.state_dim), dtype=np.float32), s], axis=0)
            a = np.concatenate([np.zeros((pad_len, self.act_dim), dtype=np.float32), a], axis=0)
            r = np.concatenate([np.zeros((pad_len, 1), dtype=np.float32), r], axis=0)
            t = np.concatenate([np.zeros((pad_len,), dtype=np.int64), t], axis=0)
            mask = np.concatenate([np.zeros((pad_len,), dtype=np.float32), np.ones((tlen,), dtype=np.float32)], axis=0)
        else:
            mask = np.ones((self.context_length,), dtype=np.float32)

        return {
            "states": torch.from_numpy(s).float(),
            "actions": torch.from_numpy(a).float(),
            "returns_to_go": torch.from_numpy(r).float(),
            "timesteps": torch.from_numpy(t).long(),
            "attention_mask": torch.from_numpy(mask).float(),
        }


# Backwards compatibility alias
TrajectorySequenceDataset = TrajectoryDataset
