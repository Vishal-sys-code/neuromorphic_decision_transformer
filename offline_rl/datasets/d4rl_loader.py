"""
D4RL Dataset Loader for MuJoCo environments.

Provides utilities to ingest offline trajectory datasets from HDF5 files or
legacy D4RL format, extract continuous trajectories, compute returns-to-go (RTG),
and compute state normalization parameters.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np


@dataclass
class Trajectory:
    """Represents a single episodic trajectory in offline RL."""
    observations: np.ndarray  # (L, state_dim)
    actions: np.ndarray       # (L, act_dim)
    rewards: np.ndarray       # (L,)
    dones: np.ndarray         # (L,)
    returns_to_go: np.ndarray # (L,)
    timesteps: np.ndarray     # (L,)
    length: int


class D4RLLoader:
    """
    Loader for D4RL MuJoCo datasets (e.g. hopper-medium-v2, halfcheetah-medium-v2).
    """

    def __init__(self, gamma: float = 1.0):
        self.gamma = gamma

    @staticmethod
    def create_synthetic_dataset(
        num_trajectories: int = 20,
        traj_len: int = 100,
        state_dim: int = 11,
        act_dim: int = 3,
        seed: int = 42,
    ) -> List[Trajectory]:
        """Generate synthetic trajectories for deterministic testing and prototyping."""
        rng = np.random.RandomState(seed)
        trajectories = []
        for _ in range(num_trajectories):
            length = rng.randint(max(10, traj_len // 2), traj_len + 1)
            obs = rng.randn(length, state_dim).astype(np.float32)
            act = np.clip(rng.randn(length, act_dim).astype(np.float32), -1.0, 1.0)
            rew = (rng.rand(length).astype(np.float32) * 2.0)
            dones = np.zeros(length, dtype=np.float32)
            dones[-1] = 1.0

            rtg = np.zeros_like(rew)
            running_return = 0.0
            for t in reversed(range(length)):
                running_return = rew[t] + running_return
                rtg[t] = running_return

            timesteps = np.arange(length, dtype=np.int64)

            trajectories.append(
                Trajectory(
                    observations=obs,
                    actions=act,
                    rewards=rew,
                    dones=dones,
                    returns_to_go=rtg,
                    timesteps=timesteps,
                    length=length,
                )
            )
        return trajectories

    def load_from_hdf5(self, file_path: str) -> List[Trajectory]:
        """
        Load trajectories from a D4RL HDF5 file.
        Expects keys: 'observations', 'actions', 'rewards', 'terminals', optional 'timeouts'.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"D4RL dataset file not found: {file_path}")

        with h5py.File(file_path, "r") as f:
            observations = f["observations"][:].astype(np.float32)
            actions = f["actions"][:].astype(np.float32)
            rewards = f["rewards"][:].astype(np.float32)
            terminals = f["terminals"][:].astype(np.bool_)
            if "timeouts" in f:
                timeouts = f["timeouts"][:].astype(np.bool_)
            else:
                timeouts = np.zeros_like(terminals)

        return self._process_raw_arrays(observations, actions, rewards, terminals, timeouts)

    def _process_raw_arrays(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        terminals: np.ndarray,
        timeouts: np.ndarray,
    ) -> List[Trajectory]:
        """Convert flat D4RL arrays into segmented Trajectory objects with returns-to-go."""
        total_steps = observations.shape[0]
        trajectories = []
        start_idx = 0

        for i in range(total_steps):
            is_done = bool(terminals[i])
            is_timeout = bool(timeouts[i])

            if is_done or is_timeout:
                end_idx = i + 1
                traj = self._build_trajectory(
                    observations[start_idx:end_idx],
                    actions[start_idx:end_idx],
                    rewards[start_idx:end_idx],
                    terminals[start_idx:end_idx].astype(np.float32),
                )
                trajectories.append(traj)
                start_idx = end_idx

        # Handle trailing segment if not explicitly terminated
        if start_idx < total_steps:
            traj = self._build_trajectory(
                observations[start_idx:],
                actions[start_idx:],
                rewards[start_idx:],
                terminals[start_idx:].astype(np.float32),
            )
            trajectories.append(traj)

        return trajectories

    def _build_trajectory(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: np.ndarray,
        dones: np.ndarray,
    ) -> Trajectory:
        length = obs.shape[0]
        rtg = np.zeros_like(rew, dtype=np.float32)
        running_ret = 0.0
        for t in reversed(range(length)):
            running_ret = rew[t] + self.gamma * running_ret
            rtg[t] = running_ret

        timesteps = np.arange(length, dtype=np.int64)

        return Trajectory(
            observations=obs,
            actions=act,
            rewards=rew,
            dones=dones,
            returns_to_go=rtg,
            timesteps=timesteps,
            length=length,
        )

    @staticmethod
    def compute_state_statistics(trajectories: List[Trajectory]) -> Tuple[np.ndarray, np.ndarray]:
        """Compute mean and standard deviation of states for normalization."""
        all_obs = np.concatenate([traj.observations for traj in trajectories], axis=0)
        mean = np.mean(all_obs, axis=0)
        std = np.std(all_obs, axis=0) + 1e-6
        return mean.astype(np.float32), std.astype(np.float32)
