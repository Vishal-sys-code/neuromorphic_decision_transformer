"""
D4RL Dataset Loader for Offline Reinforcement Learning and Decision Transformers.

Phase 2 / Gate 2: Dataset Integrity.
Provides utilities to ingest flat D4RL datasets, reconstruct episode boundaries
from termination/timeout metadata and max episode lengths, compute returns-to-go (RTG),
and validate dataset integrity.
"""

import os
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np

D4RL_URL_MAP = {
    "hopper-medium-v2": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/hopper_medium-v2.hdf5",
    "hopper-medium-replay-v2": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/hopper_medium_replay-v2.hdf5",
    "hopper-medium-expert-v2": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/hopper_medium_expert-v2.hdf5",
    "halfcheetah-medium-v2": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/halfcheetah_medium-v2.hdf5",
    "halfcheetah-medium-expert-v2": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/halfcheetah_medium_expert-v2.hdf5",
    "walker2d-medium-v2": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/walker2d_medium-v2.hdf5",
    "walker2d-medium-expert-v2": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/walker2d_medium_expert-v2.hdf5",
    "ant-medium-v2": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/ant_medium-v2.hdf5",
}


def split_d4rl_episodes(
    dataset: Dict[str, Any],
    max_episode_length: int = 1000,
    gamma: float = 1.0,
) -> List[Dict[str, np.ndarray]]:
    """
    Convert a flat D4RL dataset into episode dictionaries.
    Supports the usual D4RL keys:
        observations
        actions
        rewards
        terminals
        timeouts (optional)
        next_observations (optional)

    Returns complete episodes containing:
        tau_i = {s_t, a_t, r_t, d_t, R_t, t}
        where R_t = sum_{k=t}^{T_i-1} gamma^{k-t} r_k (gamma=1.0 by default).
    """
    observations = np.asarray(dataset["observations"], dtype=np.float32)
    actions = np.asarray(dataset["actions"], dtype=np.float32)
    rewards = np.asarray(dataset["rewards"], dtype=np.float32)
    terminals = np.asarray(dataset["terminals"], dtype=bool)

    timeouts = dataset.get("timeouts")
    if timeouts is not None:
        timeouts = np.asarray(timeouts, dtype=bool)
    else:
        timeouts = np.zeros_like(terminals)

    next_observations = dataset.get("next_observations")
    if next_observations is not None:
        next_observations = np.asarray(next_observations, dtype=np.float32)

    episodes = []
    start = 0
    n = len(observations)

    for t in range(n):
        final_step = (
            terminals[t]
            or timeouts[t]
            or (t - start + 1 >= max_episode_length)
            or (t == n - 1)
        )
        if final_step:
            end = t + 1
            episode: Dict[str, np.ndarray] = {
                "observations": observations[start:end],
                "actions": actions[start:end],
                "rewards": rewards[start:end],
                "terminals": terminals[start:end],
                "timeouts": timeouts[start:end],
            }
            if next_observations is not None:
                episode["next_observations"] = next_observations[start:end]

            # Returns-to-go (undiscounted gamma=1.0 by default)
            if gamma == 1.0:
                episode["returns_to_go"] = np.flip(
                    np.cumsum(np.flip(episode["rewards"]))
                ).copy()
            else:
                rtg = np.zeros_like(episode["rewards"], dtype=np.float32)
                running_return = 0.0
                for k in reversed(range(len(episode["rewards"]))):
                    running_return = float(episode["rewards"][k]) + gamma * running_return
                    rtg[k] = running_return
                episode["returns_to_go"] = rtg

            # Episode-local timestep
            episode["timesteps"] = np.arange(
                len(episode["observations"]),
                dtype=np.int64,
            )
            episodes.append(episode)
            start = end

    return episodes


def validate_episodes(episodes: List[Dict[str, np.ndarray]]) -> bool:
    """
    Validate the dataset integrity of segmented episodes (Gate 2 check).

    Verifies:
      1. Non-empty sequence of valid episode dictionaries.
      2. Required fields exist: observations, actions, rewards, terminals, timeouts,
         returns_to_go, timesteps.
      3. Length consistency across all per-step arrays within every episode.
      4. Episode timesteps strictly form contiguous 0-indexed ranges [0, T_i - 1].
      5. Returns-to-go match backwards cumulative sums of rewards without drift.
      6. No NaNs or infinities in numerical arrays.
      7. Terminal and timeout masks are strictly boolean.

    Returns:
        bool: True if dataset integrity checks pass, False otherwise.
    """
    if not isinstance(episodes, (list, tuple)) or len(episodes) == 0:
        return False

    required_keys = {
        "observations",
        "actions",
        "rewards",
        "terminals",
        "timeouts",
        "returns_to_go",
        "timesteps",
    }

    for ep in episodes:
        if not isinstance(ep, dict):
            return False

        if not required_keys.issubset(ep.keys()):
            return False

        ep_len = len(ep["observations"])
        if ep_len == 0:
            return False

        for k in required_keys:
            if not isinstance(ep[k], np.ndarray):
                return False
            if len(ep[k]) != ep_len:
                return False

        if "next_observations" in ep:
            if not isinstance(ep["next_observations"], np.ndarray):
                return False
            if len(ep["next_observations"]) != ep_len:
                return False

        # Timesteps check: [0, 1, ..., T_i - 1]
        expected_timesteps = np.arange(ep_len, dtype=np.int64)
        if not np.array_equal(ep["timesteps"], expected_timesteps):
            return False

        # Returns-to-go check (undiscounted check with numerical tolerance)
        expected_rtg = np.flip(np.cumsum(np.flip(ep["rewards"])))
        if not np.allclose(ep["returns_to_go"], expected_rtg, atol=1e-5):
            return False

        # Numerical integrity: no NaNs / infs
        for num_key in ["observations", "actions", "rewards", "returns_to_go"]:
            if not np.isfinite(ep[num_key]).all():
                return False

        # Boolean check
        if ep["terminals"].dtype != bool and ep["terminals"].dtype != np.bool_:
            return False
        if ep["timeouts"].dtype != bool and ep["timeouts"].dtype != np.bool_:
            return False

    return True


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
    timeouts: Optional[np.ndarray] = None
    next_observations: Optional[np.ndarray] = None

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        if key == "terminals":
            return self.dones
        raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key) or (key == "terminals" and hasattr(self, "dones"))

    @classmethod
    def from_dict(cls, episode: Dict[str, np.ndarray]) -> "Trajectory":
        return cls(
            observations=episode["observations"],
            actions=episode["actions"],
            rewards=episode["rewards"],
            dones=episode["terminals"].astype(np.float32),
            returns_to_go=episode["returns_to_go"],
            timesteps=episode["timesteps"],
            length=len(episode["observations"]),
            timeouts=episode.get("timeouts"),
            next_observations=episode.get("next_observations"),
        )


class D4RLLoader:
    """
    Loader for D4RL MuJoCo datasets (e.g. hopper-medium-v2, halfcheetah-medium-v2).
    """

    def __init__(self, gamma: float = 1.0, max_episode_length: int = 1000):
        self.gamma = gamma
        self.max_episode_length = max_episode_length

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

    @staticmethod
    def resolve_or_download_dataset(
        env_name: str,
        data_dir: str = "data/d4rl_raw",
        auto_download: bool = True,
    ) -> Optional[str]:
        """
        Locate existing HDF5 dataset file or download it from D4RL repository.
        Returns the resolved file path or None if not available.
        """
        env_key = env_name.lower().replace("_", "-")
        candidates = [
            f"{env_key}.hdf5",
            f"{env_key.replace('-', '_')}.hdf5",
            f"{env_key.split('-')[0]}_{'_'.join(env_key.split('-')[1:])}.hdf5" if '-' in env_key else f"{env_key}.hdf5",
        ]
        if env_key == "hopper-medium-v2":
            candidates.insert(0, "hopper_medium-v2.hdf5")
        elif env_key == "hopper-medium-replay-v2":
            candidates.insert(0, "hopper_medium_replay-v2.hdf5")
        elif env_key == "hopper-medium-expert-v2":
            candidates.insert(0, "hopper_medium_expert-v2.hdf5")
        elif env_key == "halfcheetah-medium-v2":
            candidates.insert(0, "halfcheetah_medium-v2.hdf5")
        elif env_key == "halfcheetah-medium-expert-v2":
            candidates.insert(0, "halfcheetah_medium_expert-v2.hdf5")
        elif env_key == "walker2d-medium-v2":
            candidates.insert(0, "walker2d_medium-v2.hdf5")
        elif env_key == "walker2d-medium-expert-v2":
            candidates.insert(0, "walker2d_medium_expert-v2.hdf5")
        elif env_key == "ant-medium-v2":
            candidates.insert(0, "ant_medium-v2.hdf5")

        search_dirs = [data_dir, "data/d4rl_raw", "data/d4rl", "data"]
        for d in search_dirs:
            if not os.path.exists(d):
                continue
            for cand in candidates:
                p = os.path.join(d, cand)
                if os.path.exists(p) and os.path.getsize(p) > 0:
                    return p

        if auto_download and env_key in D4RL_URL_MAP:
            url = D4RL_URL_MAP[env_key]
            out_filename = url.split("/")[-1]
            out_path = os.path.join(data_dir, out_filename)
            os.makedirs(data_dir, exist_ok=True)
            print(f"[D4RLLoader] Downloading {env_key} from {url} to {out_path}...")
            try:
                urllib.request.urlretrieve(url, out_path)
                if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                    return out_path
            except Exception as e:
                print(f"[D4RLLoader] Download failed ({e}). Falling back to synthetic dataset.")

        return None

    def load_from_hdf5(self, file_path: str, max_state_dim: Optional[int] = None) -> List[Trajectory]:
        """
        Load trajectories from a D4RL HDF5 file.
        Expects keys: 'observations', 'actions', 'rewards', 'terminals', optional 'timeouts'.
        If max_state_dim is specified, slices observations to match expected state dimension
        (e.g., slicing 111-dim D4RL Ant observations down to 27-dim kinematics for Ant-v4).
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"D4RL dataset file not found: {file_path}")

        with h5py.File(file_path, "r") as f:
            obs = f["observations"][:].astype(np.float32)
            if max_state_dim is not None and obs.shape[1] > max_state_dim:
                obs = obs[:, :max_state_dim]

            dataset: Dict[str, Any] = {
                "observations": obs,
                "actions": f["actions"][:].astype(np.float32),
                "rewards": f["rewards"][:].astype(np.float32),
                "terminals": f["terminals"][:].astype(bool),
            }
            if "timeouts" in f:
                dataset["timeouts"] = f["timeouts"][:].astype(bool)
            if "next_observations" in f:
                next_obs = f["next_observations"][:].astype(np.float32)
                if max_state_dim is not None and next_obs.shape[1] > max_state_dim:
                    next_obs = next_obs[:, :max_state_dim]
                dataset["next_observations"] = next_obs

        episodes = split_d4rl_episodes(
            dataset=dataset,
            max_episode_length=self.max_episode_length,
            gamma=self.gamma,
        )
        return [Trajectory.from_dict(ep) for ep in episodes]

    @staticmethod
    def compute_state_statistics(trajectories: List[Trajectory]) -> Tuple[np.ndarray, np.ndarray]:
        """Compute mean and standard deviation of states for normalization."""
        all_obs = np.concatenate([traj.observations for traj in trajectories], axis=0)
        mean = np.mean(all_obs, axis=0)
        std = np.std(all_obs, axis=0) + 1e-6
        return mean.astype(np.float32), std.astype(np.float32)
