import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)

def load_data_from_h5(h5_path):
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Dataset not found at {h5_path}")
        
    with h5py.File(h5_path, 'r') as f:
        observations = f['observations'][:]
        actions = f['actions'][:]
        rewards = f['rewards'][:]
        terminals = f['terminals'][:]
        # Handle timeouts
        if 'timeouts' in f:
            timeouts = f['timeouts'][:]
        else:
            timeouts = np.zeros_like(terminals)
            
    return observations, actions, rewards, terminals, timeouts

def compute_trajectories(observations, actions, rewards, terminals, timeouts):
    trajectories = []
    
    N = observations.shape[0]
    start = 0
    
    for i in range(N):
        done_bool = bool(terminals[i])
        final_timestep = bool(timeouts[i])
        
        if done_bool or final_timestep:
            end = i + 1
            traj = {
                'observations': observations[start:end],
                'actions': actions[start:end],
                'rewards': rewards[start:end],
                'dones': terminals[start:end], # Keep as is (0 or 1)
                'length': end - start
            }
            trajectories.append(traj)
            start = end
            
    # Handle last trajectory if not terminated explicitly
    if start < N:
        traj = {
            'observations': observations[start:],
            'actions': actions[start:],
            'rewards': rewards[start:],
            'dones': terminals[start:],
            'length': N - start
        }
        trajectories.append(traj)
        
    return trajectories

def compute_rtg(trajectories):
    for traj in trajectories:
        rewards = traj['rewards']
        rtg = np.zeros_like(rewards)
        running_return = 0
        for t in reversed(range(len(rewards))):
            running_return += rewards[t]
            rtg[t] = running_return
        traj['returns_to_go'] = rtg
    return trajectories

def compute_normalization(trajectories):
    states = []
    for traj in trajectories:
        states.append(traj['observations'])
    states = np.concatenate(states, axis=0)
    
    mean = np.mean(states, axis=0)
    std = np.std(states, axis=0) + 1e-6 # Avoid div by zero
    
    return mean, std

class D4RLSequenceDataset(Dataset):
    def __init__(self, env_name, data_dir="data/d4rl_raw", seq_len=50):
        self.env_name = env_name
        self.seq_len = seq_len
        
        # Construct filename. logic: hopper-medium-v2 -> hopper_medium-v2.hdf5?
        # Re-using logic from download/convert:
        # download stores as {url_filename}.
        # url: .../hopper_medium-v2.hdf5
        # So look for hopper_medium-v2.hdf5 if env is hopper-medium-v2.
        # But wait, env name has hyphens. Filename usually has underscores.
        # Try both.
        
        filename = f"{env_name}.hdf5"
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            # Try underscore
            filename_us = env_name.replace('-', '_') + ".hdf5"
            path = os.path.join(data_dir, filename_us)
            
        if not os.path.exists(path):
            # Try partial underscore (d4rl style: hopper_medium-v2.hdf5)
            # Split by -v
            parts = env_name.split('-v')
            base = parts[0].replace('-', '_')
            suffix = f"-v{parts[1]}"
            filename_mixed = base + suffix + ".hdf5"
            path = os.path.join(data_dir, filename_mixed)

        logger.info(f"Loading dataset from {path}")
        
        obs, act, rew, term, time = load_data_from_h5(path)
        self.trajectories = compute_trajectories(obs, act, rew, term, time)
        self.trajectories = compute_rtg(self.trajectories)
        self.state_mean, self.state_std = compute_normalization(self.trajectories)
        
        # Pre-compute valid indices
        self.indices = []
        for i, traj in enumerate(self.trajectories):
            # For each trajectory, valid start indices
            # We want windows of length seq_len.
            # If traj_len < seq_len, we only have 1 window (0 to traj_len, padded)
            # If traj_len >= seq_len, we have traj_len - seq_len + 1 windows?
            # Standard DT: samples random t in [0, traj_len - 1].
            # Then takes [t, t+seq_len].
            # Pads if goes over.
            # I will follow this "sample any start point" logic to maximize data usage.
            
            T = traj['length']
            for t in range(T):
                self.indices.append((i, t))
                
        self.state_dim = self.state_mean.shape[0]
        self.act_dim = self.trajectories[0]['actions'].shape[1]
        
        # Check discrete/continuous from data (heuristic)
        # Actually passed from config usually, but we can guess.
        # D4RL MuJoCo is continuous.
        self.is_discrete = False 

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        traj_idx, start_t = self.indices[idx]
        traj = self.trajectories[traj_idx]
        T = traj['length']
        
        # Determine end index
        end_t = start_t + self.seq_len
        
        # Prepare buffers
        states = np.zeros((self.seq_len, self.state_dim), dtype=np.float32)
        actions = np.zeros((self.seq_len, self.act_dim), dtype=np.float32)
        rewards = np.zeros((self.seq_len, 1), dtype=np.float32)
        rtg = np.zeros((self.seq_len, 1), dtype=np.float32)
        timesteps = np.zeros((self.seq_len), dtype=np.int64)
        mask = np.zeros((self.seq_len), dtype=np.float32)
        dones = np.zeros((self.seq_len, 1), dtype=np.float32)
        
        # Calculate real data range
        real_end_t = min(end_t, T)
        real_len = real_end_t - start_t
        
        # Extract data
        s_data = traj['observations'][start_t:real_end_t]
        a_data = traj['actions'][start_t:real_end_t]
        r_data = traj['rewards'][start_t:real_end_t]
        rtg_data = traj['returns_to_go'][start_t:real_end_t]
        d_data = traj['dones'][start_t:real_end_t]
        
        # Normalize states
        s_data = (s_data - self.state_mean) / self.state_std
        
        # Fill buffers
        states[:real_len] = s_data
        actions[:real_len] = a_data
        rewards[:real_len] = r_data.reshape(-1, 1)
        rtg[:real_len] = rtg_data.reshape(-1, 1)
        timesteps[:real_len] = np.arange(start_t, real_end_t)
        mask[:real_len] = 1.0
        dones[:real_len] = d_data.reshape(-1, 1)
        
        return {
            "states": torch.from_numpy(states),
            "actions": torch.from_numpy(actions),
            "rewards": torch.from_numpy(rewards),
            "returns_to_go": torch.from_numpy(rtg),
            "timesteps": torch.from_numpy(timesteps),
            "mask": torch.from_numpy(mask),
            "dones": torch.from_numpy(dones) # Optional, but good to have
        }

class D4RLTransitionDataset(Dataset):
    def __init__(self, env_name, data_dir="data/d4rl_raw"):
        self.env_name = env_name
        
        # Similar filename logic
        filename = f"{env_name}.hdf5"
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            filename_us = env_name.replace('-', '_') + ".hdf5"
            path = os.path.join(data_dir, filename_us)
        if not os.path.exists(path):
            parts = env_name.split('-v')
            base = parts[0].replace('-', '_')
            suffix = f"-v{parts[1]}"
            filename_mixed = base + suffix + ".hdf5"
            path = os.path.join(data_dir, filename_mixed)

        logger.info(f"Loading transition dataset from {path}")
        
        obs, act, rew, term, time = load_data_from_h5(path)
        
        # For transitions (s, a, r, s'), we need next states.
        # We can reconstruct next states from observations: s[t+1]
        # But we need to be careful about boundaries.
        
        # Vectorized transition creation
        # Identify terminals to mask out transitions crossing episodes
        # terminals[i] means step i is terminal. Next step i+1 is start of new episode (or end of data).
        # We want (s_i, a_i, r_i, s_{i+1}, d_i).
        # If d_i is True, s_{i+1} might be invalid or from next episode.
        # In D4RL, if d_i=True, s_{i+1} is usually the reset state of next traj.
        # But for offline RL, we treat s_{i+1} as terminal state if available, or just mask it.
        # However, many algorithms expect 'next_state' to calculate target Q.
        # If done=True, target Q is usually just r. So next_state doesn't matter much (but should be valid shape).
        
        N = obs.shape[0]
        
        # Create next_obs array
        next_obs = np.zeros_like(obs)
        next_obs[:-1] = obs[1:]
        next_obs[-1] = obs[-1] # Fallback
        
        # Compute mean/std
        self.state_mean = np.mean(obs, axis=0)
        self.state_std = np.std(obs, axis=0) + 1e-6
        
        # Normalize current states
        self.states = (obs - self.state_mean) / self.state_std
        # Normalize next states
        self.next_states = (next_obs - self.state_mean) / self.state_std
        
        self.actions = act
        self.rewards = rew.reshape(-1, 1)
        self.dones = term.reshape(-1, 1)
        
        # Filter out invalid transitions (where step i was terminal or timeout, so i+1 is not next state)
        # Actually, if step i is terminal, (s_i, a_i, r_i, s_i', d_i=1) is valid.
        # But s_i' (next_obs[i]) corresponds to obs[i+1], which is START of next episode.
        # This is WRONG. s_i' should be the terminal state of current episode.
        # But D4RL often doesn't store the final observation after 'done'.
        # However, standard practice is: if done, next_state doesn't matter for Q-value (masked by 1-done).
        # But we must ensure we don't train on (s_T, a_T, r_T, s_{0_new}, done) where s_{0_new} belongs to next trajectory
        # if the algorithm relies on s'.
        # With done=1, term in Bellman eq zeroes out V(s'), so s' value is ignored.
        # BUT, if it's a TIMEOUT (truncation), done=0 but we shouldn't bootstrap from next episode start.
        # D4RL has 'timeouts'.
        
        valid_mask = np.ones(N, dtype=bool)
        
        # Mark steps where i is end of trajectory (timeout or terminal)
        # If timeout[i] is True, then i is last step. i+1 is new traj.
        # We should probably NOT use the transition (s_i, ..., s_{i+1}) if it's a timeout?
        # Or we treat it as done=0 but mask it?
        # Standard: keep it, but ensure next_state is handled?
        # Actually, simpler approach:
        # Use the computed trajectories from before to be safe.
        
        self.trajectories = compute_trajectories(obs, act, rew, term, time)
        
        # Rebuild flat arrays from trajectories to ensure correctness
        s_list, a_list, r_list, ns_list, d_list = [], [], [], [], []
        
        for traj in self.trajectories:
            t_s = traj['observations']
            t_a = traj['actions']
            t_r = traj['rewards']
            t_d = traj['dones']
            L = len(t_s)
            
            # For each step t in 0..L-1
            # Next state:
            # If t < L-1: s[t+1]
            # If t == L-1:
            #   If done=True, s' is terminal (unknown/irrelevant). We can use s[t].
            #   If done=False (timeout), s' is unknown (truncated).
            
            # Normalize traj states first
            t_s_norm = (t_s - self.state_mean) / self.state_std
            
            # Transitions 0 to L-2
            if L > 1:
                s_list.append(t_s_norm[:-1])
                a_list.append(t_a[:-1])
                r_list.append(t_r[:-1])
                ns_list.append(t_s_norm[1:])
                d_list.append(t_d[:-1])
                
            # Last transition L-1
            s_list.append(t_s_norm[-1].reshape(1, -1))
            a_list.append(t_a[-1].reshape(1, -1))
            r_list.append(t_r[-1].reshape(1))
            # Next state for last step: duplicate current state (safe if done=1)
            ns_list.append(t_s_norm[-1].reshape(1, -1))
            d_list.append(t_d[-1].reshape(1))
            
        self.states = np.concatenate(s_list, axis=0).astype(np.float32)
        self.actions = np.concatenate(a_list, axis=0).astype(np.float32)
        self.rewards = np.concatenate(r_list, axis=0).astype(np.float32).reshape(-1, 1)
        self.next_states = np.concatenate(ns_list, axis=0).astype(np.float32)
        self.dones = np.concatenate(d_list, axis=0).astype(np.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return {
            "states": torch.from_numpy(self.states[idx]),
            "actions": torch.from_numpy(self.actions[idx]),
            "rewards": torch.from_numpy(self.rewards[idx]),
            "next_states": torch.from_numpy(self.next_states[idx]),
            "dones": torch.from_numpy(self.dones[idx])
        }