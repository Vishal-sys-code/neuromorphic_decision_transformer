import gym
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)

class RealEnv:
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self._max_episode_steps = self.env._max_episode_steps

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)
    
    def close(self):
        self.env.close()

class ReplayEnv:
    def __init__(self, dataset_path, env_name):
        self.dataset = np.load(dataset_path)
        self.states = self.dataset['states']
        self.actions = self.dataset['actions']
        self.rewards = self.dataset['rewards']
        self.dones = self.dataset['dones']
        self.masks = self.dataset['mask']
        
        # Infer spaces from data
        state_dim = self.states.shape[2]
        act_dim = self.actions.shape[2]
        
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)
        # Assuming continuous for now, can infer from metadata if needed
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)
        self._max_episode_steps = self.states.shape[1]
        
        self.current_idx = 0
        self.t = 0
        self.env_name = env_name

    def reset(self, **kwargs):
        # Pick a random clip
        self.current_idx = np.random.randint(0, len(self.states))
        self.t = 0
        
        obs = self.states[self.current_idx, 0]
        return obs, {}

    def step(self, action):
        # We ignore the action for state transition (replay)
        # But we could log it if we wanted to compute BC loss
        
        if self.t >= self._max_episode_steps - 1:
            # End of clip
            done = True
            obs = self.states[self.current_idx, self.t] # Return last state again? Or zeros?
            reward = 0
        else:
            obs = self.states[self.current_idx, self.t + 1]
            reward = self.rewards[self.current_idx, self.t]
            done = bool(self.dones[self.current_idx, self.t])
            self.t += 1
            
        return obs, float(reward), done, False, {"replay_mode": True}

    def close(self):
        pass

def create_env(env_name, simulator_available=False, dataset_path=None):
    if simulator_available:
        try:
            return RealEnv(env_name)
        except Exception as e:
            logger.warning(f"Failed to create real environment {env_name}: {e}. Falling back to ReplayEnv.")
    
    if dataset_path is None:
        raise ValueError("dataset_path must be provided for ReplayEnv (when simulator is unavailable).")
        
    logger.info(f"Creating ReplayEnv from {dataset_path}")
    return ReplayEnv(dataset_path, env_name)
