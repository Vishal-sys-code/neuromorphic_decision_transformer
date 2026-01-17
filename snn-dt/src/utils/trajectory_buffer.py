"""
Author: Vishal Pandey
Email: pandeyvishal.mlprof@gmail.com
"""

import numpy as np

"""
Author: Vishal Pandey
Email: pandeyvishal.mlprof@gmail.com
"""

import numpy as np

class TrajectoryBuffer:
    def __init__(self, max_len, state_dim, action_dim):
        self.max_len = max_len
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reset()

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def add(self, state, action, reward):
        # Ensure state is a 1D numpy array of state_dim
        state = np.asarray(state).flatten()
        if state.shape[0] != self.state_dim:
            raise ValueError(f"State shape mismatch. Expected {self.state_dim}, got {state.shape[0]}")
        self.states.append(state)

        # Ensure action is a 1D numpy array of action_dim
        action = np.asarray(action).flatten()
        if action.shape[0] != self.action_dim:
            raise ValueError(f"Action shape mismatch. Expected {self.action_dim}, got {action.shape[0]}")
        self.actions.append(action)

        self.rewards.append(reward)

    def get_trajectory(self):
        return {
            'states': np.array(self.states).reshape(-1, self.state_dim),
            'actions': np.array(self.actions).reshape(-1, self.action_dim),
            'rewards': np.array(self.rewards),
        }