import numpy as np

class TrajectoryBuffer:
    def __init__(self, max_len, state_dim, action_dim):
        self.max_len = max_len
        self.reset()

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def add(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def get_trajectory(self):
        return {
            'states': np.array(self.states),
            'actions': np.array(self.actions),
            'rewards': np.array(self.rewards),
        }