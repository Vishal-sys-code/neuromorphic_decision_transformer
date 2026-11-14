import gymnasium as gym
import numpy as np

env = gym.make("Hopper-v4")
obs, info = env.reset(seed=42)
print("Observation shape:", obs.shape)
print("Action shape:", env.action_space.shape)
print("Action bounds:", env.action_space.low)
print("Action bounds:", env.action_space.high)
for step in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(step, reward, terminated, truncated)
    if terminated or truncated:
        obs, info = env.reset()
env.close()
