import torch
import sys
from pathlib import Path
import numpy as np
import gymnasium as gym

sys.path.append(str(Path(__file__).resolve().parent.parent))

# Mock Environment
class MockDiscreteEnv:
    def __init__(self):
        self.action_space = gym.spaces.Discrete(2)
    def step(self, action):
        if not isinstance(action, (int, np.integer)):
            raise AssertionError(f"{action} ({type(action)}) invalid. Expected int.")
        return np.zeros(4), 1.0, False, False, {}

# Mock Config
class AttrDict(dict):
    __getattr__ = dict.get

cfg_scalar = AttrDict({'dataset': AttrDict({'act_dim': 1})})

print("Testing Discrete Action Fix...")

# Test Case 1: Scalar output (Regression) -> Rounding
env = MockDiscreteEnv()
action_scalar_low = np.array([0.28], dtype=np.float32)
action_scalar_high = np.array([0.8], dtype=np.float32)

print(f"Scalar Input (Low): {action_scalar_low}")
# Logic copied from run_experiment.py for independent verification
if isinstance(env.action_space, gym.spaces.Discrete):
    if cfg_scalar.dataset.act_dim == 1:
        converted_action = int(np.round(action_scalar_low.item()))
        print(f"Converted: {converted_action} (Type: {type(converted_action)})")
        try:
            env.step(converted_action)
            print("PASS: Environment accepted low scalar action.")
        except AssertionError as e:
            print(f"FAIL: {e}")

        converted_action_high = int(np.round(action_scalar_high.item()))
        print(f"Scalar Input (High): {action_scalar_high} -> Converted: {converted_action_high}")
        if converted_action_high == 1:
            print("PASS: High scalar rounded correctly to 1.")
        else:
            print("FAIL: High scalar rounding failed.")

