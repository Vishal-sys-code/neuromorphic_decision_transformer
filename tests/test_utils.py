# tests/test_utils.py

import numpy as np
import pytest

from src.utils.trajectory_buffer import TrajectoryBuffer
from src.utils.helpers import compute_returns_to_go

def test_trajectory_buffer_accumulation():
    buf = TrajectoryBuffer()
    # add 5 entries with distinct actions and rewards
    for i in range(5):
        state = np.full((3,), i)     # dummy 3‑dim state
        action = i
        reward = i * 0.1
        buf.add(state, action, reward)

    traj = buf.get_trajectory()
    # shapes
    assert traj['states'].shape == (5, 3)
    assert traj['actions'].shape == (5,)
    assert traj['rewards'].shape == (5,)
    # contents
    np.testing.assert_array_equal(traj['actions'], np.arange(5))
    np.testing.assert_allclose(traj['rewards'], np.arange(5) * 0.1)

def test_compute_returns_to_go_gamma_1():
    rewards = np.array([1.0, 2.0, 3.0])
    rtg = compute_returns_to_go(rewards, gamma=1.0)
    # returns-to-go with gamma=1.0: [1+2+3, 2+3, 3]
    np.testing.assert_allclose(rtg, [6.0, 5.0, 3.0])

def test_compute_returns_to_go_gamma_0_5():
    rewards = np.array([1.0, 2.0, 3.0])
    rtg = compute_returns_to_go(rewards, gamma=0.5)
    # expected: [1 + 0.5*(2 + 0.5*3), 2 + 0.5*3, 3]
    expected = np.array([1 + 0.5*(2 + 0.5*3), 2 + 0.5*3, 3.0])
    np.testing.assert_allclose(rtg, expected)

if __name__ == "__main__":
    pytest.main()