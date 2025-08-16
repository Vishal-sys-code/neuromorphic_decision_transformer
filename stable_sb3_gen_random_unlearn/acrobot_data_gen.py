# acrobot_data_gen.py
# Put this at the very top of the file
import numpy as _np
# NumPy 2.0 removed several deprecated aliases (np.float_, np.int, np.bool, etc.)
# Restore the common ones Gym (and other libs) still use.
# Only add aliases that are missing to avoid noisy overrides.
_aliases = {
    "float_": _np.float64,
    "float": _np.float64,
    "int_": _np.int64,
    "int": _np.int64,
    "bool_": _np.bool_,
    "bool": _np.bool_,
}
for name, val in _aliases.items():
    if not hasattr(_np, name):
        setattr(_np, name, val)

# After the shim, import the rest
import os
import gym
import numpy as np
import pickle
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

# Parameters
env_id = "Acrobot-v1"
save_path = "data/acrobot_expert.pkl"
checkpoints = [10_000, 50_000, 100_000, 200_000]  # Steps to snapshot policies
episodes_per_checkpoint = 25  # Episodes to roll out per checkpoint

# ---------------------------
# Compatibility wrapper: ensure reset returns (obs, info)
# ---------------------------
class ResetToTupleWrapper(gym.Wrapper):
    """
    Compatibility wrapper to expose Gymnasium-style API on a Gym env:
      - reset(...) -> (obs, info)
      - step(action) -> (obs, reward, terminated, truncated, info)
    This makes the env safe to use inside SB3's DummyVecEnv/VecMonitor which expect
    Gymnasium signatures.
    """
    def reset(self, **kwargs):
        res = self.env.reset(**kwargs)
        if isinstance(res, tuple):
            return res  # already (obs, info)
        return res, {}

    def step(self, action):
        res = self.env.step(action)
        # Gymnasium expects (obs, reward, terminated, truncated, info)
        if isinstance(res, tuple):
            if len(res) == 5:
                # already Gymnasium-style
                return res
            if len(res) == 4:
                obs, reward, done, info = res
                terminated = bool(done)
                truncated = False
                return obs, reward, terminated, truncated, info
        # fallback: try to be robust
        next_obs = res[0]
        reward = res[1] if len(res) > 1 else 0.0
        done = bool(res[2]) if len(res) > 2 else False
        info = res[3] if len(res) > 3 else {}
        terminated = done
        truncated = False
        return next_obs, reward, terminated, truncated, info


def make_plain_env(env_id):
    """Plain gym env factory with shimmy compatibility attribute."""
    env = gym.make(env_id)
    if not hasattr(env, "render_mode"):
        setattr(env, "render_mode", None)
    return env

def make_eval_vec_env(env_id):
    def _factory():
        e = make_plain_env(env_id)
        e = ResetToTupleWrapper(e)   # <--- wrapper that returns (obs,info) and 5-tuple steps
        return e
    vec = DummyVecEnv([_factory])
    vec = VecMonitor(vec)
    return vec


# Helpers for manual rollouts (handle both Gym and Gymnasium signatures)
def safe_reset(env, **kwargs):
    """Return obs (compatible with Gym and Gymnasium)."""
    res = env.reset(**kwargs)
    if isinstance(res, tuple):
        return res[0]
    return res

def safe_step(env, action):
    """
    Normalize step result to (next_obs, reward, done, info).
    Handles Gym (4-tuple) and Gymnasium (5-tuple).
    """
    res = env.step(action)
    if isinstance(res, tuple) and len(res) == 5:
        next_obs, reward, terminated, truncated, info = res
        done = bool(terminated or truncated)
        return next_obs, reward, done, info
    if isinstance(res, tuple) and len(res) == 4:
        next_obs, reward, done, info = res
        return next_obs, reward, bool(done), info
    # fallback
    next_obs = res[0]
    reward = res[1] if len(res) > 1 else 0.0
    done = bool(res[2]) if len(res) > 2 else False
    info = res[3] if len(res) > 3 else {}
    return next_obs, reward, done, info

# ---------------------------
# Rollout env (for collecting trajectories manually)
# ---------------------------
rollout_env = make_plain_env(env_id)

# ---------------------------
# Train DQN (let SB3 create its own wrapped env by passing the env id)
# ---------------------------
model = DQN("MlpPolicy", env_id, verbose=1, learning_rate=1e-3, buffer_size=50_000)

# ---------------------------
# Collect trajectories using rollout_env
# ---------------------------
def collect_trajectories(model, n_episodes=25):
    trajs = []
    for _ in range(n_episodes):
        obs = safe_reset(rollout_env)
        states, actions, rewards = [], [], []
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            next_obs, reward, done, _ = safe_step(rollout_env, action)
            states.append(obs)
            actions.append(action)
            rewards.append(reward)
            obs = next_obs
        trajs.append({
            "observations": np.array(states, dtype=np.float32),
            "actions": np.array(actions, dtype=np.int64),
            "rewards": np.array(rewards, dtype=np.float32),
        })
    return trajs

# ---------------------------
# Training loop with evaluation using a VecEnv that returns reset (obs, info)
# ---------------------------
all_trajs = []
steps_done = 0
for step_target in checkpoints:
    model.learn(total_timesteps=step_target - steps_done)
    steps_done = step_target

    # Create evaluation vec env where reset returns (obs, info). VecMonitor ensures proper stats.
    eval_vec_env = make_eval_vec_env(env_id)
    # evaluate_policy accepts a VecEnv; it will use the vec env's reset properly now
    mean_reward, std_reward = evaluate_policy(model, eval_vec_env, n_eval_episodes=5, return_episode_rewards=False)
    eval_vec_env.close()
    print(f"[Checkpoint {step_target}] mean reward: {mean_reward} +/- {std_reward}")

    # collect rollouts using the plain rollout_env
    trajs = collect_trajectories(model, n_episodes=episodes_per_checkpoint)
    all_trajs.extend(trajs)

# ---------------------------
# Save dataset
# ---------------------------
dataset = {
    "trajectories": all_trajs,
    "metadata": {
        "env_id": env_id,
        "num_trajs": len(all_trajs),
        "avg_len": float(np.mean([len(t["rewards"]) for t in all_trajs])) if all_trajs else 0.0,
        "avg_return": float(np.mean([np.sum(t["rewards"]) for t in all_trajs])) if all_trajs else 0.0,
        "returns": [float(np.sum(t["rewards"])) for t in all_trajs]
    }
}
os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
with open(save_path, "wb") as f:
    pickle.dump(dataset, f)

print(f"Dataset saved to {save_path}")
print(f"Num trajectories: {len(all_trajs)}")

# Clean up
rollout_env.close()
try:
    model.env.close()
except Exception:
    pass

