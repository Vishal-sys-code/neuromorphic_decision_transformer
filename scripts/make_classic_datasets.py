import numpy as np, h5py, gymnasium as gym, random, os
SEED = 42
random.seed(SEED); np.random.seed(SEED)

TASKS = {
    "CartPole-v1":  {"max_rtg": 500,  "action_type": "discrete", "max_len": 500},
    "Acrobot-v1":   {"max_rtg": 0,    "action_type": "discrete", "max_len": 500},
    "Pendulum-v1":  {"max_rtg": 0,    "action_type": "continuous", "max_len": 200},
    "MountainCar-v0": {"max_rtg": 0,  "action_type": "discrete", "max_len": 200},
}

TRANSITIONS_PER_TASK = 100_000
os.makedirs("data", exist_ok=True)

for env_id, info in TASKS.items():
    env = gym.make(env_id)
    env.reset(seed=SEED)

    # containers
    states, actions, rewards, next_states, terminals, rtgs = [], [], [], [], [], []

    obs, _ = env.reset()
    ep_rewards, rtg = [], info["max_rtg"]

    for _ in range(TRANSITIONS_PER_TASK):
        action = env.action_space.sample()
        next_obs, reward, term, trunc, _ = env.step(action)
        done = term or trunc

        states.append(obs.astype(np.float32))
        actions.append(np.array(action, dtype=np.float32))
        rewards.append(float(reward))
        next_states.append(next_obs.astype(np.float32))
        terminals.append(float(done))
        rtgs.append(float(rtg))
        rtg -= reward

        if done:
            obs, _ = env.reset()
            rtg = info["max_rtg"]
        else:
            obs = next_obs

    # save
    file_name = f"data/{env_id.lower().replace('-', '_')}_medium.h5"
    with h5py.File(file_name, "w") as f:
        f.create_dataset("observations",      data=np.array(states))
        f.create_dataset("actions",           data=np.array(actions))
        f.create_dataset("rewards",           data=np.array(rewards))
        f.create_dataset("next_observations", data=np.array(next_states))
        f.create_dataset("terminals",         data=np.array(terminals))
        f.create_dataset("rtgs",              data=np.array(rtgs))
    print(f"{env_id} -> {file_name} ({len(states)} transitions)")

print("All datasets ready.")
