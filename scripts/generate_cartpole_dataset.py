import gym
import numpy as np
import argparse
import os
import pickle
from tqdm import trange

def collect(env_name='CartPole-v1', n_episodes=200, max_steps=500, output='data/cartpole_random.pkl'):
    env = gym.make(env_name)
    trajectories = []
    for ep in trange(n_episodes):
        obs = env.reset()
        traj = {'obs': [], 'acts': [], 'rews': [], 'dones': []}
        for t in range(max_steps):
            a = env.action_space.sample()  # random policy; replace with expert if desired
            next_obs, r, done, info = env.step(a)
            traj['obs'].append(obs)
            traj['acts'].append(a)
            traj['rews'].append(r)
            traj['dones'].append(done)
            obs = next_obs
            if done:
                break
        trajectories.append(traj)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, 'wb') as f:
        pickle.dump(trajectories, f)
    print(f"Saved {len(trajectories)} episodes to {output}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='CartPole-v1')
    parser.add_argument('--n', type=int, default=200)
    parser.add_argument('--max-steps', type=int, default=500)
    parser.add_argument('--out', default='data/cartpole_random.pkl')
    args = parser.parse_args()
    collect(args.env, args.n, args.max_steps, args.out)
