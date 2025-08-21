"""
baselines/dsf_runner.py
Thin wrapper so DSFormer looks like your own evaluate() interface.
"""
import os, importlib.util, numpy as np, torch, gym

# --- dynamic import of the DSF model ---
model_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "external", "DecisionSpikeFormer",
                 "gym", "models", "decision_spikeformer_pssa.py"))
spec = importlib.util.spec_from_file_location("dsf", model_path)
dsf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dsf)
DecisionSpikeFormer = dsf.DecisionSpikeFormer

# --- lightweight eval function ---
def evaluate(checkpoint_path: str, env_name: str, num_eval_episodes: int = 100):
    """
    Evaluate DSFormer on a Gym classic-control task.
    We replicate the D4RL offline setup used by the DSF paper:
    - context length = 20
    - rtg_target = 3600 for CartPole, 0 for continuous tasks
    """
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    act_dim   = (env.action_space.shape[0]
                 if hasattr(env.action_space, 'shape')
                 else env.action_space.n)
    max_ep_len = env._max_episode_steps

    model = DecisionSpikeFormer(
        state_dim=state_dim,
        act_dim=act_dim,
        max_length=20,
        max_ep_len=max_ep_len,
        hidden_size=128,
        n_layer=3,
        n_head=1,
        n_inner=4*128,
        activation_function='relu',
        n_positions=1024,
        resid_pdrop=0.1,
        attn_pdrop=0.1,
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    returns = []
    for _ in range(num_eval_episodes):
        obs, done = env.reset(), False
        rtg = 3600 if "CartPole" in env_name else 0
        context_states  = []
        context_actions = []
        context_rtgs    = []
        episode_return = 0
        step = 0
        while not done and step < max_ep_len:
            context_states.append(obs)
            context_actions.append(np.zeros(act_dim))  # placeholder
            context_rtgs.append([rtg])

            states  = torch.tensor(context_states[-20:], dtype=torch.float32).unsqueeze(0)
            actions = torch.tensor(context_actions[-20:], dtype=torch.float32).unsqueeze(0)
            rtgs    = torch.tensor(context_rtgs[-20:], dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                act_pred = model(states, actions, rtgs).squeeze(0)[-1]
            if act_dim == 1:  # continuous
                action = act_pred.numpy()
            else:             # discrete
                action = int(torch.argmax(act_pred).item())

            obs, reward, done, _ = env.step(action)
            episode_return += reward
            rtg -= reward
            step += 1
        returns.append(episode_return)
    return float(np.mean(returns)), float(np.std(returns))

def load_dsf_model(checkpoint_path: str,
                   state_dim: int,
                   act_dim: int,
                   max_ep_len: int):
    model = DecisionSpikeFormer(
        state_dim=state_dim,
        act_dim=act_dim,
        max_length=20,
        max_ep_len=max_ep_len,
        hidden_size=128,
        n_layer=3,
        n_head=1,
        n_inner=4*128,
        activation_function='relu',
        n_positions=1024,
        resid_pdrop=0.1,
        attn_pdrop=0.1,
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    return model

def evaluate(checkpoint_path: str,
             env_name: str,
             num_eval_episodes: int = 100):
    from gym.envs import ENVS
    env = ENVS[env_name]()
    state_dim = env.observation_space.shape[0]
    act_dim   = (env.action_space.shape[0]
                 if hasattr(env.action_space, 'shape')
                 else env.action_space.n)
    max_ep_len = env._max_episode_steps

    model = load_dsf_model(checkpoint_path, state_dim, act_dim, max_ep_len)
    returns, lengths = evaluate_on_env(
        model=model,
        device="cpu",
        env=env,
        context_len=20,
        rtg_target=3000,
        rtg_scale=1000,
        num_eval_episodes=num_eval_episodes,
    )
    return float(np.mean(returns)), float(np.mean(lengths))
