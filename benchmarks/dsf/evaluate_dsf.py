#!/usr/bin/env python3
"""
Evaluate a DecisionSpikeFormer checkpoint (state_dict or whole model file).

# robust way (load model class from .py file)
python benchmarks/dsf/evaluate_dsf.py \
  --ckpt logs/cartpole_v1_medium_dsf.pt \
  --env CartPole-v1 \
  --model-path external/DecisionSpikeFormer/gym/models/decision_spikeformer_pssa.py \
  --model-class DecisionSpikeFormer \
  --episodes 50

# or (if your package is importable)
python benchmarks/dsf/evaluate_dsf.py \
  --ckpt logs/cartpole_v1_medium_dsf.pt \
  --env CartPole-v1 \
  --model-file external.DecisionSpikeFormer.gym.models.decision_spikeformer_pssa \
  --model-class DecisionSpikeFormer \
  --episodes 50
"""
import argparse, os, json, torch, importlib, importlib.util, sys, numpy as np, gym, time
from types import ModuleType
from collections import OrderedDict

import numpy as np
# restore deprecated alias names if missing (safe no-op if already present)
if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "int_"):
    np.int_ = np.int64
if not hasattr(np, "bool_"):
    np.bool_ = bool

def import_module_from_path(py_path: str, mod_name: str = "user_module"):
    py_path = os.path.abspath(py_path)
    spec = importlib.util.spec_from_file_location(mod_name, py_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def import_module_by_name(name: str):
    return importlib.import_module(name)

def infer_dims_from_env(env):
    obs_space = env.observation_space
    act_space = env.action_space
    if hasattr(obs_space, "shape") and len(obs_space.shape) > 0:
        state_dim = int(np.prod(obs_space.shape))
    else:
        # fallback
        state_dim = 1

    # For Discrete envs we use act_dim=1 (DSF code uses act_dim==1 branch).
    if isinstance(act_space, gym.spaces.Box):
        act_dim = int(np.prod(act_space.shape))
    elif isinstance(act_space, gym.spaces.Discrete):
        # DSF implementation expects an action vector dimension; training saved a scalar when actions were 1D.
        act_dim = 1
    else:
        act_dim = 1
    return state_dim, act_dim

def try_load_checkpoint(ckpt_path, map_location="cpu"):
    ckpt = torch.load(ckpt_path, map_location=map_location)
    return ckpt

def build_and_load_model(module: ModuleType, class_name: str, ckpt, env, device):
    assert hasattr(module, class_name), f"{class_name} not found in module {module.__file__}"
    cls = getattr(module, class_name)

    state_dim, act_dim = infer_dims_from_env(env)

    # Use same defaults as train_dsf.py
    model = cls(
        state_dim=state_dim,
        act_dim=act_dim,
        max_length=20,
        hidden_size=128,
        n_layer=3,
        n_head=1,
        n_inner=4*128,
        activation_function='relu',
        n_positions=1024,
        resid_pdrop=0.1,
        attn_pdrop=0.1,
    )

    # ckpt can be:
    #  - OrderedDict (state_dict)
    #  - dict with keys like 'state_dict' or 'model_state_dict' (common)
    #  - entire saved model object (rare)
    if isinstance(ckpt, OrderedDict) or (isinstance(ckpt, dict) and any(("weight" in k or "bias" in k) for k in ckpt.keys())):
        model.load_state_dict(ckpt)
    elif isinstance(ckpt, dict) and ("state_dict" in ckpt or "model_state_dict" in ckpt):
        sd = ckpt.get("state_dict", ckpt.get("model_state_dict"))
        model.load_state_dict(sd)
    else:
        # If torch.load saved a full model object (nn.Module), try to use it directly
        # but prefer to load state_dict to keep behavior consistent.
        if hasattr(ckpt, "state_dict"):
            model.load_state_dict(ckpt.state_dict())
        else:
            raise RuntimeError("Checkpoint format not recognized (neither state_dict nor serialized model).")

    model.to(device)
    model.eval()
    return model

def action_postprocess(action_pred, env):
    # action_pred: torch tensor or numpy
    act_space = env.action_space
    if isinstance(act_space, gym.spaces.Box):
        ap = action_pred
        if isinstance(ap, torch.Tensor):
            ap = ap.cpu().numpy()
        # clip to action space
        ap = np.clip(ap, act_space.low, act_space.high)
        return ap
    elif isinstance(act_space, gym.spaces.Discrete):
        # For discrete, expectation: model returns scalar in R (act_dim==1).
        # Use threshold 0 to convert to {0,1} for 2-action envs; for >2 actions,
        # this mapping likely needs a different policy (one-hot/logits).
        if isinstance(action_pred, torch.Tensor):
            action_pred = action_pred.detach().cpu().numpy()
        a = action_pred
        # a shape may be (1,) or scalar
        if np.ndim(a) == 0 or (np.ndim(a) == 1 and a.size == 1):
            val = float(np.squeeze(a))
            # threshold
            if act_space.n == 2:
                return int(val > 0.0)
            else:
                # map [-1,1] to [0, n-1]
                idx = int(np.round(((val + 1.0) / 2.0) * (act_space.n - 1)))
                idx = max(0, min(act_space.n - 1, idx))
                return idx
        else:
            # fallback: argmax
            return int(np.argmax(a))
    else:
        raise RuntimeError("Unsupported action space type for postprocessing.")

def evaluate_model(model, env_name, device, episodes=50, max_ep_len=1000):
    env = gym.make(env_name)
    results = []
    total_returns = []
    for ep in range(episodes):
        s = env.reset()
        done = False
        ep_ret = 0.0
        steps = 0

        # seed with current state and a zero action so model can pad
        state_dim = infer_dims_from_env(env)[0]
        act_dim = infer_dims_from_env(env)[1]
        # initial history: states (1, S), actions (1, A), returns_to_go (1,1), timesteps (1)
        states = [np.array(s, dtype=np.float32).reshape(-1)]
        actions = [np.zeros(act_dim, dtype=np.float32)]
        returns_to_go = [0.0]  # conservative
        timesteps = [0]

        while not done and steps < max_ep_len:
            # prepare tensors
            st = torch.tensor(np.stack(states, axis=0), dtype=torch.float32, device=device)  # [L, S]
            ac = torch.tensor(np.stack(actions, axis=0), dtype=torch.float32, device=device)  # [L, A]
            rtg = torch.tensor(np.array(returns_to_go).reshape(-1,1), dtype=torch.float32, device=device) # [L,1]
            tts = torch.tensor(np.array(timesteps, dtype=np.int64), device=device)

            # model.get_action expects (states, actions, returns_to_go, timesteps)
            with torch.no_grad():
                # DecisionSpikeFormer.get_action returns action tensor
                pred = model.get_action(st, ac, rtg, tts)
                if isinstance(pred, torch.Tensor):
                    pred = pred.detach().cpu()
            # postprocess
            action_to_env = action_postprocess(pred, env)

            obs, rew, done, info = env.step(action_to_env)
            ep_ret += float(rew)
            steps += 1

            # append to history
            states.append(np.array(obs, dtype=np.float32).reshape(-1))
            # store action in same format as model expects
            if isinstance(action_to_env, (int, np.integer)):
                actions.append(np.array([float(action_to_env)], dtype=np.float32))
            else:
                actions.append(np.array(action_to_env, dtype=np.float32).reshape(-1))
            returns_to_go.append(0.0)  # placeholder; could use dynamic RTG policy
            timesteps.append(steps)

        total_returns.append(ep_ret)
        results.append({"episode": ep, "return": float(ep_ret), "steps": steps})
        print(f"ep {ep:03d} return {ep_ret:.3f} steps {steps}")
    env.close()
    return {"episodes": episodes, "returns": total_returns, "mean_return": float(np.mean(total_returns)), "std_return": float(np.std(total_returns)), "per_episode": results}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--env", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-path", default=None, help="Filesystem path to model .py file")
    parser.add_argument("--model-file", default=None, help="Dotted module path (importable)")
    parser.add_argument("--model-class", required=True, help="class name inside module")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--out", default="logs/dsf_eval_results.json")
    args = parser.parse_args()

    device = torch.device(args.device if args.device in ("cpu","cuda") or torch.cuda.is_available() else "cpu")

    # Load module
    if args.model_path:
        module = import_module_from_path(args.model_path, mod_name="dsf_eval_module")
    elif args.model_file:
        module = import_module_by_name(args.model_file)
    else:
        raise RuntimeError("Either --model-path or --model-file must be provided (model location).")

    # Prepare env early to infer dims
    env = gym.make(args.env)

    ckpt = try_load_checkpoint(args.ckpt, map_location="cpu")
    model = build_and_load_model(module, args.model_class, ckpt, env, device)

    print("Model loaded. Running evaluation ...")
    t0 = time.time()
    metrics = evaluate_model(model, args.env, device=device, episodes=args.episodes)
    dt = time.time() - t0
    metrics["eval_time_sec"] = dt
    print(f"Done. mean_return={metrics['mean_return']:.3f} std={metrics['std_return']:.3f} time={dt:.1f}s")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved →", args.out)

if __name__ == "__main__":
    main()
