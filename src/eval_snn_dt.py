#!/usr/bin/env python3
"""
Evaluate an SNN-DT checkpoint (returns, spikes, latency, energy).
Place this file in your repo root and run it from there.

Usage examples:
 python eval_snn_dt.py --ckpt logs/CartPole-v1_snn-dt_seed42_.../CartPole-v1_snn-dt_seed42.pt --env CartPole-v1 --episodes 50 --per_spike_pj 5.0

If your checkpoint is a state_dict, provide --model-class "module.Submodule.ClassName"
and optionally --model-kwargs "arg1=val1,arg2=val2".
"""

import os
import sys
import time
import json
import torch
import argparse
import importlib
import numpy as np
from pathlib import Path
import gym

# ============ CLI =================
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", required=True, help="Path to SNN-DT checkpoint (.pt, .pth)")
parser.add_argument("--env", required=True, help="Gym env name (e.g. CartPole-v1)")
parser.add_argument("--episodes", type=int, default=50, help="Evaluation episodes")
parser.add_argument("--per_spike_pj", type=float, default=5.0, help="pJ per spike (for energy estimate)")
parser.add_argument("--device", default=None, help="cuda or cpu (default auto)")
parser.add_argument("--model-class", default=None,
                    help="Optional: model class path if checkpoint only contains state_dict, format module.Class (e.g. mypkg.model.SNNDT)")
parser.add_argument("--model-kwargs", default=None,
                    help="Optional: comma-separated key=val pairs for model constructor if model-class used")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--render", action="store_true", help="Render env during eval (slow)")
parser.add_argument("--outdir", default="results/raw", help="Where to write JSON results")
args = parser.parse_args()

# ============ Setup =================
ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT))  # ensure repo root is visible for imports

device = None
if args.device:
    device = torch.device(args.device)
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(args.seed)
np.random.seed(args.seed)

OUTDIR = Path(args.outdir)
OUTDIR.mkdir(parents=True, exist_ok=True)

# ============ Helpers =================
def try_load_checkpoint(path):
    ckpt = None
    try:
        ckpt = torch.load(path, map_location="cpu")
    except Exception as e:
        raise RuntimeError(f"Failed to torch.load checkpoint {path}: {e}")
    return ckpt

def import_class(path_str):
    # path_str like "module.submodule.ClassName"
    if "." not in path_str:
        raise ValueError("model-class must be module.ClassName or package.module.ClassName")
    module_path, cls_name = path_str.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, cls_name)
    return cls

def parse_kwargs(s):
    if not s:
        return {}
    out = {}
    for kv in s.split(","):
        if kv.strip()=="":
            continue
        if "=" not in kv:
            raise ValueError("model-kwargs entries must be key=val")
        k,v = kv.split("=",1)
        # try to parse numeric types
        try:
            v_parsed = int(v)
        except:
            try:
                v_parsed = float(v)
            except:
                if v.lower() in ["true","false"]:
                    v_parsed = v.lower()=="true"
                else:
                    v_parsed = v
        out[k.strip()] = v_parsed
    return out

# spike tally helper (robust to tuple outputs)
class SpikeTally:
    def __init__(self):
        self.reset()
    def reset(self):
        self.total = 0
        self.calls = 0
    def add(self, out):
        # out can be Tensor, tuple/list, dict
        if out is None:
            return
        if isinstance(out, torch.Tensor):
            # treat nonzero values as spikes
            try:
                s = int((out != 0).sum().item())
            except:
                s = int(out.sum().item())
            self.total += s
            self.calls += 1
        elif isinstance(out, (list, tuple)):
            for o in out:
                self.add(o)
        elif isinstance(out, dict):
            for o in out.values():
                self.add(o)
        else:
            # ignore other types
            return
    def avg_per_call(self):
        return self.total / self.calls if self.calls>0 else 0

# attach hooks to modules whose class name suggests spiking (LIF, Spike, lif, spk)
def attach_spike_hooks(model, tally):
    hooks = []
    for name, module in model.named_modules():
        cls = module.__class__.__name__.lower()
        if ("lif" in cls) or ("spike" in cls) or ("spiking" in cls) or hasattr(module, "is_spiking"):
            # register forward hook
            def make_hook():
                def hook(m, inp, out):
                    tally.add(out)
                return hook
            hooks.append(module.register_forward_hook(make_hook()))
    return hooks

# best-effort action wrapper
def model_action(model, obs_tensor):
    """
    Heuristic to get discrete action from a model:
    - Try model.act(obs)
    - Else try model.forward(...) and postprocess (argmax)
    - Else try model(obs)
    """
    # prefer model.act
    if hasattr(model, "act"):
        try:
            with torch.no_grad():
                return model.act(obs_tensor)
        except Exception:
            pass
    # try forward
    try:
        with torch.no_grad():
            out = model(obs_tensor)
        # if tuple, assume (action_logits, ...)
        if isinstance(out, tuple) or isinstance(out, list):
            logits = out[0]
        elif isinstance(out, dict):
            if "action" in out:
                logits = out["action"]
            elif "logits" in out:
                logits = out["logits"]
            else:
                logits = next(iter(out.values()))
        else:
            logits = out
        # if logits is tensor, choose argmax or sample depending on shape
        if isinstance(logits, torch.Tensor):
            if logits.dim() == 2 and logits.size(0) == 1:
                # shape [1, n_actions]
                action = int(torch.argmax(logits[0]).item())
                return action
            elif logits.dim() == 1:
                return int(torch.argmax(logits).item())
            else:
                # fallback flatten
                return int(torch.argmax(logits.view(-1)).item())
    except Exception:
        pass
    raise RuntimeError("Unable to extract action from model: implement model.act(obs) or provide standard logits output.")

# ============ Load model =================
ckpt_path = Path(args.ckpt)
if not ckpt_path.exists():
    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

ckpt = try_load_checkpoint(str(ckpt_path))

model = None
model_device = device

# Case A: checkpoint is a full saved model object (nn.Module)
if isinstance(ckpt, torch.nn.Module):
    model = ckpt
    print("[INFO] Loaded checkpoint as nn.Module object.")
# Case B: checkpoint is a dict that contains a serialized 'model' object
elif isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], torch.nn.Module):
    model = ckpt["model"]
    print("[INFO] Found 'model' key in checkpoint and it's a module.")
# Case C: checkpoint has 'state_dict' or 'model_state_dict': need to reconstruct model
elif isinstance(ckpt, dict) and ("state_dict" in ckpt or "model_state_dict" in ckpt or "model_state" in ckpt):
    state_key = "state_dict" if "state_dict" in ckpt else ("model_state_dict" if "model_state_dict" in ckpt else "model_state")
    state_dict = ckpt[state_key]
    # user must supply --model-class to reconstruct
    if args.model_class is None:
        raise RuntimeError(
            "Checkpoint contains state_dict only. Please provide --model-class 'module.ClassName' "
            "and optionally --model-kwargs 'k=v,...' so the script can instantiate the model class and load state_dict."
        )
    cls = import_class(args.model_class)
    kwargs = parse_kwargs(args.model_kwargs)
    model = cls(**kwargs)
    model.load_state_dict(state_dict)
    print(f"[INFO] Reconstructed model {args.model_class} and loaded state dict.")
# Case D: checkpoint is dict with pickled whole model under different key names (try heuristics)
elif isinstance(ckpt, dict):
    # try common keys
    for key in ["net", "model_state", "sd", "weights", "policy"]:
        if key in ckpt and isinstance(ckpt[key], torch.nn.Module):
            model = ckpt[key]
            print(f"[INFO] Loaded model from checkpoint key '{key}'.")
            break
    if model is None and "config" in ckpt and "state_dict" in ckpt:
        # try reconstruct from config if possible - not automatic
        raise RuntimeError("Checkpoint has 'config' and 'state_dict' but reconstruction not automatic. Use --model-class.")
    if model is None:
        # last attempt: maybe the checkpoint is a simple state_dict (mapping)
        # detect mapping of tensors
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            if args.model_class is None:
                raise RuntimeError("This checkpoint looks like a raw state_dict. Provide --model-class to reconstruct the model.")
            cls = import_class(args.model_class)
            kwargs = parse_kwargs(args.model_kwargs)
            model = cls(**kwargs)
            model.load_state_dict(ckpt)
            print("[INFO] Reconstructed model from raw state_dict.")
if model is None:
    raise RuntimeError("Failed to load model from checkpoint. See --model-class option if checkpoint only contains state_dict.")

# Move model to device and eval
model.to(model_device)
model.eval()

# ============ Attach spike hooks ================
tally = SpikeTally()
hooks = attach_spike_hooks(model, tally)
print(f"[INFO] Attached {len(hooks)} spike hooks (heuristic by module name/attribute).")

# ============ Eval loop ========================
env = gym.make(args.env)
returns = []
per_step_spikes = []
per_step_latency_ms = []
total_spikes_all = 0
total_steps_all = 0

for ep in range(args.episodes):
    obs = env.reset()
    done = False
    ep_return = 0.0
    ep_steps = 0
    ep_spikes = 0
    ep_latency = 0.0

    while not done:
        # convert obs to tensor (batch size 1)
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(model_device)

        # measure latency
        start = time.perf_counter()
        try:
            # prefer model.act if present
            if hasattr(model, "act"):
                with torch.no_grad():
                    act_out = model.act(obs_t)
                # model.act may return numpy or tensor or int
                if isinstance(act_out, torch.Tensor):
                    action = int(act_out.squeeze().cpu().numpy().item())
                elif isinstance(act_out, (list, tuple)):
                    # assume first item is action
                    action = int(np.array(act_out[0]).squeeze().item())
                else:
                    action = int(act_out)
            else:
                # general forward heuristic
                with torch.no_grad():
                    out = model(obs_t)
                if isinstance(out, (tuple, list)):
                    logits = out[0]
                elif isinstance(out, dict):
                    if "action" in out: logits = out["action"]
                    elif "logits" in out: logits = out["logits"]
                    else: logits = next(iter(out.values()))
                else:
                    logits = out
                if isinstance(logits, torch.Tensor):
                    action = int(torch.argmax(logits.view(logits.shape[-1])).item()) if logits.numel()>1 else int(torch.argmax(logits).item())
                else:
                    raise RuntimeError("Cannot infer action from model forward output.")
        except Exception as e:
            # remove hooks and raise for debug
            for h in hooks: h.remove()
            raise RuntimeError(f"Error during model action computation: {e}")

        end = time.perf_counter()
        latency_ms = (end - start) * 1000.0
        ep_latency += latency_ms

        # count spikes emitted during the forward (hooks already updated tally)
        # heuristic: tally.total is cumulative across all forwards so far -> compute delta
        current_total = tally.total
        # compute spikes produced in this single forward = delta from previous total
        # we'll track total across episode by summing deltas
        # store prev_total in ep loop
        if ep_steps == 0 and 'prev_tally_total' not in locals():
            prev_tally_total = 0
        spikes_this_forward = current_total - prev_tally_total
        if spikes_this_forward < 0:
            # some modules returned non-deterministic outputs; clamp
            spikes_this_forward = 0
        ep_spikes += spikes_this_forward
        prev_tally_total = current_total

        # step env
        step_result = env.step(action)
        # gym reset/step API differences: try to be compatible
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            obs, reward, done, info = step_result

        ep_return += float(reward)
        ep_steps += 1

    if ep_steps == 0:
        avg_spikes_per_step = 0.0
        avg_latency_per_step = 0.0
    else:
        avg_spikes_per_step = float(ep_spikes) / float(ep_steps)
        avg_latency_per_step = float(ep_latency) / float(ep_steps)

    returns.append(ep_return)
    per_step_spikes.append(avg_spikes_per_step)
    per_step_latency_ms.append(avg_latency_per_step)
    total_spikes_all += ep_spikes
    total_steps_all += ep_steps

    # reset tally for next episode (we maintain prev_tally_total cumulative)
    # no reset to tally.total because hooks collect cumulative sums; prev_tally_total already set
    print(f"[EP {ep+1:03d}] return={ep_return:.2f} steps={ep_steps} spikes_per_step={avg_spikes_per_step:.2f} lat_ms={avg_latency_per_step:.2f}")

# remove hooks
for h in hooks:
    h.remove()

# aggregate results
mean_return = float(np.mean(returns))
std_return = float(np.std(returns))
mean_spikes = float(np.mean(per_step_spikes))  # spikes per step (averaged across episodes)
mean_latency = float(np.mean(per_step_latency_ms))
spikes_total = int(total_spikes_all)
steps_total = int(total_steps_all)
est_energy_nJ_per_step = (mean_spikes * args.per_spike_pj) / 1e3  # pJ -> nJ

# try to get validation loss if present in checkpoint meta
val_loss = None
if isinstance(ckpt, dict):
    for k in ["val_loss", "best_val_loss", "validation_loss"]:
        if k in ckpt:
            val_loss = float(ckpt[k])
            break

# prepare output
out = {
    "env": args.env,
    "ckpt": str(ckpt_path),
    "seed": args.seed,
    "episodes": args.episodes,
    "mean_return": mean_return,
    "std_return": std_return,
    "mean_spikes_per_step": mean_spikes,
    "mean_latency_ms_per_step": mean_latency,
    "spikes_total_all_episodes": spikes_total,
    "steps_total_all_episodes": steps_total,
    "est_energy_nJ_per_step": est_energy_nJ_per_step,
    "per_spike_pJ": args.per_spike_pj,
    "validation_loss_if_any": val_loss
}

# write JSON
fname = OUTDIR / f"{args.env.replace('-','_')}_snndt_seed{args.seed}.json"
with open(fname, "w") as f:
    json.dump(out, f, indent=2)

print("[RESULT] Wrote:", fname)
print(json.dumps(out, indent=2))
