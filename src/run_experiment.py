"""
Experiment runner for SNN-DT and (optionally) DecisionSpikeFormer baseline.

Author: Vishal Pandey
Email: pandeyvishal.mlprof@gmail.com
"""

import os
import sys
import random
import inspect
import argparse
import importlib.util
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

# ---------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
SRC_ROOT = THIS_FILE.parent                  # .../repo/src
REPO_ROOT = SRC_ROOT.parent                  # .../repo

# ---------------------------------------------------------------------
# Import your SNN-DT model from src
# ---------------------------------------------------------------------
try:
    from .models.snn_dt_gpt2_attention_exp import SNNDecisionTransformer as SrcSNNDecisionTransformer
    print("[INFO] Using SNNDecisionTransformer from src.models.snn_dt_gpt2_attention")
except Exception as e:
    print(f"[ERROR] Failed to import your SNNDecisionTransformer: {e}")
    raise

# ---------------------------------------------------------------------
# Optionally import external DSF for baseline comparisons
# ---------------------------------------------------------------------
DecisionSpikeFormer = None
try:
    ext_dsf_path = REPO_ROOT / "external" / "DecisionSpikeFormer" / "gym" / "models" / "decision_spikeformer_pssa.py"
    if ext_dsf_path.exists():
        spec = importlib.util.spec_from_file_location("models.decision_spikeformer_pssa", ext_dsf_path)
        if spec and spec.loader:
            ext_dsf_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(ext_dsf_module)
            DecisionSpikeFormer = ext_dsf_module.SpikeDecisionTransformer
            print("[INFO] External DecisionSpikeFormer available for --model_type dsf")
        else:
            print("[INFO] Could not load external DecisionSpikeFormer spec.")
    else:
        print("[INFO] No external DecisionSpikeFormer found.")
except Exception:
    print("[INFO] Could not import external DSF. Only SNN-DT will be available.")

# ---------------------------------------------------------------------
# Import utils
# ---------------------------------------------------------------------
from .data_utils import dsf_collect_trajectories
from .train_utils import train_model, evaluate_model

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_log_dir(base_dir, env_name, model_type, seed):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(base_dir, f"{env_name}_{model_type}_seed{seed}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

def collect_shared_dataset(env_name, offline_steps=10000, max_length=50):
    print(f"[INFO] Collecting dataset for {env_name} (steps={offline_steps}, max_len={max_length})...")
    trajectories, act_dim = dsf_collect_trajectories(env_name, offline_steps, max_length)
    return trajectories, act_dim

# ---------------------------------------------------------------------
# Build model dynamically from __init__ signature
# ---------------------------------------------------------------------
def build_model_from_class(cls, state_dim, act_dim, args):
    """
    Smart model builder:
      - If building SNNDecisionTransformer, call with explicit positional args.
      - Otherwise, try kwargs construction, config-style, or ordered args.
    """
    # --- Case 1: Your SNNDecisionTransformer ---
    if cls.__name__ == "SNNDecisionTransformer":
        print("[INFO] Building SNNDecisionTransformer (src)")
        return cls(
            state_dim,
            act_dim,
            getattr(args, "embed_dim", 128),   # hidden_size
            n_layer=getattr(args, "n_layer", 3),
            n_head=getattr(args, "n_head", 1),
            max_length=getattr(args, "max_length", 50),
        )

    # --- Case 2: Generic DSF-style models ---
    sig = inspect.signature(cls.__init__)
    params = sig.parameters
    param_names = [p for p in params.keys() if p != 'self']

    cand = {}
    # --- common dimensions ---
    if 'state_dim' in param_names:
        cand['state_dim'] = state_dim
    if 'obs_dim' in param_names and 'state_dim' not in cand:
        cand['obs_dim'] = state_dim
    if 'act_dim' in param_names:
        cand['act_dim'] = act_dim
    if 'action_dim' in param_names and 'act_dim' not in cand:
        cand['action_dim'] = act_dim

    # --- embedding size variants ---
    if 'hidden_size' in param_names:
        cand['hidden_size'] = getattr(args, "embed_dim", 128)
    if 'embed_dim' in param_names:
        cand['embed_dim'] = getattr(args, "embed_dim", 128)
    if 'n_embd' in param_names:
        cand['n_embd'] = getattr(args, "embed_dim", 128)

    # --- transformer depth ---
    if 'n_layer' in param_names:
        cand['n_layer'] = getattr(args, "n_layer", 3)
    if 'n_head' in param_names:
        cand['n_head'] = getattr(args, "n_head", 1)

    # --- context length ---
    if 'max_length' in param_names:
        cand['max_length'] = getattr(args, "max_length", 50)
    if 'max_ep_len' in param_names and 'max_length' not in cand:
        cand['max_ep_len'] = getattr(args, "max_length", 50)
    if 'ctx_len' in param_names:
        cand['ctx_len'] = getattr(args, "max_length", 50)

    # --- config-style constructor ---
    if 'config' in param_names:
        cfg_defaults = dict(
            state_dim=int(state_dim),
            act_dim=int(act_dim),
            n_embd=int(getattr(args, "embed_dim", 128)),
            n_head=int(getattr(args, "n_head", 1)),
            n_layer=int(getattr(args, "n_layer", 3)),
            ctx_len=int(getattr(args, "max_length", 50)),
            n_positions=int(getattr(args, "max_length", 50)),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        try:
            return cls(SimpleNamespace(**cfg_defaults))
        except Exception as e_cfg:
            print(f"[WARN] Config-based init failed: {e_cfg}")

    # --- Try kwargs ---
    try:
        return cls(**cand)
    except Exception as e_kw:
        print(f"[WARN] Kwargs init failed: {e_kw}")

    # --- Try ordered positional construction ---
    try:
        ordered_param_names = [p for p in params.keys() if p != 'self']
        ordered_values = []
        for name in ordered_param_names:
            if name in ('state_dim', 'obs_dim', 'in_dim'):
                ordered_values.append(state_dim)
            elif name in ('act_dim', 'action_dim', 'out_dim'):
                ordered_values.append(act_dim)
            elif name in ('hidden_size', 'embed_dim', 'n_embd', 'd_model'):
                ordered_values.append(getattr(args, "embed_dim", 128))
            elif name in ('n_layer',):
                ordered_values.append(getattr(args, "n_layer", 3))
            elif name in ('n_head',):
                ordered_values.append(getattr(args, "n_head", 1))
            elif name in ('max_length', 'ctx_len', 'max_ep_len', 'n_positions'):
                ordered_values.append(getattr(args, "max_length", 50))
            else:
                # stop filling if unknown param appears
                break
        if ordered_values:
            return cls(*ordered_values)
    except Exception as e_ord:
        print(f"[WARN] Ordered positional init failed: {e_ord}")

    # --- Final failure ---
    raise RuntimeError(
        f"Failed to construct {cls.__name__}. "
        f"Tried kwargs={cand}, config, and ordered positional args. "
        f"Inspect the __init__ signature and extend builder if necessary."
    )

# ---------------------------------------------------------------------
# Main experiment flow
# ---------------------------------------------------------------------
def run_experiment(args):
    set_seed(args.seed)
    log_dir = make_log_dir(args.log_dir, args.env, args.model_type, args.seed)
    print(f"[INFO] Logging to {log_dir}")

    # collect dataset
    trajectories, act_dim = collect_shared_dataset(args.env, args.offline_steps, args.max_length)
    if len(trajectories) == 0:
        raise RuntimeError("No trajectories collected!")

    state_dim = len(trajectories[0]['observations'][0])

    # build model
    if args.model_type == "snn-dt":
        print("[INFO] Building SNNDecisionTransformer (src)")
        model = build_model_from_class(SrcSNNDecisionTransformer, state_dim, act_dim, args)
    elif args.model_type == "dsf":
        if DecisionSpikeFormer is None:
            raise RuntimeError("Requested DSF baseline but external not available.")
        print("[INFO] Building DecisionSpikeFormer (external)")
        model = build_model_from_class(DecisionSpikeFormer, state_dim, act_dim, args)
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    # train
    train_model(model, trajectories, args, log_dir)

    # evaluate
    eval_metrics = evaluate_model(model, args.env, args.max_length)
    print(f"[RESULT] {args.env} Seed {args.seed} Eval: {eval_metrics}")

    # save checkpoint
    ckpt_path = os.path.join(log_dir, f"{args.env}_{args.model_type}_seed{args.seed}.pt")
    try:
        torch.save(model.state_dict(), ckpt_path)
        print(f"[INFO] Saved checkpoint to {ckpt_path}")
    except Exception as e:
        print(f"[WARN] Failed to save model: {e}")

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--env", "--env-name", dest="env", type=str, default="CartPole-v1",
                        help="Gym environment id")
    parser.add_argument("--model_type", "--model-type", dest="model_type",
                        choices=["snn-dt", "dsf"], default="snn-dt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_seeds", "--num-seeds", dest="num_seeds", type=int, default=1)
    parser.add_argument("--offline_steps", "--offline-steps", dest="offline_steps", type=int, default=10000)
    parser.add_argument("--max_length", "--max-length", dest="max_length", type=int, default=50)

    parser.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int, default=64)
    parser.add_argument("--embed_dim", "--embed-dim", dest="embed_dim", type=int, default=128)
    parser.add_argument("--max_iters", "--max-iters", dest="max_iters", type=int, default=10)
    parser.add_argument("--learning_rate", "--learning-rate", dest="learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_steps_per_iter", "--num-steps-per-iter", dest="num_steps_per_iter", type=int, default=500)

    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs per seed")

    # optional fine tuning of defaults used for cfg construction
    parser.add_argument("--n_head", type=int, default=4, help="Default # attention heads for cfg (if needed)")
    parser.add_argument("--n_layer", type=int, default=2, help="Default # transformer layers for cfg (if needed)")
    parser.add_argument("--num_training_steps", type=int, default=1000,
                        help="Default total training steps reported to model config")

    parser.add_argument("--log_dir", "--log-dir", dest="log_dir", type=str, default="./logs")

    args = parser.parse_args()

if __name__ == "__main__":
    main()