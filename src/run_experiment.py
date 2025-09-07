import os
import sys
from pathlib import Path
import random
import numpy as np
import torch
import inspect
import argparse
from datetime import datetime
from types import SimpleNamespace
from pathlib import Path
import sys

if not hasattr(np, "float_"):
    np.float_ = np.float64

THIS_FILE = Path(__file__).resolve()
SRC_ROOT  = THIS_FILE.parent                  # .../repo/src
REPO_ROOT = SRC_ROOT.parent                   # .../repo

# 1. Ensure src/ is first (for your own code: utils, training, configs, etc.)
src_str = str(SRC_ROOT)
if sys.path[0] != src_str:
    sys.path = [p for p in sys.path if p != src_str]
    sys.path.insert(0, src_str)

# 2. Add external DSF gym/ so `import models.*` resolves there
ext_dsf = (REPO_ROOT / "external" / "DecisionSpikeFormer" / "gym").resolve()
ext_str = str(ext_dsf)
if ext_str not in sys.path:
    sys.path.insert(0, ext_str)

# 3. Avoid top-level repo_root/models (clashes with external/models)
if (REPO_ROOT / "models").exists():
    print("[WARN] Top-level 'models/' exists in repo_root. This may shadow external DSF models. IGNORING it.")

# --- Debugging info
print("[INFO] sys.path head:", sys.path[:4])
print("[INFO] external DSF models dir exists:", (ext_dsf / "models").exists())

# ---------------------------------------------------------------------
# Import DSF models from external repo
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Import DSF models (prefer external, fallback to src/dsf_models)
# ---------------------------------------------------------------------
try:
    # External DSF (if clean UTF-8 files)
    from models.decision_spikeformer_pssa import SpikeDecisionTransformer as PSSADecisionSpikeFormer
    from models.decision_spikeformer_tssa import SpikeDecisionTransformer as TSSADecisionSpikeFormer
    from models.decision_transformer import DecisionTransformer
    print("[INFO] Using DSF models from external/DecisionSpikeFormer")
except SyntaxError as e:
    print("[WARN] External DSF models corrupted (null bytes). Falling back to src/models/dsf_models.")
    from models.dsf_models.decision_spikeformer_pssa import SpikeDecisionTransformer as PSSADecisionSpikeFormer
    from models.dsf_models.decision_spikeformer_tssa import SpikeDecisionTransformer as TSSADecisionSpikeFormer
    from models.dsf_models.decision_transformer import DecisionTransformer

# Aliases for consistency
DecisionSpikeFormer     = PSSADecisionSpikeFormer
SNNDecisionTransformer  = PSSADecisionSpikeFormer
# ---------------------------------------------------------------------
# Import YOUR utils/training code from src/
# ---------------------------------------------------------------------
from data_utils import dsf_collect_trajectories
from train_utils import train_model, evaluate_model
# try:
#     from utils.data_utils import dsf_collect_trajectories
#     from utils.train_utils import train_model, evaluate_model
# except ImportError as e:
#     raise ImportError(
#         f"[ERROR] Could not import from src/utils/. "
#         f"Check that src/utils/data_utils.py and src/utils/train_utils.py exist. Original error: {e}"
#     )

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
    print(f"[INFO] Collecting shared dataset for {env_name} (steps={offline_steps}, max_len={max_length})...")
    trajectories, act_dim = dsf_collect_trajectories(env_name, offline_steps, max_length)
    return trajectories, act_dim

# ---------------------------------------------------------------------
# Smart model builder: inspects __init__ and supplies sensible defaults.
# If model requires a `config` argument, we populate a comprehensive config
# with the fields observed in your DSF implementation.
# ---------------------------------------------------------------------
def build_model_from_class(cls, state_dim, act_dim, args):
    if cls is None:
        raise RuntimeError("Requested model class is None.")

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
    if 'n_embd' in param_names:   # DSF expects this spelling
        cand['n_embd'] = getattr(args, "embed_dim", 128)

    # --- transformer depth ---
    if 'n_layer' in param_names:
        cand['n_layer'] = getattr(args, "n_layer", 3)
    if 'n_head' in param_names:
        cand['n_head'] = getattr(args, "n_head", 1)

    # --- context length / sequence length ---
    if 'max_length' in param_names:
        cand['max_length'] = getattr(args, "max_length", 50)
    if 'max_ep_len' in param_names and 'max_length' not in cand:
        cand['max_ep_len'] = getattr(args, "max_length", 50)
    if 'ctx_len' in param_names:  # DSF expects this spelling
        cand['ctx_len'] = getattr(args, "max_length", 50)

    # --- if config style constructor is used ---
    if 'config' in param_names:
        cfg_defaults = dict(
            state_dim=int(state_dim),
            act_dim=int(act_dim),
            n_embd=int(getattr(args, "embed_dim", 128)),
            n_head=int(getattr(args, "n_head", 1)),
            n_layer=int(getattr(args, "n_layer", 3)),
            ctx_len=int(getattr(args, "max_length", 50)),
            n_positions=int(getattr(args, "max_length", 50)),

            T=int(getattr(args, "T", 4)),
            attn_type=int(getattr(args, "attn_type", 3)),
            window_size=int(getattr(args, "window_size", 8)),
            norm_type=int(getattr(args, "norm_type", 1)),

            num_training_steps=int(getattr(args, "num_training_steps", 1000)),
            warmup_ratio=float(getattr(args, "warmup_ratio", 0.1)),
            lr=float(getattr(args, "learning_rate", 1e-4)),
            learning_rate=float(getattr(args, "learning_rate", 1e-4)),
            weight_decay=float(getattr(args, "weight_decay", 1e-2)),
            batch_size=int(getattr(args, "batch_size", 64)),
            dropout=float(getattr(args, "dropout", 0.1)),

            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        print("Building config for", cls.__name__, "with keys:", sorted(cfg_defaults.keys()))
        return cls(SimpleNamespace(**cfg_defaults))

    # --- try kwargs directly ---
    try:
        return cls(**cand)
    except Exception as e_kw:
        # fallback: try positional args
        pos_args = []
        if 'state_dim' in param_names or 'obs_dim' in param_names:
            pos_args.append(state_dim)
        if 'act_dim' in param_names or 'action_dim' in param_names:
            pos_args.append(act_dim)
        if 'n_embd' in param_names or 'hidden_size' in param_names or 'embed_dim' in param_names:
            pos_args.append(getattr(args, "embed_dim", 128))
        if 'n_layer' in param_names:
            pos_args.append(getattr(args, "n_layer", 3))
        if 'n_head' in param_names:
            pos_args.append(getattr(args, "n_head", 1))
        if 'ctx_len' in param_names or 'max_length' in param_names or 'max_ep_len' in param_names:
            pos_args.append(getattr(args, "max_length", 50))
        try:
            return cls(*pos_args)
        except Exception as e_pos:
            raise RuntimeError(
                f"Failed to construct {cls.__name__}.\n"
                f"Attempted kwargs: {cand}\n"
                f"kwargs error: {e_kw}\n"
                f"Attempted positional args: {pos_args}\n"
                f"positional error: {e_pos}\n"
                "Please inspect the model __init__ signature and extend the runner's defaults if necessary."
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
    if args.model_type in ["snn-dt", "dsf"]:
        # snn-dt is effectively an alias for dsf; both use the same underlying model
        model = build_model_from_class(DecisionSpikeFormer, state_dim, act_dim, args)
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    # call your training utils (which should accept model, trajectories, args, log_dir)
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
        print(f"[WARN] Failed to save model.state_dict(): {e}")

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

    # optional fine tuning of defaults used for cfg construction
    parser.add_argument("--n_head", type=int, default=4, help="Default # attention heads for cfg (if needed)")
    parser.add_argument("--n_layer", type=int, default=2, help="Default # transformer layers for cfg (if needed)")
    parser.add_argument("--num_training_steps", type=int, default=1000,
                        help="Default total training steps reported to model config")

    parser.add_argument("--log_dir", "--log-dir", dest="log_dir", type=str, default="./logs")

    args = parser.parse_args()

    # iterate seeds
    for seed in range(args.seed, args.seed + args.num_seeds):
        args.seed = seed
        run_experiment(args)

if __name__ == "__main__":
    main()
