import argparse
import subprocess
import sys
import numpy as np
import json
from pathlib import Path

# --- Configuration ---
DEFAULT_SEEDS = 5
DEFAULT_CONTRACT = "experiment_contract_light.yaml"

def run_single_seed(variant, env, seed, contract, device=None):
    """
    Runs a single experiment seed using run_experiment.py via subprocess.
    Returns the result dictionary (or None if failed).
    """
    cmd = [
        sys.executable,
        "ablation_studies/run_experiment.py",
        "--variant", variant,
        "--env", env,
        "--seed", str(seed),
        "--contract", contract
    ]
    
    print(f"  > Starting Seed {seed}...")
    try:
        # Run the command and capture output
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Parse the output to find the final metrics (jsonl or stdout)
        # We assume run_experiment.py logs the final metrics in a way we can grab,
        # but since it saves to runs/.../metrics.jsonl, we can also read that file.
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"  !!! Error running seed {seed} !!!")
        print(e.stderr)
        return False

def get_run_metrics(variant, env, seed):
    """
    Reads the metrics.jsonl file for a specific run to get the final performance.
    """
    # Structure: runs/{variant}/seed_{seed}/{env}/metrics.jsonl
    # Note: run_experiment.py logic for run_name:
    # run_name = cfg.model.name if cfg.model.name != 'ablation_dsformer' else args.variant
    # This might need some adjustment if the variant name != model name mapping is complex.
    # Based on run_experiment.py:
    # model_name = cfg.get('model', {}).get('name', args.variant if args.variant in ['dt', 'snn_dt', 'iql', 'cql'] else 'ablation_dsformer')
    # run_name = cfg.model.name if cfg.model.name != 'ablation_dsformer' else args.variant
    
    # We will try to reconstruct the path.
    project_root = Path(__file__).parent
    
    # Determine directory name based on variant logic from run_experiment.py
    # If variant is simple, dir is variant. If dsformer, it's the variant name.
    # To be safe, we check both possible paths.
    
    possible_run_names = [variant]
    # Add mapped names if necessary, but 'snn_dt', 'iql' etc map to themselves usually unless configured otherwise.
    
    metrics_file = None
    for r_name in possible_run_names:
        p = project_root / "runs" / r_name / f"seed_{seed}" / env / "metrics.jsonl"
        if p.exists():
            metrics_file = p
            break
            
    if not metrics_file:
        # Fallback check for model-based names if variant was just a config name
        # E.g. variant 'no_plasticity' might map to model 'ablation_dsformer' -> run_name 'no_plasticity'
        # It seems consistent.
        print(f"    [Warning] Could not find metrics file for {variant} seed {seed}")
        return None

    final_return = None
    try:
        with open(metrics_file, 'r') as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                if 'val/mean_return' in data:
                    final_return = data['val/mean_return']
    except Exception as e:
        print(f"    [Error] Reading metrics file: {e}")
        
    return final_return

def main():
    parser = argparse.ArgumentParser(description="Run a group of ablation experiments (multiple seeds) and report Mean +/- Std.")
    parser.add_argument("--variant", required=True, help="Experiment variant (e.g., snn_dt, no_plasticity)")
    parser.add_argument("--env", required=True, help="Environment (e.g., CartPole-v1)")
    parser.add_argument("--num_seeds", type=int, default=DEFAULT_SEEDS, help="Number of seeds to run (0 to N-1)")
    parser.add_argument("--contract", default=DEFAULT_CONTRACT, help="Experiment contract YAML")
    
    args = parser.parse_args()
    
    print(f"\n=======================================================")
    print(f"  Running Ablation Group: {args.variant} | {args.env}")
    print(f"  Seeds: 0 to {args.num_seeds - 1}")
    print(f"=======================================================\n")
    
    returns = []
    
    for seed in range(args.num_seeds):
        success = run_single_seed(args.variant, args.env, seed, args.contract)
        if success:
            val_return = get_run_metrics(args.variant, args.env, seed)
            if val_return is not None:
                returns.append(val_return)
                print(f"  > Seed {seed} Finished. Return: {val_return:.2f}")
            else:
                print(f"  > Seed {seed} Finished but no return found.")
        else:
            print(f"  > Seed {seed} FAILED.")
            
    print(f"\n=======================================================")
    if returns:
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        print(f"  FINAL RESULT [{args.variant} / {args.env}]:")
        print(f"  Mean Return: {mean_ret:.2f} ± {std_ret:.2f}")
        print(f"  (Based on {len(returns)}/{args.num_seeds} successful runs)")
    else:
        print(f"  NO SUCCESSFUL RUNS.")
    print(f"=======================================================\n")

if __name__ == "__main__":
    main()
