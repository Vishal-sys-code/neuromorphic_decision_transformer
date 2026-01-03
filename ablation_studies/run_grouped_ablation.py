import argparse
import subprocess
import sys
import numpy as np
import json
from pathlib import Path

# --- Configuration ---
DEFAULT_SEEDS = 5
DEFAULT_CONTRACT = "experiment_contract_light.yaml"

# --- Colors ---
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(variant, env, num_seeds):
    print(f"\n{Colors.OKCYAN}---------------------------------------------------------------{Colors.ENDC}")
    print(f"{Colors.OKCYAN}|{Colors.ENDC} {Colors.BOLD}Ablation Group:{Colors.ENDC} {variant:<15} | {env:<20} {Colors.OKCYAN}|{Colors.ENDC}")
    print(f"{Colors.OKCYAN}|{Colors.ENDC} {Colors.BOLD}Seeds:{Colors.ENDC}          0 to {num_seeds - 1:<3}                              {Colors.OKCYAN}|{Colors.ENDC}")
    print(f"{Colors.OKCYAN}---------------------------------------------------------------{Colors.ENDC}\n")

def run_single_seed(variant, env, seed, contract):
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
    
    # print(f"  > Starting Seed {seed}...", end="", flush=True) # Too noisy if we print header
    
    try:
        # Run the command and capture output
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\r  [{Colors.BOLD}SEED {seed}{Colors.ENDC}] {Colors.FAIL}x FAILED{Colors.ENDC}")
        print(f"{Colors.FAIL}  | Error Log ---------------------------------------------------{Colors.ENDC}")
        # Filter stderr to remove tqdm noise (lines containing %|)
        err_lines = [line for line in e.stderr.splitlines() if "%|" not in line and "it/s" not in line]
        # Print last 20 lines of filtered error
        for line in err_lines[-20:]:
            print(f"{Colors.FAIL}  | {line}{Colors.ENDC}")
        print(f"{Colors.FAIL}  ---------------------------------------------------------------{Colors.ENDC}")
        return False

def get_run_metrics(variant, env, seed):
    """
    Reads the metrics.jsonl file for a specific run to get the final performance.
    """
    project_root = Path(__file__).parent
    
    # Try probable paths
    possible_run_names = [variant]
    
    metrics_file = None
    for r_name in possible_run_names:
        p = project_root / "runs" / r_name / f"seed_{seed}" / env / "metrics.jsonl"
        if p.exists():
            metrics_file = p
            break
            
    if not metrics_file:
        return None

    final_return = None
    try:
        with open(metrics_file, 'r') as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                if 'val/mean_return' in data:
                    final_return = data['val/mean_return']
    except Exception:
        pass
        
    return final_return

def main():
    parser = argparse.ArgumentParser(description="Run a group of ablation experiments (multiple seeds) and report Mean +/- Std.")
    parser.add_argument("--variant", required=True, help="Experiment variant")
    parser.add_argument("--env", required=True, help="Environment")
    parser.add_argument("--num_seeds", type=int, default=DEFAULT_SEEDS, help="Number of seeds to run")
    parser.add_argument("--contract", default=DEFAULT_CONTRACT, help="Experiment contract YAML")
    
    args = parser.parse_args()
    
    print_header(args.variant, args.env, args.num_seeds)
    
    returns = []
    
    for seed in range(args.num_seeds):
        print(f"  [{Colors.BOLD}SEED {seed}{Colors.ENDC}] Running...", end="", flush=True)
        success = run_single_seed(args.variant, args.env, seed, args.contract)
        
        if success:
            val_return = get_run_metrics(args.variant, args.env, seed)
            if val_return is not None:
                returns.append(val_return)
                print(f"\r  [{Colors.BOLD}SEED {seed}{Colors.ENDC}] {Colors.OKGREEN}+ Finished{Colors.ENDC}   Return: {Colors.BOLD}{val_return:.2f}{Colors.ENDC}")
            else:
                print(f"\r  [{Colors.BOLD}SEED {seed}{Colors.ENDC}] {Colors.WARNING}? Finished{Colors.ENDC}   Return: {Colors.WARNING}Not Found{Colors.ENDC}")
        else:
             # run_single_seed prints the failure block
             pass
            
    print(f"\n{Colors.OKCYAN}-----------------------------------------------------------------{Colors.ENDC}")
    if returns:
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        mean_str = f"{mean_ret:.2f}"
        std_str = f"{std_ret:.2f}"
        
        print(f"  {Colors.BOLD}FINAL RESULT:{Colors.ENDC}")
        print(f"  Mean Return: {Colors.OKGREEN}{mean_str}{Colors.ENDC} +/- {Colors.OKGREEN}{std_str}{Colors.ENDC}")
        print(f"  Success Rate: {len(returns)}/{args.num_seeds}")
    else:
        print(f"  {Colors.FAIL}NO SUCCESSFUL RUNS{Colors.ENDC}")
    print(f"{Colors.OKCYAN}-----------------------------------------------------------------{Colors.ENDC}\n")

if __name__ == "__main__":
    main()
