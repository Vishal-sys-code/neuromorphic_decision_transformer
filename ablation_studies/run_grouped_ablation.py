import argparse
import subprocess
import sys
import numpy as np
import json
from pathlib import Path
import tempfile
import os

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
    print(f"\n{Colors.OKCYAN}---------------------------------------------------------------------------------{Colors.ENDC}")
    print(f"{Colors.OKCYAN}|{Colors.ENDC} {Colors.BOLD}Ablation Group:{Colors.ENDC} {variant:<15} | {env:<20} {Colors.OKCYAN}|{Colors.ENDC}")
    print(f"{Colors.OKCYAN}|{Colors.ENDC} {Colors.BOLD}Seeds:{Colors.ENDC}          0 to {num_seeds - 1:<3}                              {Colors.OKCYAN}|{Colors.ENDC}")
    print(f"{Colors.OKCYAN}---------------------------------------------------------------------------------{Colors.ENDC}\n")

def run_single_seed(variant, env, seed, contract):
    """
    Runs a single experiment seed using run_experiment.py via subprocess.
    Uses temp files to capture output to avoid pipe deadlocks.
    Returns (success, output_log).
    """
    cmd = [
        sys.executable,
        "ablation_studies/run_experiment.py",
        "--variant", variant,
        "--env", env,
        "--seed", str(seed),
        "--contract", contract
    ]
    
    # Create temporary files for stdout and stderr
    # This avoids buffer overflows and deadlocks on Windows with large output (e.g. from tqdm)
    with tempfile.TemporaryFile(mode='w+') as out_f, tempfile.TemporaryFile(mode='w+') as err_f:
        try:
            # Run the command
            subprocess.run(cmd, stdout=out_f, stderr=err_f, text=True, check=True)
            
            # Read output for return (rewind first)
            out_f.seek(0)
            err_f.seek(0)
            output_log = out_f.read() + "\n" + err_f.read()
            return True, output_log
            
        except subprocess.CalledProcessError as e:
            # Read output for error reporting
            out_f.seek(0)
            err_f.seek(0)
            stdout_content = out_f.read()
            stderr_content = err_f.read()
            
            # Construct error message instead of printing directly
            error_msg = []
            error_msg.append(f"  [{Colors.BOLD}SEED {seed}{Colors.ENDC}] {Colors.FAIL}x FAILED{Colors.ENDC}")
            error_msg.append(f"{Colors.FAIL}  | Error Log ---------------------------------------------------{Colors.ENDC}")
            
            # Filter stderr to remove tqdm noise (lines containing %|)
            err_lines = [line for line in stderr_content.splitlines() if "%|" not in line and "it/s" not in line]
            # Print last 20 lines of filtered error
            for line in err_lines[-20:]:
                error_msg.append(f"{Colors.FAIL}  | {line}{Colors.ENDC}")
            error_msg.append(f"{Colors.FAIL}  ---------------------------------------------------------------{Colors.ENDC}")
            
            return False, "\n".join(error_msg)

def get_run_metrics(variant, env, seed, output_log=None):
    """
    Reads the metrics.jsonl file for a specific run to get the final performance.
    If metrics file is missing/empty and output_log is provided, attempts to parse metrics from it.
    Returns tuple: (mean_return, spikes_per_inference)
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
            
    final_return = None
    final_spikes = None
    
    # Try reading from file first
    if metrics_file:
        try:
            with open(metrics_file, 'r') as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        data = json.loads(line)
                        if 'val/mean_return' in data:
                            final_return = data['val/mean_return']
                        if 'val/spikes_per_inference' in data:
                            final_spikes = data['val/spikes_per_inference']
                    except json.JSONDecodeError:
                        continue
        except Exception:
            pass
            
    # Fallback: Parse from output_log if file read failed or yielded nothing
    if (final_return is None or final_spikes is None) and output_log is not None:
        for line in output_log.splitlines():
            if "val/mean_return" in line or "val/spikes_per_inference" in line:
                try:
                    # Finds the JSON structure inside the line
                    # Usually formatted as: {"epoch": ..., "val/mean_return": ...}
                    start_idx = line.find('{')
                    end_idx = line.rfind('}')
                    if start_idx != -1 and end_idx != -1:
                        json_str = line[start_idx : end_idx + 1]
                        data = json.loads(json_str)
                        if 'val/mean_return' in data:
                            final_return = data['val/mean_return']
                        if 'val/spikes_per_inference' in data:
                            final_spikes = data['val/spikes_per_inference']
                except Exception:
                    continue
        
    return final_return, final_spikes

def main():
    parser = argparse.ArgumentParser(description="Run a group of ablation experiments (multiple seeds) and report Mean +/- Std.")
    parser.add_argument("--variant", required=True, help="Experiment variant")
    parser.add_argument("--env", required=True, help="Environment")
    parser.add_argument("--num_seeds", type=int, default=DEFAULT_SEEDS, help="Number of seeds to run")
    parser.add_argument("--start_seed", type=int, default=0, help="Starting seed index")
    parser.add_argument("--contract", default=DEFAULT_CONTRACT, help="Experiment contract YAML")
    
    parser.add_argument("--max_workers", type=int, default=5, help="Number of parallel workers (default: 5)")
    
    args = parser.parse_args()
    
    print_header(args.variant, args.env, args.num_seeds)
    
    returns = []
    spikes_list = []
    
    import concurrent.futures
    import torch

    # Adjust range to respect start_seed and num_seeds
    seeds_to_run = list(range(args.start_seed, args.start_seed + args.num_seeds))
    
    # Auto-detect single GPU environment (e.g., Colab T4) and force serial execution
    # to prevent OOM/Thrashing when trying to run multiple training jobs on one GPU.
    if args.max_workers > 1:
        if torch.cuda.is_available() and torch.cuda.device_count() == 1:
            print(f"{Colors.WARNING}Single GPU detected. Forcing max_workers=1 to prevent crashes.{Colors.ENDC}")
            args.max_workers = 1

    print(f"Starting {len(seeds_to_run)} runs with {args.max_workers} workers...\n")

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all jobs
        future_to_seed = {
            executor.submit(run_single_seed, args.variant, args.env, seed, args.contract): seed 
            for seed in seeds_to_run
        }
        
        # Process as they complete
        for future in concurrent.futures.as_completed(future_to_seed):
            seed = future_to_seed[future]
            try:
                success, output_log = future.result()
                
                if success:
                    val_return, val_spikes = get_run_metrics(args.variant, args.env, seed, output_log)
                    if val_return is not None:
                        returns.append(val_return)
                        spikes_str = f"{val_spikes:.2f}" if val_spikes is not None else "N/A"
                        if val_spikes is not None: spikes_list.append(val_spikes)
                        
                        print(f"  [{Colors.BOLD}SEED {seed}{Colors.ENDC}] {Colors.OKGREEN}+ Finished{Colors.ENDC}   Return: {Colors.BOLD}{val_return:.2f}{Colors.ENDC}   Spikes/Inf: {Colors.OKCYAN}{spikes_str}{Colors.ENDC}")
                    else:
                        print(f"  [{Colors.BOLD}SEED {seed}{Colors.ENDC}] {Colors.WARNING}? Finished{Colors.ENDC}   Return: {Colors.WARNING}Not Found{Colors.ENDC}")
                        # Print the captured output for debugging
                        print(f"{Colors.WARNING}  | Debug Output (Last 20 lines) ---------------------------------{Colors.ENDC}")
                        log_lines = [line for line in output_log.splitlines() if "%|" not in line]
                        for line in log_lines[-20:]:
                            print(f"{Colors.WARNING}  | {line}{Colors.ENDC}")
                        print(f"{Colors.WARNING}  ----------------------------------------------------------------{Colors.ENDC}")
            
                else:
                    # Failure case: output_log contains the formatted error message
                    print(output_log)

            except Exception as exc:
                print(f"  [{Colors.BOLD}SEED {seed}{Colors.ENDC}] {Colors.FAIL}Generated an exception: {exc}{Colors.ENDC}")

    print(f"\n{Colors.OKCYAN}---------------------------------------------------------------------------------{Colors.ENDC}")
    if returns:
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        mean_str = f"{mean_ret:.2f}"
        std_str = f"{std_ret:.2f}"
        
        mean_spikes = np.mean(spikes_list) if spikes_list else 0.0
        spikes_final_str = f"{mean_spikes:.2f}" if spikes_list else "N/A"
        
        print(f"  {Colors.BOLD}FINAL RESULT:{Colors.ENDC}")
        print(f"  Mean Return: {Colors.OKGREEN}{mean_str}{Colors.ENDC} +/- {Colors.OKGREEN}{std_str}{Colors.ENDC}")
        print(f"  Mean Spikes: {Colors.OKCYAN}{spikes_final_str}{Colors.ENDC}")
        print(f"  Success Rate: {len(returns)}/{args.num_seeds}")
    else:
        print(f"  {Colors.FAIL}NO SUCCESSFUL RUNS{Colors.ENDC}")
    print(f"{Colors.OKCYAN}---------------------------------------------------------------------------------{Colors.ENDC}\n")

if __name__ == "__main__":
    main()