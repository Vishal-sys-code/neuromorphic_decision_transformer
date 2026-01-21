import os
import glob
import subprocess
import time
import yaml
import argparse
import concurrent.futures
from pathlib import Path
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger()

CONFIG_DIR = Path("configs/phase0")
RESULTS_DIR = Path("results/phase0")
DATA_DIR = Path("data/d4rl_raw")

def get_configs():
    return sorted(list(CONFIG_DIR.glob("*.yaml")))

def is_run_complete(save_dir):
    # Check for metrics.csv or a specific completion flag
    return (save_dir / "metrics.csv").exists()

def run_experiment(config_path, worker_id=0):
    try:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
            
        model = cfg['model']['name']
        env = cfg['env']
        
        # Loop over seeds
        seeds = [0, 1, 2, 3, 4]
        
        for seed in seeds:
            save_dir = RESULTS_DIR / model / env / str(seed)
            
            if is_run_complete(save_dir):
                logger.info(f"Skipping {model} on {env} seed {seed} (Run Complete)")
                continue
                
            logger.info(f"Worker {worker_id}: Starting {model} on {env} seed {seed}...")
            
            # Construct Command
            cmd = [
                "python", "scripts/train.py",
                "--config", str(config_path),
                "--save-dir", str(save_dir),
                "--env", env,
                "--model", model,
                "--seed", str(seed),
                "--dataset-mode", "d4rl_direct",
                "--dataset-path", str(DATA_DIR),
                "--simulator-available"
            ]
            
            # Set environment variables for this process to limit CPU usage
            env_vars = os.environ.copy()
            # Limit threads per process to avoid thrashing
            # Assuming 3 workers on a typical 12+ core machine, 4 threads each is safe.
            # If user has fewer cores, they should reduce max-workers.
            env_vars["OMP_NUM_THREADS"] = "4"
            env_vars["MKL_NUM_THREADS"] = "4"
            env_vars["TORCH_NUM_THREADS"] = "4"
            
            # Run Synchronously (within the worker thread)
            subprocess.run(cmd, check=True, env=env_vars)
            logger.info(f"Worker {worker_id}: Finished {model} on {env} seed {seed}")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Worker {worker_id}: Failed {model} on {env}: {e}")
    except Exception as e:
        logger.error(f"Worker {worker_id}: Error processing {config_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run Phase 0 Experiments")
    parser.add_argument("--max-workers", type=int, default=3, help="Number of parallel experiments to run")
    args = parser.parse_args()

    configs = get_configs()
    logger.info(f"Found {len(configs)} experiments to run.")
    
    if not DATA_DIR.exists():
        logger.error(f"Data directory {DATA_DIR} does not exist. Please run scripts/download_d4rl.py first.")
        return

    # Use ThreadPoolExecutor to run experiments in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = []
        for i, config_path in enumerate(configs):
            # i % args.max_workers is just a rough worker ID for logging
            futures.append(executor.submit(run_experiment, config_path, i % args.max_workers))
            
        # Wait for all futures to complete
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"An experiment failed with exception: {e}")

if __name__ == "__main__":
    main()