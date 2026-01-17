import os
import glob
import subprocess
import time
import yaml
from pathlib import Path
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger()

CONFIG_DIR = Path("configs/phase0")
RESULTS_DIR = Path("results/phase0")
DATA_DIR = Path("data/d4rl")

def get_configs():
    return sorted(list(CONFIG_DIR.glob("*.yaml")))

def is_run_complete(save_dir):
    # Check for metrics.csv or a specific completion flag
    return (save_dir / "metrics.csv").exists()

def main():
    configs = get_configs()
    logger.info(f"Found {len(configs)} experiments to run.")
    
    # Verify Data Exists
    # We expect data/d4rl/{env}/dataset_v1.npz
    # We can check the first config to see what env it needs, but roughly:
    if not DATA_DIR.exists():
        logger.error(f"Data directory {DATA_DIR} does not exist. Please run convert_d4rl.py first.")
        # We could try to run conversion here automatically?
        # Let's assume the user/previous step handled it, or alert.
        pass

    for config_path in configs:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
            
        model = cfg['model']['name']
        env = cfg['env']
        
        # Construct Save Directory
        # Structure: results/phase0/{model}/{env}/{seed}
        # Using default seed 42
        seed = 42
        save_dir = RESULTS_DIR / model / env / str(seed)
        
        if is_run_complete(save_dir):
            logger.info(f"Skipping {model} on {env} (Run Complete)")
            continue
            
        logger.info(f"Starting {model} on {env}...")
        
        # Construct Command
        cmd = [
            "python", "scripts/train.py",
            "--config", str(config_path),
            "--save-dir", str(save_dir),
            "--env", env,
            "--model", model,
            "--seed", str(seed),
            # Add dataset path explicitly if needed, but train.py infers it.
            # train.py infers: project_root / f"data/{args.env}/dataset.npz"
            # Our convert script puts it in: data/d4rl/{env}/dataset_v1.npz
            # This is a MISMATCH. We need to point train.py to the right place.
            "--dataset-path", str(DATA_DIR / env / "dataset_v1.npz")
        ]
        
        try:
            # Run Synchronously for now
            subprocess.run(cmd, check=True)
            logger.info(f"Finished {model} on {env}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed {model} on {env}: {e}")
            # Continue to next experiment?
            time.sleep(1)

if __name__ == "__main__":
    main()
