import subprocess
import sys
import argparse
from pathlib import Path

# Configuration
VARIANTS = ["no_plasticity", "no_routing", "no_phase", "dt", "snn_dt", "iql", "cql"]
ENVS = ["CartPole-v1", "Acrobot-v1", "Pendulum-v1", "MountainCar-v0"]
SEEDS = [0, 1, 2, 3, 4]
CONTRACT = "experiment_contract_light.yaml"

def main():
    parser = argparse.ArgumentParser(description="Run ablation studies.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    run_script = project_root / "ablation_studies" / "run_experiment.py"

    print(f"--- Starting Ablation Studies ---")
    print(f"Variants: {VARIANTS}")
    print(f"Environments: {ENVS}")
    print(f"Seeds: {SEEDS}")
    print(f"Contract: {CONTRACT}")
    print(f"---------------------------------")

    total_jobs = len(VARIANTS) * len(ENVS) * len(SEEDS)
    current_job = 0

    for variant in VARIANTS:
        for env in ENVS:
            for seed in SEEDS:
                current_job += 1
                cmd = [
                    sys.executable,
                    str(run_script),
                    "--variant", variant,
                    "--env", env,
                    "--seed", str(seed),
                    "--contract", CONTRACT
                ]
                
                print(f"[{current_job}/{total_jobs}] Running: Variant={variant}, Env={env}, Seed={seed}")
                
                if args.dry_run:
                    print(f"Command: {' '.join(cmd)}")
                else:
                    try:
                        subprocess.run(cmd, check=True)
                    except subprocess.CalledProcessError as e:
                        print(f"Error running job: {e}")
                        # Depending on preference, we might want to continue or stop. 
                        # For now, let's continue to the next one but log the error.
                        print("Continuing to next job...")

    print("--- All targeted experimental runs complete! ---")

if __name__ == "__main__":
    main()
