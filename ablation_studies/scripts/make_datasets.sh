#!/bin/bash
set -e

# Environments from the experiment contract
ENVS=("CartPole-v1" "Acrobot-v1" "Pendulum-v1" "MountainCar-v0")

# Get the project root directory (the parent of the directory where this script is)
PROJECT_ROOT=$(dirname $(dirname $(dirname "$0")))

echo "Project root: $PROJECT_ROOT"

for env in "${ENVS[@]}"; do
    echo "--- Generating dataset for $env ---"
    
    # Output directory for the dataset
    OUT_DIR="$PROJECT_ROOT/data/$env"
    mkdir -p "$OUT_DIR"
    
    # Path to the dataset generation script in the original project
    GEN_SCRIPT="$PROJECT_ROOT/snn-dt/scripts/make_dataset.py"
    
    # Check if the script exists
    if [ ! -f "$GEN_SCRIPT" ]; then
        echo "Error: Dataset generation script not found at $GEN_SCRIPT"
        exit 1
    fi
    
    # Run the dataset generation script
    python "$GEN_SCRIPT" --env "$env" --out_dir "$OUT_DIR" --num_steps 10000 --mix "expert"
    
    echo "--- Dataset for $env generated successfully in $OUT_DIR ---"
done

echo "--- All datasets generated. ---"