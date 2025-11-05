#!/bin/bash

# Define models and environments
MODELS=("dt" "snn_dt" "dsformer" "iql" "cql")
ENVIRONMENTS=("CartPole-v1" "Acrobot-v1" "MountainCar-v0" "Pendulum-v1")

# Base directory for all results
RESULTS_BASE_DIR="results/all_runs"

# Loop through each model and environment combination
for model in "${MODELS[@]}"; do
  for env in "${ENVIRONMENTS[@]}"; do
    echo "================================================================="
    echo "Running experiment: Model=${model}, Environment=${env}"
    echo "================================================================="

    # Define a specific save directory for this run
    SAVE_DIR="${RESULTS_BASE_DIR}/${model}_${env}"

    # Construct the command to run the training script
    COMMAND="python snn-dt/scripts/train.py --model ${model} --env \"${env}\" --save-dir ${SAVE_DIR}"

    # Execute the command
    eval ${COMMAND}

    # Check the exit code of the training script
    if [ $? -eq 0 ]; then
      echo "Experiment finished successfully: Model=${model}, Environment=${env}"
    else
      echo "Experiment failed: Model=${model}, Environment=${env}"
    fi

    echo "-----------------------------------------------------------------"
  done
done

echo "All experiments complete."