#!/bin/bash
# ---
# This script runs the training process with enhanced debugging flags.
# - CUDA_LAUNCH_BLOCKING=1: Synchronizes CUDA kernels, providing clearer error messages.
# - PYTHONFAULTHANDLER=1: Enables faulthandler for better tracebacks on segfaults.
# ---
export CUDA_LAUNCH_BLOCKING=1
export PYTHONFAULTHANDLER=1

python snn-dt/scripts/train.py --model dsformer --env CartPole-v1 --config configs/dsformer_cartpole_debug.yaml --save-dir results/dsformer_debug | tee training_debug.log