#!/usr/bin/env bash
# ==============================================================================
# Script: run_all_pssa_experiments.sh
# Purpose: Execute Controlled Ablations E0-E5 to beat Decision SpikeFormer PSSA
# Targets: Hopper-medium (74.1), Hopper-medium-replay (96.3), Hopper-medium-expert (110.9)
# ==============================================================================

set -e

DEVICE="${1:-cpu}"
UPDATES="${2:-25000}"
DATASET="${3:-hopper-medium-v2}"

echo "=================================================================="
echo "Starting SNN-DT-H Ablation Suite: Beating Decision SpikeFormer PSSA"
echo "Device: $DEVICE | Updates: $UPDATES | Dataset: $DATASET"
echo "=================================================================="

# E0: Corrected Baseline (Standard Attention + LayerNorm)
echo "\n>>> [1/6] Running E0: Corrected Baseline SNN-DT..."
python offline_rl/experiments/run_pssa_ablations.py \
    --ablation e0 \
    --updates $UPDATES \
    --device $DEVICE \
    --d4rl-name $DATASET

# E1: SNN-DT + Local Spiking Attention (PSSA, window S=8)
echo "\n>>> [2/6] Running E1: SNN-DT + PSSA..."
python offline_rl/experiments/run_pssa_ablations.py \
    --ablation e1 \
    --updates $UPDATES \
    --device $DEVICE \
    --d4rl-name $DATASET

# E2: SNN-DT + Temporal Spiking Attention (TSSA, T-concatenation)
echo "\n>>> [3/6] Running E2: SNN-DT + TSSA..."
python offline_rl/experiments/run_pssa_ablations.py \
    --ablation e2 \
    --updates $UPDATES \
    --device $DEVICE \
    --d4rl-name $DATASET

# E3: Dual-Branch Uniform Averaging (PSSA + TSSA)
echo "\n>>> [4/6] Running E3: Dual-Branch Uniform..."
python offline_rl/experiments/run_pssa_ablations.py \
    --ablation e3 \
    --updates $UPDATES \
    --device $DEVICE \
    --d4rl-name $DATASET

# E4: Dual-Branch + Dendritic Routing Fusion
echo "\n>>> [5/6] Running E4: Dual-Branch + Dendritic Routing..."
python offline_rl/experiments/run_pssa_ablations.py \
    --ablation e4 \
    --updates $UPDATES \
    --device $DEVICE \
    --d4rl-name $DATASET

# E5: Full SNN-DT-H (Dual + Routing + PTBN Normalization)
echo "\n>>> [6/6] Running E5: Full SNN-DT-H with PTBN (Target: > 74.1)..."
python offline_rl/experiments/run_pssa_ablations.py \
    --ablation e5 \
    --updates $UPDATES \
    --device $DEVICE \
    --d4rl-name $DATASET

echo "\n=================================================================="
echo "All Ablation Experiments Completed!"
echo "Inspect reports in results/pssa_ablations/"
echo "=================================================================="
