#!/bin/bash
set -e

# Setup
export PYTHONPATH=$PYTHONPATH:$(pwd)/snn-dt:$(pwd)

echo "=== Phase 0 Smoke Test ==="

# 1. Download
echo "[1/3] Downloading Data (mocking if offline, but trying real)..."
python scripts/download_d4rl.py --env hopper-medium-v2

# 2. Convert
echo "[2/3] Converting Data..."
python scripts/convert_d4rl.py --raw-dir data/d4rl_raw --out-dir data/d4rl --clip-len 20

# 3. Train Models
ENV="hopper-medium-v2"
DATASET="data/d4rl/hopper-medium-v2/dataset_v1.npz"
CONFIG="configs/mujoco/hopper_light.yaml"

echo "[3/3] Training Models on $ENV..."

# DT
echo "Running DT..."
python snn-dt/scripts/train.py \
    --model dt \
    --env $ENV \
    --dataset-path $DATASET \
    --config $CONFIG \
    --save-dir results/smoke_test/dt \
    --seed 0

# SNN-DT
echo "Running SNN-DT..."
python snn-dt/scripts/train.py \
    --model snn_dt \
    --env $ENV \
    --dataset-path $DATASET \
    --config $CONFIG \
    --save-dir results/smoke_test/snn_dt \
    --seed 0

# IQL
echo "Running IQL..."
python snn-dt/scripts/train.py \
    --model iql \
    --env $ENV \
    --dataset-path $DATASET \
    --config $CONFIG \
    --save-dir results/smoke_test/iql \
    --seed 0

echo "=== Smoke Test Complete ==="
if [ -f "runs/manifest.csv" ]; then
    echo "Manifest found:"
    cat runs/manifest.csv
else
    echo "ERROR: Manifest not found!"
    exit 1
fi
