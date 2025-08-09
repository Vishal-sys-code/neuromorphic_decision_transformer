# SNN-DT vs DSF-DT Comparison Implementation Summary

## Overview
This document summarizes the implementation of the comparison between SNN-DT (Spiking Neural Network Decision Transformer) and DSF-DT (Decision SpikeFormer Decision Transformer) on the CartPole-v1 environment.

## Files Created

### 1. Main Comparison Script
- **File**: `src/run_comparison.py`
- **Purpose**: Orchestrates the complete comparison workflow
- **Functionality**:
  - Collects a shared offline dataset
  - Trains both SNN-DT and DSF-DT models
  - Evaluates both models with matched hyperparameters
  - Saves results to CSV files

### 2. Visualization Script
- **File**: `src/plot_comparison.py`
- **Purpose**: Generates visualizations of comparison results
- **Functionality**:
  - Creates bar charts for return, latency, and spike comparisons
  - Plots training loss curves for both models
  - Saves visualizations as PNG file

### 3. Documentation
- **File**: `README_comparison.md`
- **Purpose**: Documents how to run the comparison
- **Content**:
  - Overview of the comparison
  - Instructions for running the comparison
  - Expected results and outputs
  - Troubleshooting guide

### 4. Execution Scripts
- **File**: `run_complete_comparison.sh`
- **Purpose**: Bash script to run the complete comparison on Unix-like systems
- **Functionality**:
  - Executes the comparison
  - Generates plots
  - Displays results

- **File**: `run_complete_comparison.bat`
- **Purpose**: Batch script to run the complete comparison on Windows systems
- **Functionality**:
  - Executes the comparison
  - Generates plots
  - Displays results

## Workflow

### 1. Data Collection
- Collects 5000 steps of random policy data on CartPole-v1
- Saves data to `shared_offline_data_CartPole-v1.pkl`

### 2. Model Training
- Trains SNN-DT using `src/train_snn_dt.py` infrastructure
- Trains DSF-DT using `src/train_dsf_dt.py` infrastructure
- Saves checkpoints to `checkpoints/` directory
- Logs training losses to `training_losses.csv`

### 3. Model Evaluation
- Evaluates both models using `src/run_benchmark.py` infrastructure
- Collects metrics:
  - Average return over 10 episodes
  - Standard deviation of returns
  - Average inference latency (ms)
  - Average spikes per episode
  - Total parameter count

### 4. Results Generation
- Saves numerical results to `comparison_results.csv`
- Generates visualizations in `comparison_results.png`

## Hyperparameters

### Shared Hyperparameters
- Environment: CartPole-v1
- Sequence length: 20
- Batch size: 64
- Training epochs: 10
- Offline steps: 5000
- Hidden size: 128
- Layers: 2
- Heads: 1
- Learning rate: 1e-4

### Model-Specific Hyperparameters
- SNN-DT: 5 spiking time steps
- DSF-DT: 4 spiking time steps

## Expected Outputs

### CSV Files
1. `comparison_results.csv`: Numerical results for both models
2. `training_losses.csv`: Training loss curves for both models

### Visualization
1. `comparison_results.png`: Comparison plots

### Checkpoints
1. `checkpoints/offline_dt_CartPole-v1_{epoch}.pt`: SNN-DT checkpoints
2. `checkpoints/offline_dsf_CartPole-v1_{epoch}.pt`: DSF-DT checkpoints

## Usage

### Running the Comparison
```bash
# Unix-like systems
./run_complete_comparison.sh

# Windows
run_complete_comparison.bat

# Direct Python execution
python src/run_comparison.py --env CartPole-v1 --seed 42
```

### Generating Plots Only
```bash
python src/plot_comparison.py
```

## Implementation Details

### Shared Dataset
Both models are trained on the exact same offline dataset to ensure a fair comparison. The dataset is collected using a random policy on CartPole-v1.

### Evaluation Protocol
Both models are evaluated using the same protocol:
- 10 evaluation episodes
- Same random seed for reproducibility
- Identical observation preprocessing
- Consistent metric collection

### Metrics Collection
The implementation collects all required metrics:
- Performance: Average return and standard deviation
- Efficiency: Inference latency
- Spiking Behavior: Average spikes per episode
- Model Complexity: Total parameter count

## Conclusion
The implementation provides a complete framework for comparing SNN-DT and DSF-DT on CartPole-v1 with matched hyperparameters. All necessary infrastructure is in place to run the comparison and generate meaningful results.