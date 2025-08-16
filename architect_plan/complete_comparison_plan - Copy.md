# Complete Plan: SNN-DT vs DSF-DT Comparison on CartPole-v1

## Overview
This document provides a comprehensive plan for comparing SNN-DT (Spiking Neural Network Decision Transformer) and DSF-DT (Decision SpikeFormer Decision Transformer) on the CartPole-v1 environment. The comparison will evaluate both models' performance, efficiency, and spiking characteristics using matched hyperparameters.

## 1. Environment Setup

### 1.1 Dependencies
- Python 3.7+
- PyTorch 1.8+
- Gym
- NumPy
- Pandas
- Transformers library

### 1.2 Hardware Requirements
- CUDA-capable GPU (recommended)
- At least 8GB RAM
- 10GB free disk space

## 2. Data Collection

### 2.1 Shared Dataset
Both models will use the same offline dataset collected from CartPole-v1:
- 5000 steps of random policy data
- Saved to `offline_data_CartPole-v1_dsf.pkl`

### 2.2 Data Collection Command
```bash
python src/train_dsf_dt.py --collect-only --env CartPole-v1
```

## 3. Model Implementations

### 3.1 SNN-DT (Spiking Neural Network Decision Transformer)
- Based on standard Decision Transformer with spiking attention
- Uses LIF neurons for attention computation
- Implements rate encoding for spike generation
- File: `src/models/snn_dt_patch.py`

### 3.2 DSF-DT (Decision SpikeFormer Decision Transformer)
- Based on Decision SpikeFormer architecture
- Uses spiking transformer blocks with positional encoding
- Implements PTNorm for normalization
- File: `src/models/dsf_dt.py`

## 4. Hyperparameter Configuration

### 4.1 Shared Hyperparameters
| Parameter | Value |
|----------|-------|
| Environment | CartPole-v1 |
| Sequence Length | 20 |
| Batch Size | 64 |
| Training Epochs | 10 |
| Offline Steps | 5000 |
| Learning Rate | 1e-4 |
| Discount Factor (gamma) | 0.99 |

### 4.2 Model-Specific Hyperparameters
| Parameter | SNN-DT | DSF-DT |
|----------|--------|--------|
| Hidden Size | 128 | 128 |
| Layers | 2 | 2 |
| Heads | 1 | 1 |
| Time Steps | 5 | 4 |
| Norm Type | Standard LayerNorm | PTNorm |

## 5. Training Procedure

### 5.1 DSF-DT Training
```bash
python src/train_dsf_dt.py --env CartPole-v1
```
- Checkpoints saved to `checkpoints/offline_dsf_CartPole-v1_{epoch}.pt`

### 5.2 SNN-DT Training
```bash
python src/train_snn_dt.py --env CartPole-v1
```
- Checkpoints saved to `checkpoints/offline_dt_CartPole-v1_{epoch}.pt`

## 6. Evaluation Protocol

### 6.1 Evaluation Commands
```bash
# Evaluate DSF-DT
python src/run_benchmark.py --model dsf-dt --env CartPole-v1 --seed 42

# Evaluate SNN-DT
python src/run_benchmark.py --model snn-dt --env CartPole-v1 --seed 42
```

### 6.2 Metrics Collection
1. Average Return (mean total reward across 10 episodes)
2. Standard Deviation of Returns
3. Average Inference Latency (ms per inference step)
4. Average Spikes per Episode
5. Total Parameters
6. Final Training Loss

## 7. Results Analysis

### 7.1 Performance Comparison
- Compare average returns and standard deviations
- Analyze return distributions

### 7.2 Efficiency Comparison
- Compare inference latencies
- Analyze parameter efficiency

### 7.3 Spike Analysis
- Compare spike counts and patterns
- Analyze spiking efficiency

## 8. Expected Outcomes

### 8.1 Primary Deliverables
1. Trained checkpoints for both models
2. Benchmark results in CSV format
3. Comparison report with visualizations
4. Reproducible results with documented metrics

### 8.2 Timeline
1. Environment setup: 1 hour
2. Data collection: 1 hour
3. DSF-DT training: 4 hours
4. SNN-DT training: 4 hours
5. Evaluation: 2 hours
6. Reporting: 2 hours

Total estimated time: 14 hours

## 9. Risk Mitigation

### 9.1 Common Issues
1. **Training Instability**: Reduce learning rate or epochs
2. **Poor Convergence**: Verify hyperparameter matching
3. **Evaluation Failures**: Check checkpoint loading
4. **Performance Issues**: Verify data collection and preprocessing

### 9.2 Debugging Steps
1. Verify all dependencies are installed
2. Confirm environment is correctly set up
3. Check hyperparameter matching between models
4. Ensure dataset is correctly formatted

## 10. Reproducibility

### 10.1 Code
All implementation files are in the repository:
- `src/models/snn_dt_patch.py`
- `src/models/dsf_dt.py`
- `src/train_snn_dt.py`
- `src/train_dsf_dt.py`
- `src/run_benchmark.py`

### 10.2 Configuration
- `src/config.py` contains default hyperparameters
- Model-specific configurations in training scripts

### 10.3 Results
- Training logs will be generated during training
- Evaluation results will be saved to `benchmark_results.csv`
- Final comparison report will be generated