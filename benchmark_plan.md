# SNN-DT vs DSF-DT Comparison Plan

## Overview
This plan outlines the implementation of a comparison between SNN-DT (Spiking Neural Network Decision Transformer) and DSF-DT (Decision SpikeFormer Decision Transformer) on the CartPole-v1 environment.

## Models Description

### SNN-DT (Spiking Neural Network Decision Transformer)
- Based on the standard Decision Transformer architecture with spiking attention mechanisms
- Uses LIF (Leaky Integrate-and-Fire) neurons for attention computation
- Implements rate encoding to convert continuous inputs to spike trains
- Uses spiking multi-head attention with temporal integration

### DSF-DT (Decision SpikeFormer Decision Transformer)
- Based on the Decision SpikeFormer architecture
- Uses spiking transformer blocks with positional encoding
- Implements LIF neurons in attention and feed-forward layers
- Uses PTNorm (Positional Temporal Normalization) for normalization

## Hyperparameters

### Shared Hyperparameters (from config.py):
- Environment: CartPole-v1
- Sequence length (max_length): 20
- Batch size: 64
- Discount factor (gamma): 0.99
- Learning rate: 1e-4
- Offline steps: 5000
- Training epochs: 10

### Model-specific Hyperparameters:
- Hidden size: 128
- Number of layers: 2
- Number of heads: 1
- Time window: 5 (for SNN-DT)

## Implementation Steps

### 1. Environment Setup
- Ensure all dependencies are installed
- Verify CartPole-v1 environment is available
- Set up directories for checkpoints and results

### 2. Data Collection
- Collect 5000 steps of random policy data on CartPole-v1
- Save trajectories for both models to use the same dataset

### 3. DSF-DT Training
- Train DSF-DT model on CartPole-v1 with collected data
- Use the same hyperparameters as specified above
- Save checkpoints during training

### 4. SNN-DT Training
- Train SNN-DT model on CartPole-v1 with the same collected data
- Match hyperparameters as closely as possible to DSF-DT
- Save checkpoints during training

### 5. Evaluation
For each model, evaluate on CartPole-v1 and collect metrics:
- Average return over 10 episodes
- Standard deviation of returns
- Average inference latency (ms)
- Average spikes per episode
- Validation loss during training
- Total number of parameters

### 6. Comparison and Reporting
- Create a comparison table with all metrics
- Generate plots for visual comparison
- Write a summary report of findings

## Expected Results Format

Results will be saved in the following format:

```csv
model,env,seed,avg_return,std_return,avg_latency_ms,avg_spikes_per_episode,total_params
snn-dt,CartPole-v1,42,150.2,25.3,12.5,12500,150000
dsf-dt,CartPole-v1,42,145.7,30.1,18.2,18500,165000
```

## Timeline
1. Environment setup: 1 hour
2. Data collection: 1 hour
3. DSF-DT training: 4 hours
4. SNN-DT training: 4 hours
5. Evaluation: 2 hours
6. Reporting: 2 hours

Total estimated time: 14 hours