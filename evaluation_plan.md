# Evaluation Plan: SNN-DT and DSF-DT on CartPole-v1

## Overview
This document outlines the evaluation procedure for both SNN-DT and DSF-DT models on the CartPole-v1 environment, including the metrics to be collected and the evaluation protocol.

## Evaluation Protocol

### Shared Evaluation Settings
- Environment: CartPole-v1
- Number of Evaluation Episodes: 10
- Random Seed: 42
- Same checkpoint loading procedure
- Identical observation preprocessing

## Metrics to Collect

### Primary Metrics
1. **Average Return**: Mean total reward across 10 episodes
2. **Standard Deviation of Returns**: Variance in episode returns
3. **Average Inference Latency**: Mean time (ms) per inference step
4. **Average Spikes per Episode**: Mean spike count across all neurons
5. **Validation Loss**: Final training loss (if available)
6. **Total Parameters**: Model parameter count

### Secondary Metrics
1. **Episode Length**: Mean steps per episode
2. **Success Rate**: Percentage of episodes reaching maximum reward (200 for CartPole-v1)
3. **Training Time**: Total time to train each model
4. **Memory Usage**: GPU/CPU memory during inference

## Evaluation Commands

### DSF-DT Evaluation
```bash
# Evaluate DSF-DT
python src/run_benchmark.py --model dsf-dt --env CartPole-v1 --seed 42
```

### SNN-DT Evaluation
```bash
# Evaluate SNN-DT
python src/run_benchmark.py --model snn-dt --env CartPole-v1 --seed 42
```

## Expected Results Format

Results will be saved in `benchmark_results.csv` with the following columns:

```csv
model,env,seed,avg_return,std_return,avg_latency_ms,avg_spikes_per_episode,total_params
snn-dt,CartPole-v1,42,150.2,25.3,12.5,12500,150000
dsf-dt,CartPole-v1,42,145.7,30.1,18.2,18500,165000
```

## Evaluation Implementation Details

### Return Calculation
- Sum of rewards across each episode
- Average across 10 episodes
- Standard deviation calculation

### Latency Measurement
- Time measurement around `model.get_action()` call
- Average across all inference steps
- Convert to milliseconds

### Spike Count
- For DSF-DT: Use `get_total_spike_count()` method
- For SNN-DT: Implement similar method if not available
- Reset counter between episodes
- Average across episodes

### Parameter Count
- Use `sum(p.numel() for p in model.parameters() if p.requires_grad)`

## Validation Checks

### Pre-Evaluation
1. Verify checkpoints exist for both models
2. Confirm models load without errors
3. Check that environment is available
4. Ensure evaluation script runs without errors

### Post-Evaluation
1. Verify all metrics are collected
2. Check for reasonable values (e.g., positive returns)
3. Confirm results are saved to CSV
4. Validate reproducibility with same seed

## Troubleshooting

### Common Issues
1. **Checkpoint Not Found**: Verify training completed successfully
2. **Model Loading Errors**: Check model configuration matches checkpoint
3. **Environment Issues**: Verify Gym installation and version
4. **Metric Collection Errors**: Check that all required methods are implemented

### Debugging Steps
1. Run evaluation with verbose logging
2. Test individual components (model loading, environment, metrics)
3. Verify hyperparameter matching
4. Check for hardware-specific issues (CUDA availability)